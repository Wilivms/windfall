"""Windfall's agent team.

  Forecaster  finds past days that look like tomorrow (pgvector) and forecasts the price/carbon profile
  Planner     builds tomorrow's schedule for every asset (engine.py) and has Claude explain it
  Watcher     consumes the Kafka telemetry stream, detects anomalies and replans in real time
  Learner     re-estimates each cold room's physics every night (engine.learn_tau)
  Auditor     writes every decision to OpenSearch and drafts the CSRD-style carbon note
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
import time

import httpx
import numpy as np

from . import engine
from .data import day_vector, day_windows

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")


# ------------------------------------------------------------------ Claude + Keenable


def claude_json(system: str, user: str, schema: dict, name: str = "answer", max_tokens: int = 600) -> dict | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key, timeout=25)
        r = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=max_tokens, system=system,
            tools=[{"name": name, "description": "Return the structured answer. Always call this tool exactly once.", "input_schema": schema}],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user + f"\n\nAnswer by calling the `{name}` tool. If you cannot call it, reply with only the JSON object matching its schema."}])
        for b in r.content:
            if b.type == "tool_use":
                return dict(b.input)
        import re
        txt = "".join((getattr(b, "text", None) or "") for b in r.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            return json.loads(m.group(0))
    except Exception as e:  # noqa: BLE001
        print(f"[claude] {e}")
    return None


def keenable_news(query: str, n: int = 4) -> list[dict]:
    key = os.environ.get("KEENABLE_API_KEY")
    if not key:
        return []
    try:
        r = httpx.post("https://api.keenable.ai/v1/search", headers={"X-API-Key": key, "Content-Type": "application/json"},
                       json={"query": query}, timeout=15).json()
        items = r.get("results") or r.get("data") or r.get("items") or (r if isinstance(r, list) else [])
        out = []
        for it in items[:n]:
            out.append({"title": it.get("title") or it.get("name") or "", "url": it.get("url") or it.get("link") or "",
                        "snippet": (it.get("snippet") or it.get("description") or it.get("content") or "")[:240]})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[keenable] {e}")
        return []


# ------------------------------------------------------------------ the hub that owns state


class Hub:
    def __init__(self, pg, ch, vk, kf, os_) -> None:
        self.pg, self.ch, self.vk, self.kf, self.os = pg, ch, vk, kf, os_
        self.rows: list[dict] = []
        self.source = "loading"
        self.backtest: dict = {}
        self.forecast_eval: dict = {}
        self.seen_hour = None
        self.site = engine.Site()
        self.ready = False
        self.progress = "starting"
        self.listeners: list = []
        self.listeners_lock = threading.Lock()
        self.live = {"running": False}
        self.briefing_cache: dict | None = None
        self.counters = {"kafka_messages": 0, "telemetry_rows": 0, "decisions": 0}

    # ---------------- event fan-out (Valkey pub/sub when live, in-process otherwise)
    def emit(self, event: dict) -> None:
        event.setdefault("at", time.time())
        # the dashboard is fed in-process; every event is also published on the Valkey bus (windfall:events)
        self._fanout(event)
        if self.vk.live:
            self.vk.publish(event)

    def _fanout(self, event: dict) -> None:
        with self.listeners_lock:
            for q in list(self.listeners):
                try:
                    q.put_nowait(event)
                except Exception:  # noqa: BLE001
                    pass

    def start_valkey_bridge(self) -> None:
        return  # single replica: events are fanned out in-process (see emit)
        if not self.vk.live:
            return
        def loop() -> None:
            while True:
                try:
                    ps = self.vk.r.pubsub()
                    ps.subscribe("windfall:events")
                    for m in ps.listen():
                        if m.get("type") == "message":
                            self._fanout(json.loads(m["data"]))
                except Exception as e:  # noqa: BLE001
                    print(f"[bridge] {e}")
                    time.sleep(2)
        threading.Thread(target=loop, daemon=True).start()

    def audit(self, agent: str, kind: str, summary: str, explanation: str = "", asset: str = "", eur: float = 0.0, co2: float = 0.0, payload: dict | None = None) -> dict:
        doc = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "agent": agent, "kind": kind, "asset": asset,
               "summary": summary, "explanation": explanation, "eur": round(eur, 2), "co2_kg": round(co2, 2)}
        self.os.index(doc)
        self.pg.log_decision(agent, kind, summary, {**doc, **(payload or {})})
        self.kf.send("agent.decisions", doc)
        self.counters["decisions"] += 1
        self.emit({"type": "decision", **doc})
        return doc

    # ---------------- boot: data → ClickHouse → backtest (Learner) → memory (pgvector) → forecaster eval
    def boot(self, load_market) -> None:
        try:
            self.progress = "loading real market data (Energy-Charts, Open-Meteo)"
            self.rows, self.source = load_market(92)
            self.progress = f"writing {len(self.rows)} hours to ClickHouse"
            self.ch.insert("market_hourly", [[r["ts"].replace(tzinfo=None), r["price"], r["co2"], r["temp"], r["source"]] for r in self.rows],
                           ["ts", "price", "co2", "temp", "source"])
            self.progress = "replaying 90 days: baseline vs Windfall, Learner on"
            from .backtest import run_backtest
            self.backtest = run_backtest(self.rows, 90)
            run_id = dt.datetime.now().strftime("%Y%m%d%H%M")
            self.ch.insert("backtest_daily", [[run_id, dt.date.fromisoformat(d["day"]), d["baseline_cost"], d["windfall_cost"], d["baseline_co2"],
                                               d["windfall_co2"], d["baseline_peak"], d["windfall_peak"], d["safety_overrides"], d["tau_error_pct"]]
                                              for d in self.backtest["daily"]],
                           ["run_id", "day", "baseline_cost", "windfall_cost", "baseline_co2", "windfall_co2", "baseline_peak", "windfall_peak", "safety_overrides", "tau_error_pct"])
            # the site the live demo uses has already learned from the replay
            for r in self.site.rooms:
                r.tau_hat = self.backtest["learned_tau"][r.name]["estimate"]
                self.pg.upsert_asset(self.site.name, "cold_room", r.name, r.kw, r.tau_hat, r.tau_true)
            for i in range(self.site.n_vans):
                self.pg.upsert_asset(self.site.name, "ev_van", f"Van {i + 1:02d}", self.site.van_charger_kw, None, None)
            self.progress = "building the agents' memory of past days (pgvector)"
            saved = {d["day"]: d["baseline_cost"] - d["windfall_cost"] for d in self.backtest["daily"]}
            for i in day_windows(self.rows):
                day = self.rows[i]["ts"].date()
                win = self.rows[i:i + 24]
                self.pg.remember_day(day, day_vector(self.rows, i), [w["price"] for w in win], [w["co2"] for w in win], saved.get(day.isoformat(), 0.0))
            self.progress = "evaluating the Forecaster"
            self.forecast_eval = self.evaluate_forecaster()
            self.vk.set_state("site", {"name": self.site.name, "rooms": len(self.site.rooms), "vans": self.site.n_vans, "data_source": self.source})
            self.ready = True
            self.progress = "ready"
            self.audit("Learner", "model_update", f"Learned the physics of {len(self.site.rooms)} cold rooms from 90 days",
                       "Thermal constants re-estimated nightly from real trajectories; model error "
                       f"{self.backtest['summary'].get('model_error_first_pct')}% → {self.backtest['summary'].get('model_error_last_pct')}%.")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.progress = f"boot error: {e}"

    # ---------------- Forecaster
    def analog_forecast(self, i: int, k: int = 5) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        vec = day_vector(self.rows, i)
        day = self.rows[i]["ts"].date()
        sims = self.pg.similar_days(vec, day, k)
        if not sims:  # in-memory fallback: same maths without pgvector
            cands = []
            for j in day_windows(self.rows):
                if self.rows[j]["ts"].date() >= day:
                    continue
                v = np.asarray(day_vector(self.rows, j))
                cands.append((float(np.linalg.norm(v - np.asarray(vec))), j))
            cands.sort()
            sims = [{"day": self.rows[j]["ts"].date().isoformat(), "distance": round(d, 3),
                     "prices": [r["price"] for r in self.rows[j:j + 24]], "co2": [r["co2"] for r in self.rows[j:j + 24]]} for d, j in cands[:k]]
        if not sims:
            return np.zeros(24), np.zeros(24), []
        p = np.mean([s["prices"] for s in sims], axis=0)
        c = np.mean([s["co2"] for s in sims], axis=0)
        return p, c, sims

    def evaluate_forecaster(self, last: int = 30) -> dict:
        """Two days ahead (before day-ahead prices exist): analog days vs 'same as yesterday'."""
        idx = day_windows(self.rows)
        errs_a, errs_p, errs_ca, errs_cp = [], [], [], []
        for n, i in enumerate(idx[-last:]):
            pos = idx.index(i)
            if pos < 8:
                continue
            actual_p = np.array([r["price"] for r in self.rows[i:i + 24]])
            actual_c = np.array([r["co2"] for r in self.rows[i:i + 24]])
            prev = idx[pos - 2]  # persistence from two days before (what is known at decision time)
            pers_p = np.array([r["price"] for r in self.rows[prev:prev + 24]])
            pers_c = np.array([r["co2"] for r in self.rows[prev:prev + 24]])
            fp, fc, _ = self.analog_forecast(i)
            errs_a.append(np.mean(np.abs(fp - actual_p)))
            errs_p.append(np.mean(np.abs(pers_p - actual_p)))
            errs_ca.append(np.mean(np.abs(fc - actual_c)))
            errs_cp.append(np.mean(np.abs(pers_c - actual_c)))
        if not errs_a:
            return {}
        return {"days": len(errs_a), "price_mae_analog": round(float(np.mean(errs_a)), 1), "price_mae_persistence": round(float(np.mean(errs_p)), 1),
                "co2_mae_analog": round(float(np.mean(errs_ca)), 1), "co2_mae_persistence": round(float(np.mean(errs_cp)), 1)}

    # ---------------- Planner: one day, explained
    def day_detail(self, date: str | None = None) -> dict:
        idx = day_windows(self.rows)
        if not idx:
            return {}
        i = idx[-1]
        if date:
            for j in idx:
                if self.rows[j]["ts"].date().isoformat() == date:
                    i = j
        market = engine.window_from_hourly(self.rows, i)
        seed = int(market["ts"][0].strftime("%Y%m%d"))
        res = engine.run_day(self.site, market, seed)
        res["date"] = market["ts"][0].date().isoformat()
        res["hours"] = [t.strftime("%H:%M") for t in market["ts"]]
        res.pop("market", None)
        res["market"] = {"price": market["price"], "co2": market["co2"], "temp": market["temp"]}
        return res

    def briefing(self) -> dict:
        if self.briefing_cache and time.time() - self.briefing_cache["at"] < 600:
            return self.briefing_cache
        d = self.day_detail()
        if not d:
            return {}
        b, w = d["baseline"], d["windfall"]
        prices = d["market"]["price"]
        cheapest = sorted(range(24), key=lambda h: prices[h])[:4]
        neg = [d["hours"][h] for h in range(24) if prices[h] < 0]
        news = keenable_news("RTE réseau électrique France alerte tension prix électricité aujourd'hui")
        facts = {"date": d["date"], "baseline_cost": round(b["cost_eur"], 2), "windfall_cost": round(w["cost_eur"], 2),
                 "baseline_co2": round(b["co2_kg"], 1), "windfall_co2": round(w["co2_kg"], 1), "peak_baseline": round(b["peak_kw"]),
                 "peak_windfall": round(w["peak_kw"]), "cheapest_hours": [d["hours"][h] for h in cheapest], "negative_price_hours": neg,
                 "price_range": [min(prices), max(prices)], "vans": len(d["vans"]), "news": [n["title"] for n in news]}
        out = claude_json(
            "You are Windfall's Planner agent. You explain tomorrow's energy plan for a cold-storage site with an EV fleet to its operations manager. "
            "Be concrete, plain English, numbers from the facts only. Never invent numbers. Always write units: costs in € (whole euros), CO₂ in kg, power in kW, times as HH:MM. "
            "The 3 bullets cover: cold rooms (pre-cooling hours), vans (charging hours, ready at 06:00), and the 110 kW cap.",
            "Facts:\n" + json.dumps(facts),
            {"type": "object", "properties": {
                "headline": {"type": "string", "description": "max 14 words"},
                "plan": {"type": "array", "items": {"type": "string"}, "description": "3 bullets, max 22 words each"},
                "watch_out": {"type": "string", "description": "one risk to watch, max 20 words"}},
             "required": ["headline", "plan", "watch_out"]})
        by = "claude" if out else "rules"
        if not out:
            out = {"headline": f"Shift €{b['cost_eur'] - w['cost_eur']:.0f} of cost and {b['co2_kg'] - w['co2_kg']:.0f} kg CO₂ out of the evening peak",
                   "plan": [f"Pre-cool the cold rooms in the cheapest hours ({', '.join(facts['cheapest_hours'][:3])}).",
                            f"Charge the {facts['vans']} vans overnight in the cleanest hours; every van ready at 06:00.",
                            f"Never above the {self.site.site_cap_kw:.0f} kW subscribed power (today's naive peak: {facts['peak_baseline']} kW)."],
                   "watch_out": "Late van arrivals squeeze the overnight window; the Watcher will replan if one happens."}
        out.update({"facts": facts, "news": news, "at": time.time(), "by": by})
        if by == "claude" or not os.environ.get("ANTHROPIC_API_KEY"):
            self.briefing_cache = out
        return out

    # ---------------- Watcher: a live replay of the latest day through Kafka
    def start_live(self, speed: float = 3.0) -> dict:
        if self.live.get("running"):
            return {"running": True}
        threading.Thread(target=self._live_loop, args=(speed,), daemon=True).start()
        return {"running": True}

    def _live_loop(self, speed: float) -> None:
        self.live = {"running": True, "hour": 0}
        self.seen_hour = None
        idx = day_windows(self.rows)
        i = idx[-1]
        market = engine.window_from_hourly(self.rows, i)
        price, co2, t_out = (np.asarray(market[k], dtype=float) for k in ("price", "co2", "temp"))
        clock = [t.hour for t in market["ts"]]
        t_in, cop = engine.warehouse_temp(t_out), engine.cop_factor(t_out)
        cost = engine.cost_vector(price, co2)
        seed = int(market["ts"][0].strftime("%Y%m%d"))
        vans = engine.vans_for_day(seed, self.site.n_vans)
        base = engine.base_load(np.array(clock))
        temps = [r.t_target for r in self.site.rooms]
        plans = [engine.plan_room_dp(r, t, t_in, cost, cop) for r, t in zip(self.site.rooms, temps)]
        room_kw_plan = sum(p * r.kw * cop for p, r in zip(plans, self.site.rooms))
        ev = engine.plan_vans_smart(vans, self.site.van_charger_kw, cost, self.site.site_cap_kw * 0.92 - base - room_kw_plan)
        self.emit({"type": "live_start", "date": market["ts"][0].date().isoformat(), "hours": [t.strftime("%H:%M") for t in market["ts"]],
                   "price": price.tolist(), "co2": co2.tolist(), "plan_ev": ev.sum(axis=0).round(2).tolist(),
                   "plan_cold": room_kw_plan.round(2).tolist(), "base": base.tolist(), "cap": self.site.site_cap_kw,
                   "rooms": [r.name for r in self.site.rooms], "vans": vans})
        self.audit("Planner", "day_plan", f"Plan for {market['ts'][0].date()}: pre-cool in cheap hours, overnight charging in clean hours",
                   "Dynamic programming per cold room on the learned physics, vans filled cheapest-hour-first under the 110 kW cap.")
        events = {4: "door", 9: "late_van"}
        spent, co2_kg = 0.0, 0.0
        for h in range(engine.H):
            if not self.live.get("running"):
                break
            if h in events and events[h] == "door":
                room = self.site.rooms[1]
                temps[1] += 2.1  # a door left open: warm air in
                self.emit({"type": "anomaly", "hour": h, "asset": room.name, "message": f"{room.name}: temperature jumped to {temps[1]:.1f} °C (door open?)"})
                plans[1][h:] = engine.plan_room_dp(room, temps[1], t_in[h:], cost[h:], cop[h:], margin=0.3)
                why = claude_json(
                    "You are Windfall's Watcher agent for a cold-storage site. A telemetry anomaly just happened. Decide and explain in plain English for the site manager.",
                    json.dumps({"asset": room.name, "temperature_c": round(temps[1], 1), "limit_c": room.t_max, "hour": market["ts"][h].strftime("%H:%M"),
                                "price_now_eur_mwh": price[h], "cheapest_next_hours": [market["ts"][k].strftime("%H:%M") for k in sorted(range(h, 24), key=lambda k: cost[k])[:3]]}),
                    {"type": "object", "properties": {"action": {"type": "string"}, "explanation": {"type": "string", "description": "max 35 words"},
                                                      "notify_staff": {"type": "boolean"}}, "required": ["action", "explanation", "notify_staff"]})
                self.audit("Watcher", "replan", f"{room.name}: replanned after a temperature jump",
                           (why or {}).get("explanation") or "Cooling brought forward now to stay below −18 °C, then back to cheap hours; staff asked to check the door.", room.name)
            if h in events and events[h] == "late_van":
                v = vans[6]
                v["arrive"], v["need_kwh"] = (clock[h] + 2) % 24, 52.0
                self.emit({"type": "anomaly", "hour": h, "asset": v["van"], "message": f"{v['van']} back late ({v['arrive']:02d}:00) and needs 52 kWh"})
                room_kw_now = sum(p * r.kw * cop for p, r in zip(plans, self.site.rooms))
                fixed = ev.copy()
                fixed[6, :] = 0
                rest = engine.plan_vans_smart([v], self.site.van_charger_kw, cost, self.site.site_cap_kw * 0.92 - base - room_kw_now - fixed.sum(axis=0))
                fixed[6, :] = rest[0]
                ev = fixed
                self.audit("Watcher", "replan", f"{v['van']}: late return, charging rescheduled",
                           f"{v['van']} gets the cheapest remaining night hours inside the 110 kW cap; still full by 06:00.", v["van"])
            row = {"sim_ts": market["ts"][h].strftime("%Y-%m-%d %H:%M"), "hour": h, "price": float(price[h]), "co2": float(co2[h])}
            room_kw = 0.0
            for k, r in enumerate(self.site.rooms):
                applied, tt, _ = engine.realize_room(r, temps[k], t_in[h:h + 1], plans[k][h:h + 1])
                temps[k] = float(tt[-1])
                kw = float(applied[0] * r.kw * cop[h])
                room_kw += kw
                row[f"room{k}_temp"] = round(temps[k], 2)
                row[f"room{k}_kw"] = round(kw, 2)
            ev_kw = float(ev[:, h].sum())
            total = float(base[h] + room_kw + ev_kw)
            spent += total * price[h] / 1000
            co2_kg += total * co2[h] / 1000
            row.update({"cold_kw": round(room_kw, 2), "ev_kw": round(ev_kw, 2), "base_kw": float(base[h]), "site_kw": round(total, 2),
                        "spent_eur": round(spent, 2), "co2_kg": round(co2_kg, 1), "vans_charging": int((ev[:, h] > 0).sum())})
            if self.kf.live and self.kf.send("site.telemetry", row):
                # the Watcher consumes it from Kafka; if the stream lags, don't freeze the demo
                t0 = time.time()
                while self.seen_hour != h and time.time() - t0 < 2.0:
                    time.sleep(0.05)
                if self.seen_hour != h:
                    self.on_telemetry(row, via="direct")
            else:
                self.on_telemetry(row, via="direct")
            self.live["hour"] = h
            time.sleep(speed)
        self.live["running"] = False
        self.emit({"type": "live_end", "spent_eur": round(spent, 2), "co2_kg": round(co2_kg, 1)})

    def on_telemetry(self, row: dict, via: str = "kafka") -> None:
        """Kafka consumer side: persist to ClickHouse, update live state in Valkey, push to the dashboard."""
        if row.get("hour") is not None and row.get("hour") == self.seen_hour:
            return
        self.seen_hour = row.get("hour")
        if via == "kafka":
            self.counters["kafka_messages"] += 1
            self.vk.incr("kafka_messages")
        now = dt.datetime.now()
        sim = dt.datetime.strptime(row["sim_ts"], "%Y-%m-%d %H:%M")
        metrics = [(k, v) for k, v in row.items() if isinstance(v, (int, float)) and k not in ("hour",)]
        self.ch.insert("telemetry", [[now, sim, "site", k, float(v)] for k, v in metrics], ["ts", "sim_ts", "asset", "metric", "value"])
        self.counters["telemetry_rows"] += len(metrics)
        self.vk.set_state("live", row)
        self.emit({"type": "telemetry", **row})

    def datalayer(self) -> dict:
        ch_rows = self.ch.query("SELECT (SELECT count() FROM market_hourly), (SELECT count() FROM telemetry), (SELECT count() FROM backtest_daily)")
        mem = self.pg.q("SELECT count(*) FROM day_memory")
        dec = self.pg.q("SELECT count(*) FROM decisions")
        return {
            "clickhouse": {"market_hours": ch_rows[0][0] if ch_rows else len(self.rows), "telemetry_points": ch_rows[0][1] if ch_rows else self.counters["telemetry_rows"],
                           "backtest_days": ch_rows[0][2] if ch_rows else len(self.backtest.get("daily", []))},
            "postgres": {"day_memories": mem[0][0] if mem else 0, "decisions": dec[0][0] if dec else 0, "pgvector": getattr(self.pg, "vector", False)},
            "kafka": {"messages": self.vk.incr("kafka_messages", 0) or self.counters["kafka_messages"], "topics": list(self.kf.TOPICS)},
            "valkey": {"live_keys": 2 if self.vk.live else 0},
            "opensearch": {"audit_docs": self.os.count()},
        }
