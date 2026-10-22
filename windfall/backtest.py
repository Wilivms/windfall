"""Replay the last N real days: naive baseline vs Windfall, with the Learner improving the model nightly."""
from __future__ import annotations

import numpy as np

from . import engine
from .data import day_windows


def run_backtest(rows: list[dict], days: int = 90, learn: bool = True) -> dict:
    site = engine.Site()
    starts = day_windows(rows)[-days:]
    daily = []
    t0 = {"baseline": [r.t_target for r in site.rooms], "windfall": [r.t_target for r in site.rooms]}
    for n, i in enumerate(starts):
        market = engine.window_from_hourly(rows, i)
        seed = int(market["ts"][0].strftime("%Y%m%d"))
        res = engine.run_day(site, market, seed, room_t0=t0["windfall"])
        # carry the real end-of-day temperatures to the next day
        t0["windfall"] = [temps[-1] for temps in res["windfall"]["room_temps"]]
        taus_before = [r.tau_hat for r in site.rooms]
        if learn:
            t_in = engine.warehouse_temp(np.asarray(market["temp"]))
            for r, temps, duties in zip(site.rooms, res["windfall"]["room_temps"], res["windfall"]["room_duties"]):
                engine.learn_tau(r, temps, duties, t_in)
        b, w = res["baseline"], res["windfall"]
        daily.append({
            "day": market["ts"][0].date().isoformat(),
            "day_index": n,
            "baseline_cost": round(b["cost_eur"], 2), "windfall_cost": round(w["cost_eur"], 2),
            "baseline_co2": round(b["co2_kg"], 1), "windfall_co2": round(w["co2_kg"], 1),
            "baseline_peak": round(b["peak_kw"], 1), "windfall_peak": round(w["peak_kw"], 1),
            "baseline_over_cap": b["over_cap_hours"], "windfall_over_cap": w["over_cap_hours"],
            "kwh": round(w["kwh"], 1),
            "violations_baseline": b["cold_violations"], "violations_windfall": w["cold_violations"],
            "safety_overrides": w["safety_overrides"],
            "vans_ready": w["vans_ready"], "vans": len(res["vans"]),
            "tau_error_pct": round(float(np.mean([abs(th - r.tau_true) / r.tau_true for th, r in zip(taus_before, site.rooms)])) * 100, 1),
            "min_price": round(min(market["price"]), 1), "max_price": round(max(market["price"]), 1),
            "negative_hours": int(sum(1 for p in market["price"] if p < 0)),
        })
    return {"daily": daily, "summary": summarize(daily), "learned_tau": {r.name: {"estimate": round(r.tau_hat, 2), "true": r.tau_true} for r in site.rooms}}


def summarize(daily: list[dict]) -> dict:
    if not daily:
        return {}
    s = lambda k: float(sum(d[k] for d in daily))  # noqa: E731
    n = len(daily)
    bc, wc, bco, wco = s("baseline_cost"), s("windfall_cost"), s("baseline_co2"), s("windfall_co2")
    first, last = daily[: max(1, n // 6)], daily[-max(1, n // 6):]
    return {
        "days": n,
        "baseline_cost_eur": round(bc, 0), "windfall_cost_eur": round(wc, 0),
        "saved_eur": round(bc - wc, 0), "saved_pct": round((bc - wc) / bc * 100, 1) if bc else 0,
        "saved_eur_per_year": round((bc - wc) / n * 365, -2),
        "baseline_co2_kg": round(bco, 0), "windfall_co2_kg": round(wco, 0),
        "co2_avoided_kg": round(bco - wco, 0), "co2_avoided_pct": round((bco - wco) / bco * 100, 1) if bco else 0,
        "co2_avoided_t_per_year": round((bco - wco) / n * 365 / 1000, 1),
        "baseline_peak_kw": round(max(d["baseline_peak"] for d in daily), 0),
        "windfall_peak_kw": round(max(d["windfall_peak"] for d in daily), 0),
        "over_cap_hours_baseline": int(s("baseline_over_cap")), "over_cap_hours_windfall": int(s("windfall_over_cap")),
        "cold_violations_windfall": int(s("violations_windfall")),
        "vans_ready_pct": round(s("vans_ready") / s("vans") * 100, 1),
        "safety_overrides_first_days": round(sum(d["safety_overrides"] for d in first) / len(first), 2),
        "safety_overrides_last_days": round(sum(d["safety_overrides"] for d in last) / len(last), 2),
        "model_error_first_pct": first[0]["tau_error_pct"], "model_error_last_pct": last[-1]["tau_error_pct"],
        "negative_price_hours": int(s("negative_hours")),
    }
