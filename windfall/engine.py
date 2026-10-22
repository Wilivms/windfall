"""Windfall core engine: a site model, the naive baseline, the Windfall planner and the learner.

Hourly resolution. A planning day runs from 12:00 on day D to 12:00 on D+1, the window for which
day-ahead electricity prices are already known (they are published around 12:45 CET for the next day).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

H = 24  # hours in a planning window
START_HOUR = 12  # window starts at noon

# ------------------------------------------------------------------ site description


@dataclass
class ColdRoom:
    name: str
    tau_true: float  # thermal time constant in hours (how fast it warms up), unknown to the planner
    tau_hat: float  # the planner's current estimate, refined every night by the Learner
    kw: float = 10.0  # compressor rating
    cool_rate: float = 2.2  # °C removed per hour at full duty
    t_min: float = -22.0
    t_max: float = -18.0
    t_target: float = -20.0


@dataclass
class Site:
    name: str = "Lumen Cold Logistics, Rungis"
    rooms: list[ColdRoom] = field(default_factory=lambda: [
        ColdRoom("Cold room A · frozen food", tau_true=42.0, tau_hat=75.0),
        ColdRoom("Cold room B · ice cream", tau_true=30.0, tau_hat=75.0, kw=12.0),
        ColdRoom("Cold room C · overflow", tau_true=55.0, tau_hat=75.0, kw=8.0),
    ])
    n_vans: int = 12
    van_charger_kw: float = 11.0
    site_cap_kw: float = 110.0  # subscribed power: above it, overrun penalties


DUTIES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
LAMBDA_CO2 = 0.15  # € per kg CO2 used by the planner as a shadow carbon price (≈ €150/t)


def warehouse_temp(t_out: np.ndarray) -> np.ndarray:
    """Air temperature around the cold rooms: buffered outdoor temperature."""
    return 12.0 + 0.45 * (t_out - 12.0)


def cop_factor(t_out: np.ndarray) -> np.ndarray:
    """Compressors need more electricity when it is hot outside (condenser side)."""
    return 1.0 + 0.018 * (t_out - 10.0)


def step_temp(t: float, t_in: float, tau: float, duty: float, cool_rate: float) -> float:
    alpha = 1.0 - math.exp(-1.0 / tau)
    return t + (t_in - t) * alpha - duty * cool_rate


# ------------------------------------------------------------------ fleet


def vans_for_day(seed: int, n: int) -> list[dict]:
    """Vans come back between 17:00 and 21:00, leave at 06:00, need 22-55 kWh for tomorrow's rounds."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        arrive = int(rng.integers(17, 22))  # clock hour
        need = float(np.round(rng.uniform(22, 55), 1))
        out.append({"van": f"Van {i + 1:02d}", "arrive": arrive, "depart": 6, "need_kwh": need})
    return out


def hour_index(clock_hour: int) -> int:
    """Clock hour → index in the noon-to-noon window."""
    return (clock_hour - START_HOUR) % 24


def base_load(clock_hours: np.ndarray) -> np.ndarray:
    return np.where((clock_hours >= 6) & (clock_hours < 20), 9.0, 3.5)


# ------------------------------------------------------------------ strategies


def cost_vector(price_eur_mwh: np.ndarray, co2_g_kwh: np.ndarray, lam: float = LAMBDA_CO2) -> np.ndarray:
    """€ per kWh as the planner sees it: energy price plus a shadow price on carbon."""
    return price_eur_mwh / 1000.0 + lam * co2_g_kwh / 1000.0


def plan_room_thermostat(room: ColdRoom, t0: float, t_in: np.ndarray) -> np.ndarray:
    """Baseline: a classic thermostat holding the setpoint, blind to price and carbon."""
    duties = np.zeros(H)
    t = t0
    for h in range(H):
        # pick the duty that lands closest to the setpoint
        best, best_err = 0.0, 1e9
        for d in DUTIES:
            nt = step_temp(t, t_in[h], room.tau_true, d, room.cool_rate)
            err = abs(nt - room.t_target)
            if err < best_err:
                best, best_err = d, err
        duties[h] = best
        t = step_temp(t, t_in[h], room.tau_true, best, room.cool_rate)
    return duties


def plan_room_dp(room: ColdRoom, t0: float, t_in: np.ndarray, cost_kwh: np.ndarray, cop: np.ndarray,
                 margin: float = 0.4) -> np.ndarray:
    """Windfall: dynamic programming over the room temperature, using the Learner's tau estimate.

    Minimises Σ energy·cost while keeping the temperature inside [t_min+margin, t_max-margin]
    and ending the window no warmer than the setpoint (no hidden debt for tomorrow)."""
    lo, hi = room.t_min + margin, room.t_max - margin
    grid = np.round(np.arange(lo, hi + 1e-9, 0.1), 1)
    n = len(grid)
    inf = 1e18
    value = np.where(grid <= room.t_target + 0.05, 0.0, inf)  # terminal condition
    T = len(t_in)
    policy = np.zeros((T, n), dtype=int)
    alpha = 1.0 - math.exp(-1.0 / room.tau_hat)
    for h in range(T - 1, -1, -1):
        new_value = np.full(n, inf)
        for di, d in enumerate(DUTIES):
            nt = grid + (t_in[h] - grid) * alpha - d * room.cool_rate
            idx = np.round((nt - lo) / 0.1).astype(int)
            ok = (idx >= 0) & (idx < n)
            step_cost = d * room.kw * cop[h] * cost_kwh[h]
            cand = np.where(ok, step_cost + value[np.clip(idx, 0, n - 1)], inf)
            better = cand < new_value
            new_value = np.where(better, cand, new_value)
            policy[h] = np.where(better, di, policy[h])
        value = new_value
    duties = np.zeros(T)
    t = float(np.clip(t0, lo, hi))
    for h in range(T):
        i = int(np.clip(round((t - lo) / 0.1), 0, n - 1))
        d = DUTIES[policy[h, i]]
        duties[h] = d
        t = t + (t_in[h] - t) * alpha - d * room.cool_rate
    return duties


def realize_room(room: ColdRoom, t0: float, t_in: np.ndarray, duties: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Run a plan against the true physics. A safety layer forces full cooling before any breach."""
    n = len(duties)
    temps = np.zeros(n + 1)
    temps[0] = t0
    applied = np.asarray(duties, dtype=float).copy()
    overrides = 0
    for h in range(n):
        nt = step_temp(temps[h], t_in[h], room.tau_true, applied[h], room.cool_rate)
        if nt > room.t_max - 0.1:
            # safety override: cool harder now
            for d in DUTIES[DUTIES > applied[h]]:
                applied[h] = d
                nt = step_temp(temps[h], t_in[h], room.tau_true, d, room.cool_rate)
                if nt <= room.t_max - 0.1:
                    break
            overrides += 1
        temps[h + 1] = nt
    return applied, temps, overrides


def plan_vans_asap(vans: list[dict], kw: float) -> np.ndarray:
    """Baseline: every van charges at full power from the moment it is plugged in."""
    load = np.zeros((len(vans), H))
    for i, v in enumerate(vans):
        need = v["need_kwh"]
        h = hour_index(v["arrive"])
        end = hour_index(v["depart"])
        while need > 1e-6 and h < end:
            e = min(kw, need)
            load[i, h] = e
            need -= e
            h += 1
    return load


def plan_vans_smart(vans: list[dict], kw: float, cost_kwh: np.ndarray, headroom_kw: np.ndarray) -> np.ndarray:
    """Windfall: fill each van in its cheapest, cleanest hours, inside the site's spare capacity.

    Vans with the tightest windows are served first so every van is full by 06:00."""
    load = np.zeros((len(vans), H))
    room_left = headroom_kw.copy()
    order = sorted(range(len(vans)), key=lambda i: (hour_index(vans[i]["depart"]) - hour_index(vans[i]["arrive"])) * kw - vans[i]["need_kwh"])
    for i in order:
        v = vans[i]
        a, d = hour_index(v["arrive"]), hour_index(v["depart"])
        hours = sorted(range(a, d), key=lambda h: cost_kwh[h])
        need = v["need_kwh"]
        for h in hours:
            if need <= 1e-6:
                break
            e = min(kw, need, max(0.0, room_left[h]))
            if e > 0:
                load[i, h] += e
                room_left[h] -= e
                need -= e
        if need > 1e-6:  # capacity too tight: finish as early as possible regardless of the cap
            for h in range(a, d):
                e = min(kw - load[i, h], need)
                if e > 0:
                    load[i, h] += e
                    need -= e
                if need <= 1e-6:
                    break
    return load


# ------------------------------------------------------------------ one day, both strategies


def run_day(site: Site, market: dict, seed: int, room_t0: list[float] | None = None) -> dict:
    """market: arrays of length H for price (€/MWh), co2 (g/kWh), temp (°C), clock (hour of day)."""
    price, co2, t_out, clock = (np.asarray(market[k], dtype=float) for k in ("price", "co2", "temp", "clock"))
    t_in = warehouse_temp(t_out)
    cop = cop_factor(t_out)
    cost = cost_vector(price, co2)
    vans = vans_for_day(seed, site.n_vans)
    base = base_load(clock.astype(int))
    t0s = room_t0 or [r.t_target for r in site.rooms]

    out = {}
    for strategy in ("baseline", "windfall"):
        room_kw = np.zeros(H)
        overrides = 0
        temps_all, duties_all = [], []
        violations = 0
        for r, t0 in zip(site.rooms, t0s):
            if strategy == "baseline":
                planned = plan_room_thermostat(r, t0, t_in)
            else:
                planned = plan_room_dp(r, t0, t_in, cost, cop)
            applied, temps, ov = realize_room(r, t0, t_in, planned)
            overrides += ov if strategy == "windfall" else 0
            violations += int(np.sum((temps > r.t_max + 1e-6) | (temps < r.t_min - 1e-6)))
            room_kw += applied * r.kw * cop
            temps_all.append(temps.round(2).tolist())
            duties_all.append(applied.tolist())
        headroom = site.site_cap_kw * 0.92 - base - room_kw
        ev = plan_vans_asap(vans, site.van_charger_kw) if strategy == "baseline" else plan_vans_smart(vans, site.van_charger_kw, cost, headroom)
        ev_kw = ev.sum(axis=0)
        total_kw = base + room_kw + ev_kw
        kwh = float(total_kw.sum())
        out[strategy] = {
            "cost_eur": float(np.sum(total_kw * price / 1000.0)),
            "co2_kg": float(np.sum(total_kw * co2 / 1000.0)),
            "kwh": kwh,
            "peak_kw": float(total_kw.max()),
            "over_cap_hours": int(np.sum(total_kw > site.site_cap_kw + 1e-6)),
            "cold_violations": violations,
            "safety_overrides": overrides,
            "vans_ready": int(sum(1 for i, v in enumerate(vans) if ev[i].sum() >= v["need_kwh"] - 0.05)),
            "load": {"base": base.tolist(), "cold": room_kw.round(2).tolist(), "ev": ev_kw.round(2).tolist(), "total": total_kw.round(2).tolist()},
            "room_temps": temps_all,
            "room_duties": duties_all,
            "ev_by_van": ev.round(2).tolist(),
        }
    out["vans"] = vans
    out["market"] = {"price": price.tolist(), "co2": co2.tolist(), "temp": t_out.tolist(), "clock": clock.astype(int).tolist()}
    return out


# ------------------------------------------------------------------ the Learner


def learn_tau(room: ColdRoom, temps: list[float], duties: list[float], t_in: np.ndarray, rate: float = 0.6) -> float:
    """Re-estimate a room's thermal constant from yesterday's real trajectory (least squares)."""
    temps = np.asarray(temps)
    duties = np.asarray(duties)
    y = temps[1:] - temps[:-1] + duties * room.cool_rate  # warming that happened by itself
    x = t_in - temps[:-1]
    denom = float(np.sum(x * x))
    if denom <= 1e-9:
        return room.tau_hat
    alpha = float(np.clip(np.sum(x * y) / denom, 1e-4, 0.9))
    tau_obs = -1.0 / math.log(1.0 - alpha)
    room.tau_hat = (1 - rate) * room.tau_hat + rate * tau_obs
    return room.tau_hat


def window_from_hourly(rows: list[dict], start_index: int) -> dict:
    """Slice 24 hourly rows (dicts with ts, price, co2, temp) into a planning window."""
    sl = rows[start_index:start_index + H]
    return {
        "price": [r["price"] for r in sl],
        "co2": [r["co2"] for r in sl],
        "temp": [r["temp"] for r in sl],
        "clock": [r["ts"].hour for r in sl],
        "ts": [r["ts"] for r in sl],
    }
