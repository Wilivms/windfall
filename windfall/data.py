"""Real market data for France: day-ahead prices, generation mix (→ grid carbon intensity) and weather.

Sources (open, no key):
  - Energy-Charts (Fraunhofer ISE): day-ahead price for the FR bidding zone and public power by fuel
  - Open-Meteo: hourly temperature in Rungis, past 92 days + 2 days of forecast
A deterministic synthetic generator is used only if a source is unreachable, and every row says which.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from zoneinfo import ZoneInfo

import httpx
import numpy as np

PARIS = ZoneInfo("Europe/Paris")
LAT, LON = 48.76, 2.37  # Rungis, south of Paris

# Life-cycle emission factors, gCO2eq/kWh (IPCC AR5 medians; gas CCGT/OCGT blend for the French fleet)
EF = {
    "nuclear": 12, "solar": 45, "wind": 11, "hydro": 24, "biomass": 230, "geothermal": 38,
    "gas": 490, "coal": 820, "lignite": 1050, "oil": 650, "waste": 330, "other": 300,
}
SKIP = ("load", "residual", "share", "trading", "consumption", "import", "export")


def _fuel(name: str) -> str | None:
    n = name.lower()
    if any(s in n for s in SKIP):
        return None
    for key in ("nuclear", "solar", "wind", "hydro", "biomass", "geothermal", "gas", "lignite", "coal", "oil", "waste"):
        if key in n:
            return key
    return "other"


def _hourly(ts: list[int], vals: list[float | None]) -> dict[dt.datetime, float]:
    buckets: dict[dt.datetime, list[float]] = defaultdict(list)
    for t, v in zip(ts, vals):
        if v is None:
            continue
        h = dt.datetime.fromtimestamp(t, PARIS).replace(minute=0, second=0, microsecond=0)
        buckets[h].append(float(v))
    return {h: sum(v) / len(v) for h, v in buckets.items()}


def fetch_real(days: int = 92, timeout: float = 40.0) -> list[dict]:
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=days + 1)
    with httpx.Client(timeout=timeout, follow_redirects=True) as c:
        p = c.get("https://api.energy-charts.info/price", params={"bzn": "FR", "start": start.isoformat(), "end": end.isoformat()}).json()
        g = c.get("https://api.energy-charts.info/public_power", params={"country": "fr", "start": start.isoformat(), "end": end.isoformat()}).json()
        w = c.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LAT, "longitude": LON, "hourly": "temperature_2m", "past_days": min(days, 92),
            "forecast_days": 2, "timezone": "Europe/Paris"}).json()
    price = _hourly(p["unix_seconds"], p["price"])
    # carbon intensity = Σ(gen_f · EF_f) / Σ gen_f
    num: dict[dt.datetime, float] = defaultdict(float)
    den: dict[dt.datetime, float] = defaultdict(float)
    for pt in g["production_types"]:
        fuel = _fuel(pt["name"])
        if fuel is None:
            continue
        for h, mw in _hourly(g["unix_seconds"], pt["data"]).items():
            if mw > 0:
                num[h] += mw * EF[fuel]
                den[h] += mw
    co2 = {h: num[h] / den[h] for h in den if den[h] > 0}
    temp = {}
    for t, v in zip(w["hourly"]["time"], w["hourly"]["temperature_2m"]):
        if v is not None:
            temp[dt.datetime.fromisoformat(t).replace(tzinfo=PARIS)] = float(v)
    rows = []
    for h in sorted(price):
        if h in co2 and h in temp:
            rows.append({"ts": h, "price": round(price[h], 2), "co2": round(co2[h], 1), "temp": round(temp[h], 1), "source": "real"})
    if len(rows) < 24 * 14:
        raise RuntimeError(f"only {len(rows)} aligned hours from the real sources")
    return rows


def synthetic(days: int = 92, seed: int = 7) -> list[dict]:
    """Shape-faithful stand-in (evening peak, solar dip, weekend effect), used only if sources fail."""
    rng = np.random.default_rng(seed)
    now = dt.datetime.now(PARIS).replace(minute=0, second=0, microsecond=0)
    start = now - dt.timedelta(days=days)
    rows = []
    for i in range(days * 24 + 36):
        ts = start + dt.timedelta(hours=i)
        h, doy, wd = ts.hour, ts.timetuple().tm_yday, ts.weekday()
        temp = 13 + 7 * math.sin(2 * math.pi * (doy - 110) / 365) + 4 * math.sin(2 * math.pi * (h - 9) / 24) + rng.normal(0, 1.2)
        solar = max(0.0, math.sin(math.pi * (h - 7) / 12)) if 7 <= h <= 19 else 0.0
        evening = math.exp(-((h - 19.5) ** 2) / 4)
        morning = math.exp(-((h - 8.5) ** 2) / 3)
        wk = 0.82 if wd >= 5 else 1.0
        price = wk * (70 + 55 * evening + 25 * morning - 65 * solar) + rng.normal(0, 9)
        co2 = 22 + 38 * evening + 12 * morning - 9 * solar + rng.normal(0, 4)
        rows.append({"ts": ts, "price": round(price, 2), "co2": round(max(8, co2), 1), "temp": round(temp, 1), "source": "synthetic"})
    return rows


def load_market(days: int = 92) -> tuple[list[dict], str]:
    try:
        return fetch_real(days), "real"
    except Exception as e:  # noqa: BLE001
        print(f"[data] real sources unavailable ({e}); using synthetic market data")
        return synthetic(days), "synthetic"


def day_windows(rows: list[dict]) -> list[int]:
    """Indices where a noon-to-noon window starts and is complete."""
    idx = []
    for i, r in enumerate(rows):
        if r["ts"].hour == 12 and i + 24 <= len(rows):
            if (rows[i + 23]["ts"] - r["ts"]) == dt.timedelta(hours=23):
                idx.append(i)
    return idx


def day_vector(rows: list[dict], i: int) -> list[float]:
    """24-dim fingerprint of a day used for similar-day memory (pgvector): temperature + calendar."""
    temps = [rows[i + k]["temp"] for k in range(0, 24, 2)]  # 12 values
    ts = rows[i]["ts"]
    wd = [1.0 if ts.weekday() == d else 0.0 for d in range(7)]
    season = [math.sin(2 * math.pi * ts.timetuple().tm_yday / 365), math.cos(2 * math.pi * ts.timetuple().tm_yday / 365)]
    mean_t = sum(temps) / len(temps)
    spread = max(temps) - min(temps)
    vec = [t / 30.0 for t in temps] + wd + season + [mean_t / 30.0, spread / 15.0, 1.0 if ts.weekday() >= 5 else 0.0]
    return [round(v, 4) for v in vec]
