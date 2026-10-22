from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import stores
from .agents import Hub, claude_json
from .data import load_market

app = FastAPI(title="Windfall")
STATIC = Path(__file__).parent / "static"
HUB: Hub | None = None


@app.on_event("startup")
def startup() -> None:
    global HUB
    def init() -> None:
        global HUB
        pg, ch, vk, kf, os_ = stores.PG(), stores.CH(), stores.VK(), stores.KF(), stores.OS()
        hub = Hub(pg, ch, vk, kf, os_)
        HUB = hub
        hub.start_valkey_bridge()
        if kf.live:
            kf.consume("site.telemetry", hub.on_telemetry)
        hub.boot(load_market)
        # services created at the same time as the app can take a few minutes to resolve: keep retrying
        import time
        for _ in range(60):
            time.sleep(30)
            if not hub.os.live:
                hub.os = stores.OS()
            if not hub.ch.live:
                hub.ch = stores.CH()
            if not hub.vk.live:
                hub.vk = stores.VK()
                if hub.vk.live:
                    hub.start_valkey_bridge()
            if hub.os.live and hub.ch.live and hub.vk.live:
                break
    threading.Thread(target=init, daemon=True).start()


def hub() -> Hub:
    if HUB is None:
        raise RuntimeError("starting")
    return HUB


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    h = HUB
    return JSONResponse({"ready": bool(h and h.ready), "progress": h.progress if h else "starting", "data_source": h.source if h else None,
                         "stores": stores.STATUS, "claude": bool(os.environ.get("ANTHROPIC_API_KEY")), "keenable": bool(os.environ.get("KEENABLE_API_KEY"))})


@app.get("/api/backtest")
def backtest() -> JSONResponse:
    h = hub()
    return JSONResponse({**h.backtest, "forecast": h.forecast_eval, "data_source": h.source})


@app.get("/api/day")
def day(date: str | None = None) -> JSONResponse:
    return JSONResponse(hub().day_detail(date))


@app.get("/api/days")
def days() -> JSONResponse:
    return JSONResponse([d["day"] for d in hub().backtest.get("daily", [])])


@app.get("/api/briefing")
def briefing() -> JSONResponse:
    return JSONResponse(hub().briefing())


@app.get("/api/forecast")
def forecast() -> JSONResponse:
    h = hub()
    from .data import day_windows
    i = day_windows(h.rows)[-1]
    p, c, sims = h.analog_forecast(i)
    return JSONResponse({"day": h.rows[i]["ts"].date().isoformat(), "forecast_price": [round(x, 1) for x in p], "forecast_co2": [round(x, 1) for x in c],
                         "actual_price": [r["price"] for r in h.rows[i:i + 24]], "similar_days": [{k: v for k, v in s.items() if k in ("day", "distance", "saved_eur")} for s in sims],
                         "evaluation": h.forecast_eval})


@app.get("/api/datalayer")
def datalayer() -> JSONResponse:
    return JSONResponse({"status": stores.STATUS, "counts": hub().datalayer()})


@app.get("/api/audit")
def audit(q: str = Query("", max_length=200)) -> JSONResponse:
    h = hub()
    docs = h.os.search(q)
    if not docs:
        rows = h.pg.q("SELECT at, agent, kind, summary, payload FROM decisions ORDER BY at DESC LIMIT 30")
        docs = [{"at": r[0].isoformat(), "agent": r[1], "kind": r[2], "summary": r[3], "explanation": (r[4] or {}).get("explanation", "")} for r in rows]
    return JSONResponse(docs)


@app.get("/api/report")
def report() -> JSONResponse:
    h = hub()
    s = h.backtest.get("summary", {})
    out = claude_json(
        "You are Windfall's Auditor agent. Draft the energy & scope-2 emissions note a small company can paste into its CSRD / sustainability report. "
        "Use only the numbers given. Location-based method, French grid hourly carbon intensity computed from the real generation mix.",
        json.dumps({"site": h.site.name, "period_days": s.get("days"), **s, "data_source": h.source}),
        {"type": "object", "properties": {"title": {"type": "string"}, "paragraphs": {"type": "array", "items": {"type": "string"}},
                                          "method": {"type": "string"}}, "required": ["title", "paragraphs", "method"]}, max_tokens=900)
    if not out:
        out = {"title": f"Energy and scope-2 emissions, last {s.get('days')} days",
               "paragraphs": [f"Windfall shifted the site's flexible consumption (3 cold rooms, 12 EV vans) to cheaper, lower-carbon hours. Energy cost fell by €{s.get('saved_eur')} ({s.get('saved_pct')}%).",
                              f"Location-based scope-2 emissions fell by {s.get('co2_avoided_kg')} kg CO₂e ({s.get('co2_avoided_pct')}%), with zero cold-chain violations and 100% of vans ready at 06:00."],
               "method": "Hourly French grid intensity from the real generation mix (Energy-Charts) × hourly site consumption; IPCC life-cycle emission factors."}
    h.audit("Auditor", "report", out["title"], out["method"])
    return JSONResponse(out)


@app.post("/api/live/start")
def live_start(speed: float = 2.5) -> JSONResponse:
    return JSONResponse(hub().start_live(max(0.4, min(speed, 8.0))))


@app.post("/api/live/stop")
def live_stop() -> JSONResponse:
    hub().live["running"] = False
    return JSONResponse({"running": False})


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    q: queue.Queue = queue.Queue(maxsize=500)
    h = hub()
    with h.listeners_lock:
        h.listeners.append(q)

    async def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    ev = q.get_nowait()
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.15)
                    yield ": ping\n\n"
        finally:
            with h.listeners_lock:
                if q in h.listeners:
                    h.listeners.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
