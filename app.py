import os, json, time, datetime as dt
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, Body, HTTPException

# ================== CONFIG ==================
# Gắn trực tiếp BOT_TOKEN & CHAT_ID của anh
BOT_TOKEN  = os.getenv("BOT_TOKEN", "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o")
CHAT_ID    = os.getenv("CHAT_ID", "5614513021")

def env_json(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default

PLAN       = env_json("PLAN_JSON", {})
PIVOTS     = env_json("PIVOTS_JSON", {})
ACCOUNT_VALUE = float(os.getenv("ACCOUNT_VALUE", "30000000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
RESOLUTION = os.getenv("RESOLUTION", "5")   # "5" = 5m, "D" = daily

# ================== FASTAPI APP ==================
app = FastAPI(title="VN21 Scanner PRO-FULL", version="2.0.0")

# ================== HELPERS ==================
def tg_send(text: str) -> Optional[Dict[str, Any]]:
    if not (BOT_TOKEN and CHAT_ID):
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if r.ok:
            return r.json()
    except Exception as e:
        print("Telegram error:", e)
    return None

def format_signal(sym: str, msg: str) -> str:
    return f"📡 {sym} — {msg}"

# ================== FETCH DATA ==================
def fetch_ohlcv(symbol: str, resolution=RESOLUTION, days_back=120):
    now = dt.datetime.now()
    frm = int((now - dt.timedelta(days=days_back)).timestamp())
    to  = int(now.timestamp())
    url = "https://dchart-api.vndirect.com.vn/dchart/history"
    r = requests.get(url, params={
        "symbol": symbol,
        "resolution": resolution,
        "from": frm,
        "to": to
    }, timeout=15)
    r.raise_for_status()
    js = r.json()
    if not js or "t" not in js or not js["t"]:
        return None
    df = pd.DataFrame({
        "t": [dt.datetime.fromtimestamp(x) for x in js["t"]],
        "o": js["o"], "h": js["h"], "l": js["l"], "c": js["c"], "v": js["v"]
    })
    return df

# ================== CORE LOGIC ==================
last_alert_time = {}  # cooldown
latest_alerts: List[str] = []

def scan_symbols(symbols: List[str]) -> List[str]:
    alerts = []

    # Market bias theo VNIndex
    vnindex_df = fetch_ohlcv("VNINDEX", "D", 120)
    market_bias = "Bull"
    if vnindex_df is not None and len(vnindex_df) >= 50:
        ma50_idx = vnindex_df["c"].rolling(50).mean().iloc[-1]
        if vnindex_df["c"].iloc[-1] < ma50_idx:
            market_bias = "Bear"

    for sym in symbols:
        pivot = PIVOTS.get(sym.upper(), None)
        plan  = PLAN.get(sym.upper(), {})

        try:
            df = fetch_ohlcv(sym, RESOLUTION, 120)
            if df is None or len(df) < 60:
                continue

            price = df["c"].iloc[-1]
            vol   = df["v"].iloc[-1]

            # ========= INDICATORS =========
            ma20 = df["c"].rolling(20).mean().iloc[-1]
            ma50 = df["c"].rolling(50).mean().iloc[-1]
            ma200 = df["c"].rolling(100).mean().iloc[-1]
            atr14 = (df["h"] - df["l"]).rolling(14).mean().iloc[-1]
            rvol = vol / df["v"].rolling(20).mean().iloc[-1]

            perf_3m = (price / df["c"].iloc[-60]) - 1
            vnindex_perf = 0.05
            rs = (perf_3m - vnindex_perf) * 100

            lookback = 20
            box_high = df["h"].iloc[-lookback:].max()

            # ========= CONDITIONS =========
            cond_vol = rvol >= 1.5
            cond_trend = ma20 > ma50 > ma200
            cond_breakout = price > box_high
            cond_fake = price < box_high * 1.01 or vol < 1.5 * df["v"].rolling(20).mean().iloc[-1]
            cond_atr = (df["h"].iloc[-1] - df["l"].iloc[-1]) >= 0.5 * atr14
            cond_rs = rs > 0

            if pivot and cond_vol and cond_trend and cond_breakout and not cond_fake and cond_atr and cond_rs:
                # Cooldown 10 phút
                now_ts = time.time()
                if sym in last_alert_time and now_ts - last_alert_time[sym] < 600:
                    continue
                last_alert_time[sym] = now_ts

                t1 = plan.get("t1", round(price + 1*atr14, 2))
                t2 = plan.get("t2", round(price + 2*atr14, 2))
                sl = plan.get("sl", round(price - 1*atr14, 2))

                risk_amount = ACCOUNT_VALUE * RISK_PER_TRADE
                qty = int(risk_amount / max(price - sl, 0.01))

                msg = (
                    f"{sym} – BUY {round(price,2)} | T1: {t1} | T2: {t2} | SL: {sl} | "
                    f"⚡ Breakout PRO-FULL (RVOL={rvol:.2f}, RS={rs:.1f}, ATR={atr14:.2f}) "
                    f"| Qty≈{qty} | Market: {market_bias}"
                )
                alerts.append(msg)

        except Exception as e:
            print(f"[{sym}] fetch error:", e)

    return alerts

# ================== ENDPOINTS ==================
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "vn21-scanner-pro-full"}

@app.get("/")
async def root():
    return {
        "message": "VN21 Scanner PRO-FULL is running 🚀",
        "endpoints": ["/healthz", "/update", "/notify", "/plan", "/pivots", "/env-check", "/summary"],
    }

@app.get("/plan")
async def get_plan():
    return PLAN

@app.post("/plan")
async def set_plan(payload: Dict[str, Any] = Body(...)):
    global PLAN
    if not isinstance(payload, dict):
        raise HTTPException(400, "Body must be a JSON object (symbol -> config).")
    PLAN = payload
    return {"ok": True, "size": len(PLAN)}

@app.get("/pivots")
async def get_pivots():
    return PIVOTS

@app.post("/pivots")
async def set_pivots(payload: Dict[str, float] = Body(...)):
    global PIVOTS
    PIVOTS = {k.upper(): float(v) for k, v in payload.items()}
    return {"ok": True, "size": len(PIVOTS)}

@app.get("/notify")
async def notify(text: str):
    r = tg_send(text)
    return {"ok": bool(r), "preview": text[:120]}

@app.post("/update")
async def update(background_tasks: BackgroundTasks, symbols: Optional[List[str]] = Body(default=None)):
    syms = symbols or sorted(set(list(PLAN.keys()) + list(PIVOTS.keys())))
    syms = [s.upper() for s in syms]

    def _run_and_push():
        global latest_alerts
        alerts = scan_symbols(syms)
        latest_alerts = alerts
        if not alerts:
            tg_send("✅ Scanner chạy xong: chưa có tín hiệu mới.")
        else:
            for a in alerts:
                tg_send(format_signal(a.split(" – ")[0], a))

    background_tasks.add_task(_run_and_push)
    return {"queued": True, "symbols": syms}

@app.get("/env-check")
async def env_check():
    return {
        "bot_token": BOT_TOKEN[:10] + "...",
        "chat_id": CHAT_ID,
        "plan_size": len(PLAN),
        "pivots_size": len(PIVOTS),
        "resolution": RESOLUTION,
    }

@app.get("/summary")
async def summary():
    return {"latest_alerts": latest_alerts}
