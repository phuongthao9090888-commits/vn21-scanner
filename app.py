# app.py
import os
import time
import math
import json
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional

import requests
import pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# =========================
# Config & constants
# =========================
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

DEFAULT_TICKERS = [
    "VPB", "MBB", "TCB", "CTG", "DCM", "KDH",
    "HPG", "VHM", "VIX", "DDV", "BSR", "POW",
    "REE", "GMD", "VNM", "MWG"
]

TOKEN = os.getenv("TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

# pivots
try:
    PIVOTS: Dict[str, float] = json.loads(os.getenv("PIVOTS_JSON", "{}"))
except Exception:
    PIVOTS = {}

# plans (targets/SL)
try:
    PLAN: Dict[str, Dict[str, float]] = json.loads(os.getenv("PLAN_JSON", "{}"))
except Exception:
    PLAN = {}

TICKERS = [x.strip().upper() for x in os.getenv("TICKERS", "").split(",") if x.strip()] or DEFAULT_TICKERS
SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "60"))
TRADING_MIN_PER_DAY = int(os.getenv("TRADING_MIN_PER_DAY", "270"))

# VNDirect DChart endpoints
DCHART_URL = "https://dchart-api.vndirect.com.vn/dchart/history"

# memory to avoid duplicate alerts per day
sent_today: Dict[str, str] = {}  # {symbol: "YYYY-MM-DD"}

app = FastAPI(title="VN21 Scanner", version="1.0")


# =========================
# Utils
# =========================
def now_ts() -> int:
    return int(datetime.now(tz=TZ).timestamp())

def hs(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=TZ).strftime("%H:%M")

def get_plan_levels(sym: str, ref_price: float) -> Dict[str, float]:
    """Return T1/T2/SL for symbol, fallback to % levels from ref_price."""
    plan = PLAN.get(sym, {})
    if {"t1", "t2", "sl"} <= set(plan.keys()):
        return {"t1": float(plan["t1"]), "t2": float(plan["t2"]), "sl": float(plan["sl"])}
    # fallback: +3% / +6% / -3% từ pivot hoặc giá gần nhất
    base = PIVOTS.get(sym, ref_price)
    return {
        "t1": round(base * 1.03, 3),
        "t2": round(base * 1.06, 3),
        "sl": round(base * 0.97, 3),
    }

def telegram_send(text: str) -> None:
    if not (TOKEN and CHAT_ID):
        return
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception:
        pass


# =========================
# Data fetchers (VNDirect)
# =========================
def fetch_intraday_5m(symbol: str, days: int = 2) -> pd.DataFrame:
    """
    Lấy nến 5m trong N ngày gần nhất từ DChart.
    Trả về DataFrame: t (epoch), o,h,l,c,v
    """
    to_ts = now_ts()
    from_ts = to_ts - days * 86400
    params = {
        "symbol": symbol,
        "resolution": "5",
        "from": from_ts,
        "to": to_ts,
    }
    r = requests.get(DCHART_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame({
        "t": data.get("t", []),
        "o": data.get("o", []),
        "h": data.get("h", []),
        "l": data.get("l", []),
        "c": data.get("c", []),
        "v": data.get("v", []),
    })
    return df

def fetch_daily(symbol: str, days: int = 40) -> pd.DataFrame:
    """Lấy nến D để tính vol trung bình 20 ngày."""
    to_ts = now_ts()
    from_ts = to_ts - days * 86400
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": from_ts,
        "to": to_ts,
    }
    r = requests.get(DCHART_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("s") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame({
        "t": data.get("t", []),
        "o": data.get("o", []),
        "h": data.get("h", []),
        "l": data.get("l", []),
        "c": data.get("c", []),
        "v": data.get("v", []),
    })
    return df


# =========================
# Indicators / models
# =========================
def avg_per_minute_volume_20d(symbol: str) -> Optional[float]:
    """
    Ước lượng vol trung bình mỗi phút dựa trên vol trung bình 20 phiên (nến D).
    """
    dfd = fetch_daily(symbol, days=60)
    if dfd.empty:
        return None
    dfd = dfd.tail(20)
    if dfd.empty:
        return None
    avg_daily_vol = float(pd.to_numeric(dfd["v"], errors="coerce").dropna().mean())
    if avg_daily_vol <= 0:
        return None
    return avg_daily_vol / max(TRADING_MIN_PER_DAY, 1)

def wick_ok(o: float, h: float, l: float, c: float) -> bool:
    """Không có bóng trên dài >60% thân nến."""
    body = abs(c - o)
    upper = h - max(o, c)
    # nếu thân quá nhỏ, cho phép upper nhỏ hơn 0.2% giá
    if body < 1e-4:
        return upper <= 0.002 * c
    return upper <= 0.6 * body

def detect_darvas_box(df5: pd.DataFrame, lookback_bars: int = 60) -> Optional[Dict[str, float]]:
    """
    Darvas Box đơn giản: tìm hộp gần nhất trong lookback.
    Trả về {top, bottom} nếu phát hiện.
    """
    if df5.empty:
        return None
    seg = df5.tail(lookback_bars)
    high = float(seg["h"].max())
    low = float(seg["l"].min())
    if math.isfinite(high) and math.isfinite(low) and high > low:
        return {"top": round(high, 3), "bottom": round(low, 3)}
    return None

def breakout_signal(symbol: str, pivots: Dict[str, float]) -> Optional[str]:
    """
    Áp bộ tiêu chí:
      (1) Close > pivot 2 nến 5m liên tiếp HOẶC close > pivot * 1.01
      (2) Vol 5m >= 1.5x vol trung bình / phút * 5
      (3) Wick trên không dài
    => Nếu đạt, trả về thông điệp cảnh báo 1 dòng; ngược lại trả None.
    """
    pivot = float(pivots.get(symbol, 0) or 0)
    if pivot <= 0:
        return None

    df5 = fetch_intraday_5m(symbol, days=2)
    if df5.empty or len(df5) < 3:
        return None

    # Lấy 2 nến 5m đã hoàn thành (bỏ nến đang chạy)
    last2 = df5.tail(3).iloc[:2]  # 2 candles trước nến hiện tại
    c1, c2 = float(last2.iloc[-2]["c"]), float(last2.iloc[-1]["c"])
    o2, h2, l2 = float(last2.iloc[-1]["o"]), float(last2.iloc[-1]["h"]), float(last2.iloc[-1]["l"])
    v2 = float(last2.iloc[-1]["v"])

    # Điều kiện (1)
    cond1 = (c1 > pivot and c2 > pivot) or (c2 >= pivot * 1.01)

    # Điều kiện (2): vol 5m so với MA20 per-minute
    apm = avg_per_minute_volume_20d(symbol)
    if apm is None or apm <= 0:
        return None
    vol_threshold = 1.5 * apm * 5  # 1.5x mỗi phút * 5 phút
    cond2 = v2 >= vol_threshold

    # Điều kiện (3): wick
    cond3 = wick_ok(o2, h2, l2, c2)

    if not (cond1 and cond2 and cond3):
        return None

    # Model tags (Darvas/CANSLIM/Zanger) — đánh dấu heuristic nhẹ
    model_tags = []
    box = detect_darvas_box(df5)
    if box and c2 > box["top"]:
        model_tags.append("Darvas")
    # CANSLIM / Zanger nhãn tham khảo (không đánh giá EPS/RS ở đây)
    if c2 > pivot * 1.02 and v2 > vol_threshold * 1.3:
        model_tags.append("Zanger")
    else:
        model_tags.append("CANSLIM")

    # Entry/Targets/SL
    plan = get_plan_levels(symbol, ref_price=c2)
    entry_low = round(pivot, 3)
    entry_high = round(c2, 3)

    msg = (
        f"{symbol} – BUY {entry_low}-{entry_high} | "
        f"T1: {plan['t1']} | T2: {plan['t2']} | SL: {plan['sl']} | "
        f"⚡ Breakout xác nhận (vol {int(v2):,} vs ≥{int(vol_threshold):,}; {hs(int(df5.tail(2).iloc[0]['t']))}-{hs(int(df5.tail(2).iloc[1]['t']))}) | "
        f"Model: {('/'.join(model_tags))}"
    )
    return msg


# =========================
# Scanner job
# =========================
async def scan_once() -> Dict[str, Any]:
    results = {"time": datetime.now(tz=TZ).isoformat(), "signals": []}
    pivots = PIVOTS or {}
    # nếu thiếu pivot, bỏ qua mã tương ứng
    for sym in TICKERS:
        try:
            if sym not in pivots:
                continue
            msg = breakout_signal(sym, pivots)
            if msg:
                # chống lặp trong ngày
                today = datetime.now(tz=TZ).strftime("%Y-%m-%d")
                if sent_today.get(sym) != today:
                    telegram_send(msg)
                    sent_today[sym] = today
                results["signals"].append(msg)
        except Exception as e:
            # không làm rơi toàn job
            results.setdefault("errors", {})[sym] = str(e)
            continue
        # nhẹ nhàng với API
        await asyncio.sleep(0.4)
    return results


# =========================
# Schedules (trading hours VN)
# =========================
scheduler = AsyncIOScheduler(timezone=str(TZ))

def add_trading_windows():
    """
    Chạy mỗi SCAN_INTERVAL_SEC trong 2 phiên:
      Sáng: 09:00–11:30
      Chiều: 13:00–15:00
    T2–T6.
    """
    step = max(SCAN_INTERVAL_SEC, 30)
    # buổi sáng
    scheduler.add_job(
        scan_once,
        CronTrigger(day_of_week="mon-fri", hour="9-11", minute=f"*/{max(step//60,1)}", second="0"),
        name="morning-scan",
        misfire_grace_time=60,
    )
    # buổi chiều
    scheduler.add_job(
        scan_once,
        CronTrigger(day_of_week="mon-fri", hour="13-15", minute=f"*/{max(step//60,1)}", second="0"),
        name="afternoon-scan",
        misfire_grace_time=60,
    )

add_trading_windows()
scheduler.start()


# =========================
# FastAPI endpoints
# =========================
@app.get("/", response_class=PlainTextResponse)
def root():
    return "VN21-scanner running. See /healthz, /config, /scan."

# UptimeRobot dùng HEAD → trả 200 OK
@app.head("/healthz", response_class=PlainTextResponse)
def healthz_head():
    return Response(status_code=200)

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "OK"

@app.get("/config", response_class=JSONResponse)
def get_config():
    return {
        "tickers": TICKERS,
        "pivots": PIVOTS,
        "plan": PLAN,
        "interval_sec": SCAN_INTERVAL_SEC,
        "tz": "Asia/Ho_Chi_Minh",
        "trading_min_per_day": TRADING_MIN_PER_DAY,
    }

@app.post("/scan", response_class=JSONResponse)
async def scan_now():
    res = await scan_once()
    return res


# (Tùy chọn) chạy cục bộ: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
