# app.py  — VN21 Scanner (FastAPI + APScheduler + VNDirect)
# Author: VN21
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# ==========
# Config
# ==========

# Telegram (có thể override bằng ENV; default dùng thông tin bạn đã cấp)
TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5614513021")

# VN21 (ưu tiên theo dõi/giữ)
VN21_CORE: List[str] = [
    "VPB","MBB","TCB","CTG","DCM","KDH",
    "HPG","VHM","VIX","DDV","BSR","POW",
    "REE","GMD","VNM","MWG"
]

# Universe mở rộng (mid/large cap). Penny sẽ lọc ở runtime.
UNIVERSE_EXTRA: List[str] = [
    # Ngân hàng
    "BID","VCB","STB","ACB","TPB","EIB","LPB","HDB",
    # Chứng khoán
    "SSI","HCM","VND","VIX","SHS","MBS","FTS",
    # Dầu khí / Điện
    "PVD","PVS","BSR","PLX","GAS","POW","PPC",
    # BĐS/KCN
    "VHM","VIC","VRE","KDH","NLG","KBC","GEX","DXG",
    # Thép / VLXD
    "HPG","HSG","NKG","KSB",
    # Tiêu dùng / Công nghệ / Bán lẻ
    "VNM","MSN","SAB","FPT","MWG","DGW","FRT",
    # Hạ tầng / CN
    "REE","GMD","GVR","VTP","VGI","LTG","PAN"
]

DEFAULT_TICKERS: List[str] = sorted(list(set(VN21_CORE + UNIVERSE_EXTRA)))

# Pivots tay cho VN21; mã khác dùng Darvas/ATR fallback
PIVOTS: Dict[str, float] = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# Penny filter (theo giá hiện tại)
MIN_PRICE = 10.0

# Breakout params
WICK_RATIO_LIMIT = 0.60      # upper wick <= 60% thân nến
VOL_RATIO_BREAK = 1.5        # vol/ phút >= 1.5× avg
CLOSE_PCT_FORCE = 0.01       # >+1% trên pivot
SUPPORT_BAND = 0.02          # ~ 2% dưới pivot

TZ = pytz.timezone("Asia/Ho_Chi_Minh")

# ==========
# Utils
# ==========

def now_vn() -> datetime:
    return datetime.now(TZ)

def tstamp() -> int:
    return int(time.time())

def telegram_send(text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

def fmt_price(x: Optional[float]) -> str:
    return "-" if x is None else (f"{x:.2f}".rstrip("0").rstrip("."))

def in_trading_session(dt: Optional[datetime] = None) -> bool:
    """08:55–11:35 & 12:55–15:05 VN time"""
    dt = dt or now_vn()
    hm = dt.hour * 60 + dt.minute
    am_open, am_close = 8*60+55, 11*60+35
    pm_open, pm_close = 12*60+55, 15*60+5
    return (am_open <= hm <= am_close) or (pm_open <= hm <= pm_close)

# ==========
# VNDirect fetchers
# ==========

def fetch_5m_candles(symbol: str, nbars: int = 120) -> Optional[pd.DataFrame]:
    """
    Lấy nến 5' gần nhất từ VNDirect dchart API
    """
    try:
        to_ts   = tstamp()
        frm_ts  = to_ts - 60*60*24*7   # 7 ngày ~ đủ cho 120 nến
        url = "https://dchart-api.vndirect.com.vn/dchart/history"
        params = {"symbol": symbol, "resolution": "5", "from": frm_ts, "to": to_ts}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data or "t" not in data:
            return None
        df = pd.DataFrame({
            "ts": data["t"],
            "open": data.get("o", []),
            "high": data.get("h", []),
            "low":  data.get("l", []),
            "close":data.get("c", []),
            "volume": data.get("v", []),
        })
        df["dt"] = pd.to_datetime(df["ts"], unit="s").dt.tz_localize("UTC").dt.tz_convert(TZ)
        df = df.dropna().tail(nbars).reset_index(drop=True)
        return df
    except Exception as e:
        logging.exception(f"fetch_5m_candles({symbol}) failed: {e}")
        return None

def last_traded_price(df: pd.DataFrame) -> Optional[float]:
    try:
        return float(df["close"].iloc[-1])
    except Exception:
        return None

# ==========
# Pivot engines (Darvas/ATR fallback)
# ==========

def atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].shift(1).fillna(df["close"]).values
    tr = np.maximum(h-l, np.maximum(np.abs(h-c), np.abs(l-c)))
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])

def darvas_pivot(df: pd.DataFrame) -> Optional[float]:
    """Lấy swing high gần nhất như 1 pivot box top"""
    try:
        highs = df["high"]
        # Đi tìm local maxima (tam giác 5 nến)
        for i in range(len(highs)-6, 5, -1):
            window = highs[i-3:i+3]
            if highs.iloc[i] == window.max():
                return float(highs.iloc[i])
        return float(highs.iloc[-20:].max())
    except Exception:
        return None

def get_pivot(symbol: str, df: pd.DataFrame) -> Optional[float]:
    if symbol in PIVOTS:
        return PIVOTS[symbol]
    pv = darvas_pivot(df)
    if pv is None:
        return None
    # tinh chỉnh bằng ATR nhỏ
    _atr = atr(df, 14)
    return round(pv, 2) if _atr is None else round(pv, 2)

# ==========
# Pattern / Volume logic
# ==========

def upper_wick_ratio(o: float, h: float, l: float, c: float) -> float:
    body = abs(c - o)
    if body <= 1e-6:
        return 1.0  # coi như wick dài
    wick_up = max(0.0, h - max(o, c))
    return wick_up / body

def bearish_reversal(o: float, h: float, l: float, c: float, prev_o: float, prev_c: float) -> bool:
    # Shooting star: wick trên dài & close < open
    shoot = (upper_wick_ratio(o, h, l, c) > 0.8) and (c < o)
    # Bearish engulfing
    engulf = (c < o) and (prev_c > prev_o) and (o >= prev_c) and (c <= prev_o)
    return shoot or engulf

def vol_per_min(vol_5m: float) -> float:
    return float(vol_5m) / 5.0

# ==========
# Core signal
# ==========

def analyze_symbol(symbol: str) -> Optional[str]:
    df = fetch_5m_candles(symbol)
    if df is None or len(df) < 25:
        return None

    last = last_traded_price(df)
    if last is None or last < MIN_PRICE:   # bỏ penny theo runtime
        return None

    # Vol averages
    vpm = vol_per_min(df["volume"].iloc[-1])
    vpm_avg = vol_per_min(df["volume"].tail(20).mean())

    # 2 nến gần nhất
    o1, h1, l1, c1, v1, t1 = df.iloc[-1][["open","high","low","close","volume","dt"]]
    o2, h2, l2, c2, v2, t2 = df.iloc[-2][["open","high","low","close","volume","dt"]]

    # Pivot
    pivot = get_pivot(symbol, df)
    if pivot is None:
        return None

    # Flags
    wick_ok = (upper_wick_ratio(o1, h1, l1, c1) <= WICK_RATIO_LIMIT)
    vol_ok  = (vpm >= VOL_RATIO_BREAK * vpm_avg)

    close_force = (c1 >= pivot * (1 + CLOSE_PCT_FORCE))
    two_closes  = (c2 >= pivot) and (c1 >= pivot)

    breakout = (wick_ok and vol_ok and (close_force or two_closes))

    # Hỗ trợ vùng mua (giá về gần pivot -2%)
    support_buy = (pivot * (1 - SUPPORT_BAND) <= c1 < pivot)

    # Cảnh báo rủi ro: nến đảo chiều + vol spike
    risk = bearish_reversal(o1, h1, l1, c1, o2, c2) and (vpm >= 1.8 * vpm_avg)

    # Targets/SL (đơn giản hóa; nếu có plan riêng thì map từ ENV/Sheets)
    atr14 = atr(df, 14) or 0.0
    entry_low  = round(max(pivot, min(c1, o1)) * 0.999, 2)
    entry_high = round(max(c1, o1), 2)
    t1 = round(c1 + 1.0 * atr14, 2)
    t2 = round(c1 + 2.0 * atr14, 2)
    sl = round(pivot * 0.97, 2)

    model = "Darvas" if symbol not in PIVOTS else "Pivot+CANSLIM"

    # Compose messages
    if breakout:
        note = (f"{symbol} – BUY {fmt_price(entry_low)}–{fmt_price(entry_high)} | "
                f"T1: {fmt_price(t1)} | T2: {fmt_price(t2)} | SL: {fmt_price(sl)} | "
                f"⚡ Breakout xác nhận (vol {vpm/vpm_avg:.1f}×, {t1 if isinstance(t1,str) else df['dt'].iloc[-1].strftime('%H:%M')}) | "
                f"Model: {model}")
        return note

    if support_buy:
        note = (f"{symbol} – HỖ TRỢ VÙNG MUA quanh {fmt_price(pivot)} "
                f"(close {fmt_price(c1)} ~ {100*(pivot-c1)/pivot:.1f}% dưới) | "
                f"Entry: {fmt_price(entry_low)}–{fmt_price(entry_high)} | SL: {fmt_price(sl)} | "
                f"Model: {model}")
        return note

    if risk:
        note = (f"{symbol} – ⚠️ CẢNH BÁO RỦI RO (đảo chiều/vol bất thường: {vpm/vpm_avg:.1f}×) | "
                f"Close: {fmt_price(c1)} | Pivot: {fmt_price(pivot)} | "
                f"Xem xét hạ tỷ trọng/đợi xác nhận lại")
        return note

    return None

def scan_all(tickers: List[str]) -> List[str]:
    msgs = []
    for s in tickers:
        try:
            m = analyze_symbol(s)
            if m:
                msgs.append(m)
        except Exception as e:
            logging.exception(f"analyze_symbol({s}) failed: {e}")
    return msgs

# ==========
# FastAPI
# ==========

app = FastAPI(title="VN21 Scanner", version="1.0")

@app.get("/")
def root():
    return {"service": "vn21-scanner", "time": now_vn().isoformat()}

@app.get("/healthz")
def healthz():
    # Trả 200 cho GET (UptimeRobot), HEAD đã có route riêng
    return {"status": "ok", "time": now_vn().isoformat()}

@app.head("/healthz")
def healthz_head():
    # Render/UptimeRobot hay dùng HEAD -> trả 200 OK
    return Response(status_code=200)

@app.get("/scan")
def scan_endpoint():
    msgs = scan_all(DEFAULT_TICKERS)
    return {"count": len(msgs), "messages": msgs}

# ==========
# Scheduler (5 phút trong giờ giao dịch VN)
# ==========

def scheduled_job():
    try:
        if not in_trading_session():
            return
        msgs = scan_all(DEFAULT_TICKERS)
        if msgs:
            # gửi gộp 1 lần cho đỡ spam
            telegram_send("📈 <b>VN21 – Tín hiệu chiến lược (5’)</b>\n" + "\n".join(msgs))
    except Exception as e:
        logging.exception(f"scheduled_job failed: {e}")

scheduler = BackgroundScheduler(timezone=str(TZ))
# mỗi 5 phút, mọi ngày; bên trong sẽ tự check giờ giao dịch VN
scheduler.add_job(scheduled_job, CronTrigger.from_crontab("*/5 * * * *"))
scheduler.start()

# ==========
# Local run (Render dùng Uvicorn qua Procfile)
# ==========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
