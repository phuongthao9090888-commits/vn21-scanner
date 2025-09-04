from fastapi import FastAPI, Response

app = FastAPI()

@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return Response(content="ok", media_type="text/plain")
# app.py
# VN21 Scanner – unified app (Parts 1–40 merged)
# FastAPI service + scheduler + Telegram + real-time scanner skeleton

from __future__ import annotations
import os
import time
import math
import json
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse, PlainTextResponse

# -----------------------------
# 0) Config & Constants
# -----------------------------

TZ = timezone(timedelta(hours=7))  # GMT+7
APP_NAME = "vn21-scanner"

# Telegram (đã nhúng sẵn theo yêu cầu; nếu bạn muốn dùng ENV thì sửa ở đây)
TELEGRAM_TOKEN = "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o"
TELEGRAM_CHAT_ID = "5614513021"

# Danh mục VN21 (core)
VN21_CORE = [
    "VPB","MBB","TCB","CTG","DCM","KDH",
    "HPG","VHM","VIX","DDV","BSR","POW",
    "REE","GMD","VNM","MWG"
]

# Universe mở rộng đã loại penny (lọc runtime nữa cho chắc)
UNIVERSE_EXTRA = [
    # Bank
    "BID","STB","SHB","ACB","TPB","EIB","LPB","HDB",
    # Chứng khoán
    "SSI","HCM","VND","VIX","SHS","MBS",
    # Dầu khí
    "PVD","PVS","BSR","PLX","POW","PVG",
    # BĐS/KCN
    "KDH","VHM","GEX","KBC","NLG","DXG",
    # Thép/VLXD
    "HPG","HSG","NKG","KSB",
    # Công nghệ/tiêu dùng
    "FPT","MWG","VNM","MSN","SAB","DGW","FRT",
    # Khác vốn hoá lớn
    "VIC","VGI","REE","GMD","GVR","VTP","LTG","PAN"
]

DEFAULT_TICKERS = sorted(list(set(VN21_CORE + UNIVERSE_EXTRA)))

# Pivot tay ưu tiên (VN21); mã khác dùng Darvas
PIVOTS: Dict[str, float] = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# Targets/SL ví dụ – đặt theo kế hoạch gần nhất của bạn
PLAN_TARGETS: Dict[str, Dict[str, float]] = {
    # ví dụ: "VPB": {"t1": 36.5, "t2": 38.0, "sl": 34.2}
}

# -----------------------------
# 1) Utilities
# -----------------------------

def now_ts() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def is_trading_time(dt: Optional[datetime]=None) -> bool:
    d = dt or datetime.now(TZ)
    if d.weekday() >= 5:  # T7, CN
        return False
    hm = d.strftime("%H%M")
    # Sáng: 08:55–11:35 | Chiều: 12:55–15:05
    return ("0855" <= hm <= "1135") or ("1255" <= hm <= "1505")

def reject_penny(df: pd.DataFrame, min_price: float = 10.0) -> pd.DataFrame:
    if "price" in df.columns:
        return df[df["price"] >= min_price].copy()
    return df

async def tg_send(msg: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass

# -----------------------------
# 2) Data fetchers (VNDirect skeleton)
# -----------------------------

def vnd_stock_realtime(symbols: List[str]) -> pd.DataFrame:
    """
    Skeleton realtime fetcher cho VNDirect.
    Bạn có thể tối ưu bằng websocket/stream sau. Ở đây dùng REST placeholder.
    Trả về DataFrame cột: ticker, price, volume, time
    """
    rows = []
    for s in symbols:
        try:
            # TODO: thay bằng endpoint chính thức/caching của bạn
            # Placeholder: lấy 1 quote giả định qua một nguồn REST public (bạn gắn vào)
            # Ở đây mình để khung; bạn nối API thật vào return rows.
            rows.append({
                "ticker": s,
                "price": np.nan,      # cần fill data thật
                "volume": np.nan,     # cần fill data thật
                "time": now_ts()
            })
        except Exception:
            rows.append({"ticker": s, "price": np.nan, "volume": np.nan, "time": now_ts()})
    df = pd.DataFrame(rows)
    return df

def vnd_intraday_5m(symbol: str, lookback: int = 60) -> pd.DataFrame:
    """
    Skeleton tải nến 5m; cần thay bằng API thật của bạn.
    Trả về cột: time, open, high, low, close, volume
    """
    # stub: tạo khung rỗng – để app chạy được, không lỗi
    cols = ["time","open","high","low","close","volume"]
    return pd.DataFrame(columns=cols)

# -----------------------------
# 3) Indicators & models
# -----------------------------

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def darvas_pivot(df: pd.DataFrame, lookback_bars: int = 40) -> Optional[float]:
    if len(df) < lookback_bars:
        return None
    window = df.tail(lookback_bars)
    # kháng cự: swing high gần nhất
    return float(window["high"].rolling(3, center=True).apply(lambda x: x[1] if x[1] == max(x) else np.nan).dropna().tail(1).values[0]) if window.shape[0] >= 3 and not window["high"].empty else None

def long_upper_wick(close: float, high: float, low: float, open_: float, threshold: float = 0.6) -> bool:
    body = abs(close - open_)
    if body == 0:  # doji, coi như wick dài
        return True
    upper = max(high - max(close, open_), 0.0)
    return upper >= threshold * body

def vol_burst(vol_5m: float, avg_per_min_20d: float, factor: float = 1.5) -> bool:
    # 5 phút ~ 5 * avg_per_min
    return vol_5m >= factor * 5.0 * avg_per_min_20d

# -----------------------------
# 4) Strategy rules (Breakout/Buy zone/Risk)
# -----------------------------

def check_breakout(df5: pd.DataFrame, pivot: float) -> Tuple[bool, Dict]:
    """
    Điều kiện STRONG BREAKOUT:
    (1) Đóng 2 nến 5m liên tiếp trên pivot hoặc close > pivot*1.01
    (2) Volume 5m >= 1.5x 20d per-minute
    (3) Không có râu trên dài > 60% body
    """
    info = {"why": []}
    if df5.shape[0] < 3 or pd.isna(pivot):
        return False, {"why": ["not_enough_data_or_pivot"]}

    last2 = df5.tail(2).copy()
    # (1)
    cond1 = (last2["close"] > pivot).all() or (last2["close"].iloc[-1] >= pivot * 1.01)
    if not cond1:
        info["why"].append("price_not_confirmed")

    # (2) – stub: chưa có avg_per_min_20d => tạm so với SMA(20) volume 5m
    v_sma = df5["volume"].rolling(20).mean().iloc[-1] if df5["volume"].notna().any() else np.nan
    cond2 = False if (np.isnan(v_sma) or v_sma == 0) else (last2["volume"].iloc[-1] >= 1.5 * v_sma)
    if not cond2:
        info["why"].append("volume_not_confirmed")

    # (3)
    last = df5.iloc[-1]
    wick_bad = long_upper_wick(last["close"], last["high"], last["low"], last["open"])
    cond3 = not wick_bad
    if not cond3:
        info["why"].append("long_upper_wick")

    ok = cond1 and cond2 and cond3
    return ok, info

def check_buy_zone(price: float, pivot: float, band: float = 0.02) -> bool:
    if any(pd.isna([price, pivot])): return False
    return pivot*(1 - band) <= price <= pivot*(1 + 0.003)  # gần pivot -2% tới +0.3%

def detect_risk_reversal(df5: pd.DataFrame) -> Optional[str]:
    if df5.shape[0] < 2: return None
    last = df5.iloc[-1]
    prev = df5.iloc[-2]
    # ví dụ đơn giản: Bearish Engulfing / râu trên lớn
    if last["close"] < last["open"] and last["open"] > prev["close"] and last["close"] < prev["open"]:
        return "Bearish engulfing"
    if long_upper_wick(last["close"], last["high"], last["low"], last["open"], threshold=0.6):
        return "Long upper wick"
    return None

# -----------------------------
# 5) Core scan loop (skeleton)
# -----------------------------

def compute_entry_targets_sl(ticker: str, pivot: float, last_close: float) -> Tuple[str, float, float, float]:
    # Range vào lệnh quanh pivot + small buffer
    entry = f"{pivot*1.001:.2f}–{max(last_close, pivot*1.01):.2f}"
    if ticker in PLAN_TARGETS:
        t1, t2, sl = PLAN_TARGETS[ticker]["t1"], PLAN_TARGETS[ticker]["t2"], PLAN_TARGETS[ticker]["sl"]
    else:
        # fallback: target ~ +3% / +6% từ pivot, SL ~ -2.5%
        t1 = pivot * 1.03
        t2 = pivot * 1.06
        sl = pivot * 0.975
    return entry, t1, t2, sl

def choose_model_tag(ticker: str, df5: pd.DataFrame, pivot: float) -> str:
    # stub: nếu pivot là tay => "Darvas/CANSLIM/Zanger" dựa pattern đơn giản
    return "Darvas" if ticker not in PIVOTS else "Zanger"

def one_line_signal(ticker: str, entry: str, t1: float, t2: float, sl: float, vol: Optional[float], tm: str, model: str) -> str:
    s = f"{ticker} – BUY {entry} | T1: {t1:.2f} | T2: {t2:.2f} | SL: {sl:.2f}"
    if vol: s += f" | ⚡ Breakout xác nhận (vol≈{vol:.0f}, {tm})"
    s += f" | Model: {model}"
    return s

def scan_once(tickers: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    # realtime quote (placeholder)
    q = vnd_stock_realtime(tickers)
    q = reject_penny(q, min_price=10.0)

    for s in q["ticker"]:
        try:
            df5 = vnd_intraday_5m(s, lookback=120)
            if df5.empty:
                continue

            pivot = PIVOTS.get(s)
            if pivot is None:
                pv = darvas_pivot(df5)
                pivot = pv if pv else np.nan

            last_close = df5["close"].iloc[-1] if "close" in df5.columns and not df5.empty else np.nan

            # Signals
            brk, why = check_breakout(df5, pivot)
            buy_zone = check_buy_zone(last_close, pivot, band=0.02)
            risk = detect_risk_reversal(df5)

            entry, t1, t2, sl = compute_entry_targets_sl(s, pivot, last_close)
            model = choose_model_tag(s, df5, pivot)

            out[s] = {
                "price": float(last_close) if not pd.isna(last_close) else None,
                "pivot": float(pivot) if not pd.isna(pivot) else None,
                "breakout": brk, "why": why.get("why", []),
                "buy_zone": buy_zone, "risk": risk,
                "entry": entry, "t1": float(t1), "t2": float(t2), "sl": float(sl),
                "model": model,
                "time": now_ts()
            }
        except Exception as e:
            out[s] = {"error": str(e), "time": now_ts()}
    return out

# -----------------------------
# 6) FastAPI app & routes
# -----------------------------

app = FastAPI(title=APP_NAME)

# Healthz – hỗ trợ GET & HEAD để không còn 405
@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz_any():
    return JSONResponse({"status": "ok", "ts": now_ts()})

# Root
@app.get("/")
def root():
    return {"service": APP_NAME, "ts": now_ts(), "tickers": DEFAULT_TICKERS[:10], "note": "VN21 scanner alive"}

# Scan API đơn giản
@app.get("/scan")
def scan_api():
    res = scan_once(DEFAULT_TICKERS)
    return res

# Webhook thủ công để gửi 1 báo cáo nhanh
@app.post("/notify-once")
def notify_once():
    res = scan_once(VN21_CORE)
    lines = []
    for s, d in res.items():
        if d.get("breakout"):
            ln = one_line_signal(
                s, d["entry"], d["t1"], d["t2"], d["sl"],
                vol=None, tm=d["time"], model=d["model"]
            )
            lines.append(ln)
        elif d.get("buy_zone"):
            lines.append(f"{s} – VỀ VÙNG MUA quanh pivot {d['pivot']:.2f} | SL: {d['sl']:.2f}")
        elif d.get("risk"):
            lines.append(f"{s} – CẢNH BÁO RỦI RO: {d['risk']}")

    if not lines:
        lines = ["Không có tín hiệu mạnh ở VN21 (skeleton)."]
    asyncio.create_task(tg_send("\n".join(lines)))
    return {"sent": True, "lines": lines, "ts": now_ts()}

# -----------------------------
# 7) Scheduler (optional, stub)
# -----------------------------
# Nếu muốn chạy định kỳ, bạn có thể bật APScheduler/BackgroundTasks.
# Render free có thể sleep; dùng UptimeRobot ping GET /healthz để giữ ấm.
