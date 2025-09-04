# scanner.py
# VN21 Scanner – unified (Parts 1–40 merged)
# FastAPI service + scheduler + Telegram + realtime breakout scanner

import os
import time
import math
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# -------------------------
# Config & Constants
# -------------------------
TZ = timezone(timedelta(hours=7))  # GMT+7
APP_NAME = "vn21-scanner"

# Token/chat_id của bạn (đã yêu cầu điền trực tiếp)
TELEGRAM_TOKEN = "8207349630:AAFQ1Sq8eumEtNoNINSg4DboQ-SMzBLui95o"
TELEGRAM_CHAT_ID = "5614513021"

# Pivots theo kế hoạch
PIVOTS = {
    "VPB": 35.4, "MBB": 28.2, "TCB": 40.1, "CTG": 52.5,
    "DCM": 40.0, "KDH": 36.5, "HPG": 29.5, "VHM": 106,
    "VIX": 38.5, "DDV": 31.5, "BSR": 27.5, "POW": 16.5,
    "REE": 65.5, "GMD": 70.0, "VNM": 62.5, "MWG": 80.0,
}
TICKERS = list(PIVOTS.keys())

VND_HIST_API = "https://dchart-api.vndirect.com.vn/dchart/history"

# -------------------------
# Telegram helper
# -------------------------
def send_telegram(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram error:", e)


# -------------------------
# Data fetch from VNDirect
# -------------------------
def fetch_5m_df(symbol: str, limit: int = 120) -> pd.DataFrame:
    url = f"{VND_HIST_API}?resolution=5&symbol={symbol}&from=0&to=9999999999"
    r = requests.get(url, timeout=10)
    data = r.json()
    if not data or "t" not in data:
        return pd.DataFrame()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["t"], unit="s"),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"],
    })
    return df.tail(limit)


# -------------------------
# Indicators & helpers
# -------------------------
def _upper_wick_ratio(row) -> float:
    body = abs(row["close"] - row["open"])
    if body == 0:
        return 0
    return (row["high"] - max(row["close"], row["open"])) / body

def _rolling_vol(df: pd.DataFrame, window: int = 20):
    vol_per_min = df["volume"] / 5.0
    return vol_per_min.rolling(window).mean()

def _atr(df: pd.DataFrame, window: int = 14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _darvas_box_break(df: pd.DataFrame) -> bool:
    box_high = df["high"].tail(5).max()
    last_close = df["close"].iloc[-1]
    return last_close > box_high


# -------------------------
# Strategy scoring
# -------------------------
def score_by_strategies(df: pd.DataFrame, pivot: float) -> dict:
    if df is None or df.empty or len(df) < 20:
        return {}

    last2 = df.iloc[-2:]
    last = last2.iloc[-1]
    prev = last2.iloc[-2]

    signals = {}

    # (1) Break pivot
    if last["close"] > pivot * 1.01 or (
        prev["close"] > pivot and last["close"] > pivot
    ):
        signals["pivot_break"] = True

    # (2) Volume spike
    ma20 = _rolling_vol(df).iloc[-1]
    if (last["volume"] / 5.0) >= 1.5 * ma20:
        signals["vol_spike"] = True

    # (3) Wick filter
    if _upper_wick_ratio(last) <= 0.6:
        signals["no_long_wick"] = True

    # (4) ATR breakout
    atr = _atr(df).iloc[-1]
    if atr and (last["close"] - prev["close"]) > 0.5 * atr:
        signals["atr_break"] = True

    # (5) Darvas Box breakout
    if _darvas_box_break(df):
        signals["darvas"] = True

    # strong breakout
    if {"pivot_break", "vol_spike", "no_long_wick"} <= signals.keys():
        signals["STRONG_BREAKOUT"] = True

    return signals


# -------------------------
# Analyzer
# -------------------------
def analyze_symbol(symbol: str, pivot: float):
    df = fetch_5m_df(symbol)
    sigs = score_by_strategies(df, pivot)
    if not sigs.get("STRONG_BREAKOUT"):
        return None

    last = df.iloc[-1]
    entry = f"{last['close']:.2f}"
    t1 = f"{pivot * 1.05:.2f}"
    t2 = f"{pivot * 1.10:.2f}"
    sl = f"{pivot * 0.97:.2f}"
    model = "Darvas/CANSLIM/Zanger"

    text = (
        f"{symbol} – BUY {entry} | "
        f"T1: {t1} | T2: {t2} | SL: {sl} | "
        f"⚡ Breakout xác nhận (vol, time) | "
        f"Model: {model}"
    )
    return text


# -------------------------
# Scanner loop
# -------------------------
def run_scanner():
    last_push = {}
    cooldown_sec = 600  # 10 phút
    while True:
        try:
            now = datetime.now(TZ)
            if 9 <= now.hour < 15:  # trong giờ giao dịch
                for s in TICKERS:
                    try:
                        text = analyze_symbol(s, PIVOTS[s])
                        if text:
                            last_t = last_push.get(s, 0.0)
                            now_ts = time.time()
                            if now_ts - last_t >= cooldown_sec:
                                send_telegram(text)
                                last_push[s] = now_ts
                    except Exception:
                        pass
                time.sleep(20)
            else:
                time.sleep(60)
        except Exception:
            time.sleep(10)
