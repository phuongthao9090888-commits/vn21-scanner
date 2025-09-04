import os
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

# --- Telegram config ---
TELEGRAM_TOKEN = "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o"
TELEGRAM_CHAT_ID = "5614513021"

# --- Universe ---
VN21_CORE = [
    "VPB","MBB","TCB","CTG","DCM","KDH",
    "HPG","VHM","VIX","DDV","BSR","POW",
    "REE","GMD","VNM","MWG"
]

UNIVERSE_EXTRA = [
    "BID","STB","SHB","ACB","TPB","EIB","LPB","HDB",
    "SSI","HCM","VND","SHS","MBS",
    "PVD","PVS","PLX","PVG",
    "GEX","KBC","NLG","DXG",
    "HSG","NKG","KSB",
    "FPT","MSN","SAB","DGW","FRT",
    "VIC","VGI","GVR","VTP","LTG","PAN"
]

DEFAULT_TICKERS = sorted(list(set(VN21_CORE + UNIVERSE_EXTRA)))

# --- Pivots ---
PIVOTS = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# --- App init ---
app = FastAPI()

# --- Health endpoints (fix 405) ---
@app.head("/healthz")
def healthz_head():
    return Response(status_code=200)

@app.get("/healthz")
def healthz_get():
    return PlainTextResponse("ok")

@app.head("/")
def root_head():
    return Response(status_code=200)

@app.get("/")
def root_get():
    return PlainTextResponse("vn21-scanner up")

# --- Utils ---
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("Telegram error:", e)

# Giả lập data realtime (sau thay bằng VNDirect API)
def fetch_realtime_data(ticker: str):
    price = np.random.uniform(20, 120)
    volume = np.random.randint(1e5, 5e6)
    return {"ticker": ticker, "price": price, "volume": volume}

# Logic breakout nâng cao
def check_breakout(ticker, data):
    pivot = PIVOTS.get(ticker)
    if not pivot:
        return None
    price = data["price"]
    volume = data["volume"]
    # điều kiện breakout
    cond_price = price > pivot * 1.01
    cond_vol = volume > 1.5e6
    cond_wick = True  # stub: cần OHLC 5m để check wick
    if cond_price and cond_vol and cond_wick:
        return f"{ticker} – BUY {round(price,2)} | T1: {round(pivot*1.05,2)} | T2: {round(pivot*1.1,2)} | SL: {round(pivot*0.97,2)} | ⚡ Breakout xác nhận (vol={volume}, time={datetime.now().strftime('%H:%M')}) | Model: Darvas"
    # hỗ trợ vùng mua
    if price <= pivot*0.98:
        return f"{ticker} – Hỗ trợ vùng mua quanh {round(price,2)} (gần pivot {pivot})"
    # cảnh báo rủi ro (stub)
    if volume > 3e6 and price < pivot:
        return f"{ticker} – ⚠️ Rủi ro: vol bất thường, giá dưới pivot {pivot}"
    return None

# --- Scheduler task ---
def scan_market():
    alerts = []
    for t in VN21_CORE:  # quét VN21 trước
        data = fetch_realtime_data(t)
        msg = check_breakout(t, data)
        if msg:
            alerts.append(msg)
    if alerts:
        send_telegram("\n".join(alerts))

scheduler = BackgroundScheduler()
scheduler.add_job(scan_market, "interval", minutes=5)
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
