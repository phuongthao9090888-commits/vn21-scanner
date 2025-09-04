# scanner.py — VN21 Scanner (skeleton ổn định, không lỗi)

from __future__ import annotations
import os, time, json
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Dict, Optional
import requests

# ===== Config chung =====
TZ = timezone(timedelta(hours=7))  # GMT+7
APP_NAME = "vn21-scanner"

# Telegram — ĐÃ CẮM CỨNG token/chat_id của bạn
TELEGRAM_TOKEN = "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o"
TELEGRAM_CHAT_ID = "5614513021"

def tg_send(text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass

# ===== Universe (đã loại penny) =====
VN21_CORE = ["VPB","MBB","TCB","CTG","DCM","KDH","HPG","VHM","VIX","DDV","BSR","POW","REE","GMD","VNM","MWG"]
UNIVERSE_EXTRA = [
    "BID","STB","SHB","ACB","TPB","EIB","LPB","HDB",
    "SSI","HCM","VND","SHS","MBS",
    "PVD","PVS","PLX",
    "GEX","KBC","NLG","DXG",
    "HSG","NKG","KSB",
    "FPT","MSN","SAB","DGW","FRT","VIC","VGI","GVR","VTP","LTG","PAN"
]
TICKERS = sorted(list({*VN21_CORE, *UNIVERSE_EXTRA}))

PIVOTS: Dict[str, float] = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# ===== Thời gian giao dịch VN =====
SESSIONS = [
    (dtime(8,55),  dtime(11,35)),
    (dtime(12,55), dtime(15, 5)),
]
def market_open_now(dt: Optional[datetime] = None) -> bool:
    dt = dt or datetime.now(TZ)
    t = dt.time()
    if dt.weekday() > 4:
        return False
    return any(start <= t <= end for start, end in SESSIONS)

# ===== Lấy giá/vol từ VNDirect =====
def fetch_quote_vndirect(symbol: str) -> Optional[dict]:
    try:
        url = "https://finfo-api.vndirect.com.vn/v4/stock_prices"
        params = {"symbols": symbol, "sort": "-time", "size": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        x = data[0]
        price = float(x.get("lastPrice") or x.get("matchPrice") or 0)
        vol   = int(float(x.get("nmVolume") or x.get("matchVolume") or 0))
        tstr  = x.get("time") or ""
        return {"symbol": symbol, "price": price, "volume": vol, "time": tstr}
    except Exception:
        return None

# ===== Rule đơn giản (skeleton) =====
def simple_watch(symbol: str, quote: dict) -> Optional[str]:
    pv = PIVOTS.get(symbol)
    if pv and quote["price"] >= pv * 1.01:
        return f"{symbol} vượt pivot {pv} (≈+1%) @ {quote['price']} lúc {quote['time']}"
    return None

# ===== Scanner loop =====
def run_scanner():
    tg_send("🚀 VN21 Scanner đã khởi động.")
    last_ping = 0
    while True:
        try:
            now = datetime.now(TZ)
            if now.timestamp() - last_ping > 600:
                last_ping = now.timestamp()
                print(f"[{now}] heartbeat alive")

            if not market_open_now(now):
                time.sleep(30)
                continue

            for symbol in TICKERS:
                q = fetch_quote_vndirect(symbol)
                if not q:
                    continue
                msg = simple_watch(symbol, q)
                if msg:
                    tg_send(f"⚡ {msg}")
                time.sleep(0.8)

        except Exception as e:
            print("scanner error:", e)
            time.sleep(3)
