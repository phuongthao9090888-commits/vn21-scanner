# app.py — Web service + khởi động scanner dạng background + self-ping giờ giao dịch

import os
import time
import threading
from datetime import datetime, time as dtime, timedelta, timezone

import requests
from fastapi import FastAPI
from scanner import run_scanner

app = FastAPI()

# ---- health & root ----
# CHO PHÉP CẢ GET VÀ HEAD để UptimeRobot (free) không báo 405
@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return {"ok": True}

@app.get("/")
def root():
    return {"service": "vn21-scanner", "status": "live"}

# ---- market hours (VN, GMT+7) ----
TZ = timezone(timedelta(hours=7))
def market_open_now() -> bool:
    now_t = datetime.now(TZ).time()
    # mở cửa 09:00–11:30 và 13:00–15:00
    m1s, m1e = dtime(9, 0),  dtime(11, 30)
    m2s, m2e = dtime(13, 0), dtime(15, 0)
    return (m1s <= now_t <= m1e) or (m2s <= now_t <= m2e)

# ---- self-ping để Render không sleep trong giờ giao dịch ----
def self_ping_loop():
    base_url = os.getenv("RENDER_EXTERNAL_URL") or "https://vn21-scanner.onrender.com"
    url = f"{base_url.rstrip('/')}/healthz"
    while True:
        try:
            if market_open_now():
                # ping mỗi 5 phút (Render free thường idle >15' mới sleep)
                requests.get(url, timeout=10)
                time.sleep(300)
            else:
                # ngoài giờ thì nghỉ ping để tiết kiệm
                time.sleep(60)
        except Exception:
            # lỗi mạng thì chờ ngắn rồi thử lại
            time.sleep(60)

# ---- chạy scanner nền khi service khởi động ----
_started = False

@app.on_event("startup")
def start_bg():
    global _started
    if not _started:
        threading.Thread(target=run_scanner, daemon=True).start()
        threading.Thread(target=self_ping_loop, daemon=True).start()
        _started = True
