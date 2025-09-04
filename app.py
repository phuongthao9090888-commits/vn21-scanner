# app.py — Web service + khởi động scanner dạng background

import threading
from fastapi import FastAPI
from scanner import run_scanner

app = FastAPI()

# CHO PHÉP CẢ GET VÀ HEAD để UptimeRobot bản free không báo 405
@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return {"ok": True}

@app.get("/")
def root():
    return {"service": "vn21-scanner", "status": "live"}

# chạy scanner nền khi service khởi động
_started = False

@app.on_event("startup")
def start_bg():
    global _started
    if not _started:
        th = threading.Thread(target=run_scanner, daemon=True)
        th.start()
        _started = True
