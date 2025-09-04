# app.py — Web service + nền chạy scanner
import threading
from fastapi import FastAPI
from scanner import run_scanner

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

# chạy scanner ở background khi web khởi động
_started = False
@app.on_event("startup")
def start_bg():
    global _started
    if not _started:
        th = threading.Thread(target=run_scanner, daemon=True)
        th.start()
        _started = True
