# app.py — FastAPI + chạy scanner nền
import threading
from fastapi import FastAPI
from scanner import loop_poll, now, is_trading_time

app = FastAPI(title="VN21 Scanner")

# Root endpoint: hiển thị thông tin service
@app.get("/")
def root():
    return {
        "service": "vn21-scanner",
        "status": "running",
        "server_time": now().isoformat(),
        "session_open": is_trading_time()
    }

# Health check (dùng cho UptimeRobot / GitHub Actions)
@app.get("/healthz")
def healthz():
    return {"ok": True}

# Khởi chạy vòng quét ở background khi service bật
_started = False
@app.on_event("startup")
def start_worker():
    global _started
    if _started:
        return
    t = threading.Thread(target=loop_poll, daemon=True)
    t.start()
    _started = True
