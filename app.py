# app.py — Web service + khởi động scanner ở background
import threading
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from scanner import run_scanner  # toàn bộ logic nằm ở scanner.py

app = FastAPI()

# healthz: hỗ trợ cả GET và HEAD để tránh 405 từ UptimeRobot
@app.get("/healthz")
def healthz_get():
    return JSONResponse({"ok": True})

@app.head("/healthz")
def healthz_head():
    # Trả rỗng + 200 OK cho HEAD
    return PlainTextResponse("", status_code=200)

# chạy scanner ở background khi web khởi động
_started = False
@app.on_event("startup")
def start_bg():
    global _started
    if not _started:
        th = threading.Thread(target=run_scanner, daemon=True)
        th.start()
        _started = True
