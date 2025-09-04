# app.py — Web service + khởi động scanner
import threading
from fastapi import FastAPI, Response
from scanner import run_scanner

app = FastAPI()

# Healthcheck: hỗ trợ cả GET và HEAD
@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    # HEAD không cần body -> trả 200 rỗng
    return Response(content=b'{"ok": true}', media_type="application/json")

# (tuỳ chọn) thêm root cho dễ test trên trình duyệt
@app.get("/")
def root():
    return {"service": "vn21-scanner", "status": "ok"}

# chạy scanner ở background khi web khởi động
_started = False
@app.on_event("startup")
def start_bg():
    global _started
    if not _started:
        th = threading.Thread(target=run_scanner, daemon=True)
        th.start()
        _started = True
