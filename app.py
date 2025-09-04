# app.py — Web service + nền chạy scanner (fixed HEAD 405)

import threading
from fastapi import FastAPI, Response
from starlette.middleware.base import BaseHTTPMiddleware
from scanner import run_scanner

app = FastAPI()

# Middleware để chuyển HEAD -> GET (tránh 405 Method Not Allowed)
class HeadToGetMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "HEAD":
            request.scope["method"] = "GET"
        return await call_next(request)

app.add_middleware(HeadToGetMiddleware)

# Health check chấp nhận cả GET và HEAD
@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz():
    return Response(content='{"ok": true}', media_type="application/json")

# chạy scanner ở background khi web khởi động
_started = False

@app.on_event("startup")
def start_bg():
    global _started
    if not _started:
        th = threading.Thread(target=run_scanner, daemon=True)
        th.start()
        _started = True
