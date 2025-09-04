import threading
from fastapi import FastAPI
from scanner import run_loop

app = FastAPI()

# healthcheck
@app.get("/healthz")
def healthz():
    return {"ok": True}

# chạy scanner ở background khi app khởi động
def _start():
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

_start()
