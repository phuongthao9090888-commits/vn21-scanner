# app.py
from fastapi import FastAPI
import datetime as dt

app = FastAPI(title="VN21 Scanner")

@app.get("/")
def root():
    return {"service": "vn21-scanner", "status": "running"}

@app.get("/healthz")
def healthz():
    return {"ok": True, "time_utc": dt.datetime.utcnow().isoformat() + "Z"}
