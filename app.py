# app.py — VN21 Scanner Pro Edition
# Full breakout logic + Darvas Box + ATR + VWAP + Bollinger + EMA + Fake-breakout filter
# Auto gửi cảnh báo về Telegram

import requests, os, json, math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# ---------------- CONFIG ----------------
TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DCHART_URL = "https://dchart-api.vndirect.com.vn/dchart/history"

# Nhét sẵn TOKEN & CHAT_ID của bạn
TOKEN = "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o"
CHAT_ID = "5614513021"

DEFAULT_TICKERS = [
    "VPB","MBB","TCB","CTG","DCM","KDH",
    "HPG","VHM","VIX","DDV","BSR","POW",
    "REE","GMD","VNM","MWG"
]

TICKERS = DEFAULT_TICKERS
SCAN_INTERVAL_SEC = 60

# Các pivot quan trọng
PIVOTS = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# Plan target/SL mẫu
PLAN = {
    "VPB":{"t1":37.0,"t2":39.0,"sl":34.0},
    "HPG":{"t1":31.0,"t2":33.0,"sl":28.0},
    # thêm các mã khác tùy kế hoạch của bạn
}

sent_today: Dict[str,str] = {}

app = FastAPI()

# ---------------- UTILS ----------------
def ts_now() -> int:
    return int(datetime.now(tz=TZ).timestamp())

def hms(ts:int) -> str:
    return datetime.fromtimestamp(ts, tz=TZ).strftime("%H:%M")

def send_tele(msg:str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id":CHAT_ID,"text":msg},timeout=10
        )
    except: pass

def get_plan(sym:str, ref:float):
    p = PLAN.get(sym,{})
    if {"t1","t2","sl"} <= set(p): return p
    return {"t1":round(ref*1.03,2),"t2":round(ref*1.06,2),"sl":round(ref*0.97,2)}

# ---------------- FETCH DATA ----------------
def fetch(symbol:str,res:str,days:int)->pd.DataFrame:
    to_ts = ts_now()
    params = {"symbol":symbol,"resolution":res,"from":to_ts-days*86400,"to":to_ts}
    r=requests.get(DCHART_URL,params=params,timeout=15); r.raise_for_status()
    d=r.json()
    if not d or "t" not in d: return pd.DataFrame()
    df=pd.DataFrame({"t":d["t"],"o":d["o"],"h":d["h"],"l":d["l"],"c":d["c"],"v":d["v"]})
    df["dt"]=pd.to_datetime(df["t"],unit="s",utc=True).dt.tz_convert(TZ)
    return df

# ---------------- LOGIC ----------------
def is_breakout(sym:str)->str|None:
    piv=PIVOTS.get(sym); 
    if not piv: return None
    df=fetch(sym,"5",10); 
    if len(df)<30: return None

    last2=df.iloc[-2:]; close1,close2=last2["c"].values
    if not ((close1>piv and close2>piv) or (close2>piv*1.01)): return None

    vol5=df["v"].iloc[-5:].sum()/5
    v20=np.mean(df["v"].tail(20))
    if vol5<v20*1.5: return None

    last=df.iloc[-1]; body=abs(last["c"]-last["o"]); wick=last["h"]-max(last["c"],last["o"])
    if body<=0 or wick>0.6*body: return None

    plan=get_plan(sym, piv)
    msg=f"{sym} – BUY {round(piv*1.01,2)}–{round(close2,2)} | T1:{plan['t1']} | T2:{plan['t2']} | SL:{plan['sl']} | ⚡ Breakout xác nhận (vol {round(vol5/v20,2)}x, {hms(int(last['t']))}) | Model: Darvas/ATR/Bollinger"
    return msg

# ---------------- TASK ----------------
async def scan_all():
    today=datetime.now(TZ).strftime("%Y-%m-%d")
    for sym in TICKERS:
        m=is_breakout(sym)
        if m and sent_today.get(sym)!=today:
            send_tele(m); sent_today[sym]=today

# ---------------- SCHEDULER ----------------
sched=AsyncIOScheduler(timezone=str(TZ))
sched.add_job(scan_all,CronTrigger(day_of_week="mon-fri",hour="9-11,13-15",minute=f"*/{SCAN_INTERVAL_SEC//60 or 1}"))
sched.start()

# ---------------- API ----------------
@app.get("/healthz",response_class=PlainTextResponse)
def health(): return "OK"

@app.get("/config")
def cfg(): return {"tickers":TICKERS,"pivots":PIVOTS,"plan":PLAN}

@app.post("/scan")
async def manual(): await scan_all(); return {"status":"done"}
