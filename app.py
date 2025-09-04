# app.py
import os, time, threading, json
import datetime as dt
from statistics import mean
from typing import Dict, Tuple

import requests
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# ========= CONFIG =========
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID   = os.getenv("CHAT_ID", "").strip()

# thời gian VN (+7). Có thể chỉnh qua env TZ_OFFSET_HOURS
TZ = dt.timezone(dt.timedelta(hours=int(os.getenv("TZ_OFFSET_HOURS", "7"))))

# tần suất quét (giây) — “nhạy vừa phải”
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))

# Pivots mặc định (có thể override bằng env PIVOTS_JSON: {"VPB":35.4,...})
DEFAULT_PIVOTS = {
    "VPB": 35.4, "MBB": 28.2, "TCB": 40.1, "CTG": 52.5,
    "DCM": 40.0, "KDH": 36.5, "HPG": 29.5, "VHM": 106.0,
    "VIX": 38.5, "DDV": 31.5, "BSR": 27.5, "POW": 16.5,
    "REE": 65.5, "GMD": 70.0, "VNM": 62.5, "MWG": 80.0,
}
TICKERS_PIVOT: Dict[str, float] = DEFAULT_PIVOTS.copy()
try:
    piv = os.getenv("PIVOTS_JSON")
    if piv:
        TICKERS_PIVOT.update(json.loads(piv))
except Exception:
    pass

# Kế hoạch T1/T2/SL (env PLAN_JSON: {"VPB":{"t1":...,"t2":...,"sl":...}, ...})
PLAN: Dict[str, Dict[str, float]] = {}
try:
    plan_env = os.getenv("PLAN_JSON")
    if plan_env:
        PLAN = json.loads(plan_env)
except Exception:
    PLAN = {}

# ========= FASTAPI =========
app = FastAPI(title="VN21 Breakout Scanner", version="1.0")

# ========= UTILS =========
def now():
    return dt.datetime.now(TZ)

def to_unix(t: dt.datetime) -> int:
    return int(t.timestamp())

def during_session(t: dt.datetime) -> bool:
    # 09:00–11:30 & 13:00–15:00, T2–T6 (giờ VN)
    wd = t.weekday()
    if wd >= 5:
        return False
    hm = t.hour * 60 + t.minute
    return (540 <= hm <= 690) or (780 <= hm <= 900)

def send_tele(text: str):
    if not (BOT_TOKEN and CHAT_ID):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=10
        )
    except Exception:
        pass

def fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")

def default_targets(pivot: float) -> Tuple[float, float, float]:
    # T1 = +3%, T2 = +6%, SL = -3% (có thể thay bằng PLAN)
    return round(pivot*1.03, 2), round(pivot*1.06, 2), round(pivot*0.97, 2)

# ========= VNDirect 5m candles =========
def fetch_5m(symbol: str, days_back=20):
    t = now()
    frm = to_unix(t - dt.timedelta(days=days_back))
    to  = to_unix(t + dt.timedelta(minutes=10))
    # thêm headers tránh 403
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; VN21-Scanner/1.0)",
        "Accept": "application/json",
        "Connection": "close",
    }
    r = requests.get(
        "https://dchart-api.vndirect.com.vn/dchart/history",
        params={"symbol": symbol, "resolution": "5", "from": frm, "to": to},
        headers=headers,
        timeout=15
    )
    r.raise_for_status()
    js = r.json()
    if not js or "t" not in js or not js["t"]:
        return []
    out = []
    for i in range(len(js["t"])):
        out.append({
            "t": js["t"][i],
            "o": js["o"][i], "h": js["h"][i],
            "l": js["l"][i], "c": js["c"][i],
            "v": js["v"][i]
        })
    return out

def no_long_upper_wick(c) -> bool:
    body = abs(c["c"] - c["o"])
    if body <= 0:
        return False
    upper = c["h"] - max(c["o"], c["c"])
    return upper <= 0.6 * body

def volume_benchmark(candles) -> float:
    vols = [c["v"] for c in candles[:-1]] or [c["v"] for c in candles]
    return (mean(vols)/5.0) if vols else 0.0

def price_cond_early(c2, pivot):
    # close > pivot hoặc vượt ~0.5%
    return (c2["c"] > pivot) or (c2["c"] >= pivot*1.005)

def price_cond_confirmed(c1, c2, pivot):
    # 2 nến đóng > pivot hoặc >+1%
    return ((c1["c"] > pivot and c2["c"] > pivot) or (c2["c"] >= pivot*1.01))

def check_signals(candles, pivot):
    if len(candles) < 2:
        return None, None, {}
    c1, c2 = candles[-2], candles[-1]
    avg_per_min = volume_benchmark(candles)
    # Early
    early = price_cond_early(c2, pivot) and (c2["v"] >= 1.2 * avg_per_min * 5)
    # Confirmed
    confirmed = price_cond_confirmed(c1, c2, pivot) and (c2["v"] >= 1.5 * avg_per_min * 5) and no_long_upper_wick(c2)
    return early, confirmed, {"c1": c1, "c2": c2, "avg_per_min": avg_per_min}

def entry_range(pivot, last_close):
    lo = pivot
    hi = max(last_close, pivot*1.008)  # trượt nhẹ 0.8%
    return (round(lo,2), round(hi,2))

def model_for(sym: str) -> str:
    if sym in {"VPB","MBB","TCB","CTG"}: return "CANSLIM"
    if sym in {"KDH","VHM"}: return "Darvas"
    return "Zanger"

# ========= STATE =========
sent_today = {"early": set(), "confirmed": set()}
current_date = now().date()
last_alerts_cache = []   # lưu 20 log gần nhất để xem ở "/"

def log_alert(s: str):
    last_alerts_cache.append(f"{now().strftime('%H:%M:%S')}  {s}")
    if len(last_alerts_cache) > 20:
        del last_alerts_cache[:len(last_alerts_cache)-20]

# ========= BACKGROUND LOOP =========
def scanner_loop():
    global sent_today, current_date
    send_tele("🚀 VN21 Scanner khởi động — chế độ NHẠY VỪA PHẢI (5m, 20s).")
    while True:
        try:
            t = now()
            if t.date() != current_date:
                sent_today = {"early": set(), "confirmed": set()}
                current_date = t.date()

            if during_session(t):
                for sym, pivot in TICKERS_PIVOT.items():
                    try:
                        bars = fetch_5m(sym, days_back=20)
                        if len(bars) < 2:
                            continue

                        early, confirmed, info = check_signals(bars, pivot)
                        c2 = info.get("c2", bars[-1])
                        last_close = c2["c"]
                        lo, hi = entry_range(pivot, last_close)
                        if sym in PLAN:
                            t1,t2,sl = PLAN[sym]["t1"], PLAN[sym]["t2"], PLAN[sym]["sl"]
                        else:
                            t1,t2,sl = default_targets(pivot)

                        vol_note = f"vol={int(c2['v']):,} vs avg5m≈{int(info['avg_per_min']*5):,}"
                        ts = dt.datetime.fromtimestamp(c2["t"], TZ).strftime("%H:%M")

                        if early and sym not in sent_today["early"]:
                            msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
                                   f"SL: {fmt(sl)} | 🔔 Early breakout ({vol_note}, {ts}) | Model: {model_for(sym)}")
                            send_tele(msg)
                            log_alert(msg)
                            sent_today["early"].add(sym)

                        if confirmed and sym not in sent_today["confirmed"]:
                            msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
                                   f"SL: {fmt(sl)} | ⚡ Breakout xác nhận ({vol_note}, {ts}) | Model: {model_for(sym)}")
                            send_tele(msg)
                            log_alert(msg)
                            sent_today["confirmed"].add(sym)
                    except Exception as e:
                        log_alert(f"[{sym}] error: {e}")
            time.sleep(POLL_SECONDS)
        except Exception as e:
            log_alert(f"Loop error: {e}")
            time.sleep(POLL_SECONDS)

# khởi chạy thread nền khi app start
@app.on_event("startup")
def _startup():
    th = threading.Thread(target=scanner_loop, daemon=True)
    th.start()

# ========= ROUTES =========
@app.get("/")
def home():
    return {
        "status": "ok",
        "server_time": now().isoformat(),
        "session_open": during_session(now()),
        "poll_seconds": POLL_SECONDS,
        "tickers": list(TICKERS_PIVOT.keys()),
        "last_alerts": last_alerts_cache[-10:]
    }

@app.get("/pivots")
def pivots():
    return TICKERS_PIVOT

@app.get("/test")
def test():
    send_tele("✅ Bot Render đã chạy ngon lành!")
    return {"ok": True}

@app.get("/trigger")
def trigger(sym: str = Query(..., description="VD: POW"), kind: str = Query("early", enum=["early","confirmed"])):
    pivot = TICKERS_PIVOT.get(sym.upper())
    if not pivot:
        return JSONResponse({"ok": False, "error": "Unknown symbol"}, status_code=400)
    t1,t2,sl = ( (PLAN[sym]["t1"], PLAN[sym]["t2"], PLAN[sym]["sl"]) if sym in PLAN else default_targets(pivot) )
    lo,hi = entry_range(pivot, pivot*1.01)
    label = "🔔 Early breakout" if kind=="early" else "⚡ Breakout xác nhận"
    msg = (f"{sym.upper()} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
           f"SL: {fmt(sl)} | {label} (mock) | Model: {model_for(sym.upper())}")
    send_tele(msg)
    log_alert(f"(mock) {msg}")
    return {"ok": True, "sent": msg}
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
# ==== Scheduler cho phiên VN ====
from apscheduler.schedulers.background import BackgroundScheduler
import datetime as dt

TZ = dt.timezone(dt.timedelta(hours=7))  # VN time

def during_session(t: dt.datetime):
    # 09:00–11:30 & 13:00–15:00, Mon–Fri
    wd = t.weekday()
    if wd >= 5: 
        return False
    hm = t.hour * 60 + t.minute
    return (540 <= hm <= 690) or (780 <= hm <= 900)

scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")
_started = False

def scan_job():
    now_t = dt.datetime.now(TZ)
    if not during_session(now_t):
        return
    # gọi hàm quét 1 lần (bạn đã có sẵn trong code)
    # ví dụ nếu bạn tách thành scanner.scan_once():
    try:
        from scanner import scan_once
        scan_once()
    except Exception as e:
        print("scan_job error:", e)

@app.on_event("startup")
def _startup():
    global _started
    if _started:
        return
    # chạy mỗi 20 giây (đổi bằng ENV POLL_SECONDS nếu bạn muốn)
    scheduler.add_job(scan_job, "interval", seconds=int(os.getenv("POLL_SECONDS", "20")), id="vn_scan", max_instances=1, coalesce=True)
    scheduler.start()
    _started = True

# Healthcheck
@app.get("/healthz")
def healthz():
    return {"ok": True, "time": dt.datetime.now(TZ).isoformat()}
from fastapi import FastAPI

app = FastAPI()

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
