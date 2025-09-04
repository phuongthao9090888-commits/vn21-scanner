# app.py — VN21 Pro Scanner (FastAPI + APScheduler + VNDirect)
# Full: Breakout mạnh + Hỗ trợ vùng mua + Fake breakout + ATR targets/SL
# Patterns: Cup&Handle, Double Bottom, Flag/Pennant (heuristics gọn)
# Non-penny filter (last close < 10.0 loại)
# Sector & breadth cơ bản
# /healthz hỗ trợ GET + HEAD (Render/UptimeRobot)
# Telegram đã nhúng sẵn TOKEN/CHAT_ID của bạn (có thể override bằng ENV)

import os, math, time, json, logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# =========================
# Telegram (hard-coded; có thể override bằng ENV)
# =========================
TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5614513021")

TG_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

# =========================
# Timezone & Session
# =========================
TZ = pytz.timezone("Asia/Ho_Chi_Minh")

def now_vn() -> datetime:
    return datetime.now(TZ)

def in_trading_session(dt: Optional[datetime] = None) -> bool:
    dt = dt or now_vn()
    if dt.weekday() >= 5:  # T7-CN
        return False
    hm = dt.hour * 60 + dt.minute
    # 08:55–11:35 và 12:55–15:05
    return (535 <= hm <= 695) or (775 <= hm <= 905)

# =========================
# Universe & Pivots
# =========================
VN21_CORE: List[str] = [
    "VPB","MBB","TCB","CTG","DCM","KDH",
    "HPG","VHM","VIX","DDV","BSR","POW",
    "REE","GMD","VNM","MWG"
]

UNIVERSE_EXTRA: List[str] = [
    # Bank
    "BID","VCB","STB","ACB","TPB","EIB","LPB","HDB",
    # CK
    "SSI","HCM","VND","VIX","SHS","MBS","FTS",
    # Dầu khí / Điện
    "PVD","PVS","BSR","PLX","GAS","POW","PPC",
    # BĐS/KCN
    "VHM","VIC","VRE","KDH","NLG","KBC","GEX","DXG",
    # Thép / VLXD
    "HPG","HSG","NKG","KSB",
    # Tiêu dùng / Công nghệ / Bán lẻ
    "VNM","MSN","SAB","FPT","MWG","DGW","FRT",
    # Hạ tầng / CN
    "REE","GMD","GVR","VTP","VGI","LTG","PAN"
]

DEFAULT_TICKERS: List[str] = sorted(list(set(VN21_CORE + UNIVERSE_EXTRA)))

# Pivots tay cho VN21; mã khác fallback Darvas
PIVOTS: Dict[str, float] = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# Penny filter theo giá hiện tại (nghìn VND)
MIN_PRICE = 10.0
TRADING_MIN_PER_DAY = 270  # ~ 9:00–11:30, 13:00–15:00

# =========================
# VNDirect fetchers
# =========================
DCHART_URL = "https://dchart-api.vndirect.com.vn/dchart/history"

def _now_ts() -> int:
    return int(time.time())

def fetch_candles(symbol: str, resolution: str, days: int) -> Optional[pd.DataFrame]:
    try:
        to_ts = _now_ts()
        frm_ts = to_ts - days * 86400
        r = requests.get(DCHART_URL, params={"symbol": symbol, "resolution": resolution, "from": frm_ts, "to": to_ts}, timeout=15)
        r.raise_for_status()
        js = r.json()
        if not js or "t" not in js or not js["t"]:
            return None
        df = pd.DataFrame({
            "ts": js["t"],
            "o":  js.get("o", []),
            "h":  js.get("h", []),
            "l":  js.get("l", []),
            "c":  js.get("c", []),
            "v":  js.get("v", []),
        }).dropna()
        df["dt"] = pd.to_datetime(df["ts"], unit="s").dt.tz_localize("UTC").dt.tz_convert(TZ)
        return df.reset_index(drop=True)
    except Exception as e:
        logging.exception(f"fetch_candles({symbol},{resolution}) failed: {e}")
        return None

def fetch_5m(symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
    return fetch_candles(symbol, "5", days)

def fetch_daily(symbol: str, days: int = 200) -> Optional[pd.DataFrame]:
    return fetch_candles(symbol, "D", days)

# =========================
# Indicators & helpers
# =========================
def atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0
    h = df["h"].astype(float).values
    l = df["l"].astype(float).values
    c_prev = df["c"].astype(float).shift(1).fillna(df["c"]).values
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])

def upper_wick_ratio(o: float, h: float, l: float, c: float) -> float:
    body = abs(c - o)
    if body <= 1e-6:
        return 1.0
    return max(0.0, h - max(o, c)) / body

def avg_per_minute_volume_20d(symbol: str) -> Optional[float]:
    dfd = fetch_daily(symbol, 90)
    if dfd is None or dfd.empty:
        return None
    d20 = dfd.tail(20)
    if d20.empty: return None
    avg_daily = float(pd.to_numeric(d20["v"], errors="coerce").dropna().mean())
    if avg_daily <= 0: return None
    return avg_daily / TRADING_MIN_PER_DAY

def last_close_daily(symbol: str) -> Optional[float]:
    dfd = fetch_daily(symbol, 10)
    if dfd is None or dfd.empty:
        return None
    return float(dfd["c"].iloc[-1])

def darvas_pivot(df5: pd.DataFrame) -> Optional[float]:
    if df5 is None or len(df5) < 30:
        return None
    highs = df5["h"].astype(float)
    # local maxima window ±3
    for i in range(len(highs) - 4, 3, -1):
        win = highs.iloc[i-3:i+4]
        if highs.iloc[i] == win.max():
            return float(round(highs.iloc[i], 2))
    return float(round(highs.tail(20).max(), 2))

# Pattern heuristics (gọn, chống nhiễu)
def pattern_cup_handle(df_d: pd.DataFrame) -> bool:
    if df_d is None or len(df_d) < 80: return False
    c = df_d["c"].astype(float)
    left = c.iloc[-80:-40].max()
    bottom = c.iloc[-40:-25].min()
    right = c.iloc[-25:].max()
    depth = (left - bottom) / max(1e-6, left)
    shape_ok = (0.08 <= depth <= 0.35) and (right >= left*0.98)
    return bool(shape_ok)

def pattern_double_bottom(df_d: pd.DataFrame) -> bool:
    if df_d is None or len(df_d) < 60: return False
    c = df_d["c"].astype(float)
    a = float(c.iloc[-50:-35].min()); b = float(c.iloc[-25:-10].min())
    return abs(a - b) / max(1e-6, (a+b)/2) <= 0.03

def pattern_flag(df_d: pd.DataFrame) -> bool:
    if df_d is None or len(df_d) < 40: return False
    c = df_d["c"].astype(float)
    surge = c.iloc[-40:-30].pct_change().sum() > 0.05
    pull = c.iloc[-30:].pct_change().sum()
    return surge and (-0.04 <= pull <= 0.02)

# =========================
# Core decisions
# =========================
WICK_LIMIT = 0.60
VOL_MULT_BREAK = 1.5
CLOSE_PCT_FORCE = 0.01    # +1% trên pivot
SUPPORT_BAND = 0.02       # ~2% dưới pivot

def get_pivot(symbol: str, df5: pd.DataFrame) -> Optional[float]:
    if symbol in PIVOTS:
        return PIVOTS[symbol]
    return darvas_pivot(df5)

def analyze_symbol(symbol: str) -> Optional[str]:
    # Lọc penny (dựa daily close)
    last_d = last_close_daily(symbol)
    if last_d is None or last_d < MIN_PRICE:
        return None

    df5 = fetch_5m(symbol, 7)
    if df5 is None or len(df5) < 20:
        return None

    # Lấy 2 nến ĐÃ đóng (bỏ nến đang chạy)
    bar2 = df5.iloc[-2]   # nến mới đóng
    bar3 = df5.iloc[-3]   # nến trước đó

    c1, o1, h1, l1, v1, t1 = float(bar2["c"]), float(bar2["o"]), float(bar2["h"]), float(bar2["l"]), float(bar2["v"]), bar2["dt"]
    c0, o0, h0, l0, v0, t0 = float(bar3["c"]), float(bar3["o"]), float(bar3["h"]), float(bar3["l"]), float(bar3["v"]), bar3["dt"]

    pivot = get_pivot(symbol, df5)
    if pivot is None:
        return None

    # Volume benchmark
    apm = avg_per_minute_volume_20d(symbol)
    if apm is None or apm <= 0:
        return None

    # Điều kiện STRONG BREAKOUT
    cond_price = (c0 > pivot and c1 > pivot) or (c1 >= pivot*(1+CLOSE_PCT_FORCE))
    cond_vol   = v1 >= VOL_MULT_BREAK * apm * 5
    cond_wick  = upper_wick_ratio(o1, h1, l1, c1) <= WICK_LIMIT

    # Hỗ trợ vùng mua (close gần dưới pivot ~2%)
    support_buy = (pivot*(1-SUPPORT_BAND) <= c1 < pivot)

    # Fake breakout / Risk (đảo chiều + vol bất thường)
    wick_long   = upper_wick_ratio(o1, h1, l1, c1) > 0.8 and c1 < o1
    engulf      = (c1 < o1) and (c0 > o0) and (o1 >= c0) and (c1 <= o0)
    risk_vol    = v1 >= 1.8 * apm * 5
    risk_alert  = (wick_long or engulf) and risk_vol

    # ATR targets/SL
    _atr = atr(df5, 14)
    entry_lo  = round(max(pivot, min(c1, o1))*0.999, 2)
    entry_hi  = round(max(c1, o1), 2)
    t1 = round(c1 + 1.0*_atr, 2)
    t2 = round(c1 + 2.0*_atr, 2)
    sl = round(pivot*0.97, 2)

    # Patterns (daily)
    dfd = fetch_daily(symbol, 200)
    tags = []
    if pattern_cup_handle(dfd): tags.append("Cup&Handle")
    if pattern_double_bottom(dfd): tags.append("DoubleBottom")
    if pattern_flag(dfd): tags.append("Flag")
    if symbol in PIVOTS: tags.append("CANSLIM")
    if not tags: tags.append("Darvas")

    # Compose message theo ưu tiên: Breakout > Support > Risk
    if cond_price and cond_vol and cond_wick:
        return (f"{symbol} – BUY {entry_lo}-{entry_hi} | T1: {t1} | T2: {t2} | SL: {sl} | "
                f"⚡ Breakout xác nhận (vol {int(v1):,} ≈ {v1/(apm*5):.1f}×, {t1.strftime('%H:%M')}) | "
                f"Model: {'/'.join(tags)}")

    if support_buy:
        return (f"{symbol} – HỖ TRỢ VÙNG MUA quanh {pivot:.2f} "
                f"(close {c1:.2f}, ~{100*(pivot-c1)/pivot:.1f}% dưới) | "
                f"Entry: {entry_lo}-{entry_hi} | SL: {sl} | Model: {'/'.join(tags)}")

    if risk_alert:
        return (f"{symbol} – ⚠️ CẢNH BÁO RỦI RO (đảo chiều/vol bất thường {v1/(apm*5):.1f}×) | "
                f"Close: {c1:.2f} | Pivot: {pivot:.2f}")

    return None

# =========================
# Market breadth & sector (gọn)
# =========================
SECTOR: Dict[str, str] = {
    # Bank
    "BID":"Bank","VCB":"Bank","CTG":"Bank","TCB":"Bank","VPB":"Bank","MBB":"Bank","STB":"Bank","ACB":"Bank","TPB":"Bank","HDB":"Bank","EIB":"Bank","LPB":"Bank",
    # Chứng khoán
    "SSI":"Securities","HCM":"Securities","VND":"Securities","VIX":"Securities","SHS":"Securities","MBS":"Securities","FTS":"Securities",
    # Dầu khí / Điện
    "PVD":"Energy","PVS":"Energy","BSR":"Energy","PLX":"Energy","GAS":"Energy","POW":"Energy","PPC":"Energy",
    # BĐS/KCN
    "VHM":"Property","VIC":"Property","VRE":"Property","KDH":"Property","NLG":"Property","KBC":"Property","GEX":"Property","DXG":"Property",
    # Thép / VLXD
    "HPG":"Materials","HSG":"Materials","NKG":"Materials","KSB":"Materials",
    # Tiêu dùng / Công nghệ / Bán lẻ
    "VNM":"Consumer","MSN":"Consumer","SAB":"Consumer","FPT":"Tech","MWG":"Retail","DGW":"Retail","FRT":"Retail",
    # Hạ tầng / CN
    "REE":"Industrial","GMD":"Industrial","GVR":"Industrial","VTP":"Industrial","VGI":"Industrial","LTG":"Industrial","PAN":"Industrial",
    # Khác không map
}

def breadth_summary(tickers: List[str]) -> str:
    above20 = above50 = above200 = total = 0
    for s in tickers:
        dfd = fetch_daily(s, 250)
        if dfd is None or len(dfd) < 200:
            continue
        c = dfd["c"].astype(float)
        ma20 = c.rolling(20).mean().iloc[-1]
        ma50 = c.rolling(50).mean().iloc[-1]
        ma200 = c.rolling(200).mean().iloc[-1]
        last = float(c.iloc[-1])
        total += 1
        if last > ma20: above20 += 1
        if last > ma50: above50 += 1
        if last > ma200: above200 += 1
    if total == 0: return "Breadth: N/A"
    return f"Breadth: >MA20 {above20}/{total}, >MA50 {above50}/{total}, >MA200 {above200}/{total}"

def sector_strength(tickers: List[str]) -> str:
    # RS 3 tháng (xấp xỉ) theo % thay đổi giá
    scores: Dict[str, List[float]] = {}
    for s in tickers:
        dfd = fetch_daily(s, 90)
        if dfd is None or len(dfd) < 60:
            continue
        c = dfd["c"].astype(float)
        base = float(c.iloc[-60])
        last = float(c.iloc[-1])
        if base <= 0: continue
        rs = last/base - 1.0
        sec = SECTOR.get(s, "Other")
        scores.setdefault(sec, []).append(rs)
    if not scores: return "Sector: N/A"
    avg = {k: np.mean(v) for k,v in scores.items() if v}
    top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:4]
    return " | ".join([f"{k}:{v*100:.1f}%" for k,v in top])

# =========================
# Dedup to avoid spam
# =========================
LAST_ALERT: Dict[str, float] = {}        # key = symbol + kind
COOLDOWN_SEC = 9*60                      # 9 phút

def should_alert(key: str) -> bool:
    now = time.time()
    last = LAST_ALERT.get(key, 0)
    if now - last >= COOLDOWN_SEC:
        LAST_ALERT[key] = now
        return True
    return False

# =========================
# Scan & Send
# =========================
def telegram_send(text: str):
    try:
        requests.post(TG_URL, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

def scan_once() -> Dict[str, List[str]]:
    msgs_break, msgs_support, msgs_risk = [], [], []

    # Lọc non-penny ngay từ đầu
    filtered = []
    for s in DEFAULT_TICKERS:
        last = last_close_daily(s)
        if last is not None and last >= MIN_PRICE:
            filtered.append(s)

    for s in filtered:
        try:
            m = analyze_symbol(s)
            if not m: 
                continue
            if "⚡ Breakout xác nhận" in m and should_alert(s+"#bo"):
                msgs_break.append(m)
            elif "HỖ TRỢ VÙNG MUA" in m and should_alert(s+"#sup"):
                msgs_support.append(m)
            elif "⚠️ CẢNH BÁO RỦI RO" in m and should_alert(s+"#risk"):
                msgs_risk.append(m)
        except Exception as e:
            logging.exception(f"analyze_symbol({s}) failed: {e}")

    return {"breakout": msgs_break, "support": msgs_support, "risk": msgs_risk}

# =========================
# FastAPI
# =========================
app = FastAPI(title="VN21 Pro Scanner", version="2.0")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "VN21 Pro Scanner — /healthz /scan /market"

@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {"status":"ok","time": now_vn().isoformat()}

@app.head("/healthz")
def healthz_head():
    return Response(status_code=200)

@app.post("/scan", response_class=JSONResponse)
def scan_api():
    out = scan_once()
    return out

@app.get("/market", response_class=PlainTextResponse)
def market_brief():
    br = breadth_summary(VN21_CORE)  # breadth trên core (nhanh)
    sec = sector_strength(DEFAULT_TICKERS)
    return f"{br}\nSectors: {sec}"

# =========================
# Scheduler: quét mỗi 5'
# =========================
def scheduled_job():
    if not in_trading_session():
        return
    out = scan_once()
    lines = []
    if out["breakout"]:
        lines += ["⚡ Breakout:", *[f"• {x}" for x in out["breakout"][:8]]]
    if out["support"]:
        lines += ["🟦 Hỗ trợ mua:", *[f"• {x}" for x in out["support"][:6]]]
    if out["risk"]:
        lines += ["⚠️ Rủi ro:", *[f"• {x}" for x in out["risk"][:6]]]
    if lines:
        # thêm snapshot breadth/sector ngắn gọn
        lines.append("")
        lines.append(market_brief())
        telegram_send("📈 VN21 — Tín hiệu (5')\n" + "\n".join(lines))

scheduler = BackgroundScheduler(timezone=str(TZ))
scheduler.add_job(scheduled_job, CronTrigger.from_crontab("*/5 * * * *"))  # mỗi 5 phút
scheduler.start()

# Local run (Render dùng Procfile)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")))
