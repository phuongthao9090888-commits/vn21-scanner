# app.py — VN21 Pro Scanner (Merged Part 1→7)
# Full module: MTF breakout, patterns, ATR targets/SL, sizing, fake-breakout alerts,
# breadth & sector, non-penny filter, early/confirmed alerts, health endpoints.

import os, math, time, logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import pytz
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# =========================
# Telegram (hard-coded; allow ENV override)
# =========================
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5614513021")
TG_URL  = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

# =========================
# Timezone & Session
# =========================
TZ = pytz.timezone("Asia/Ho_Chi_Minh")

def now_vn() -> datetime:
    return datetime.now(TZ)

def in_trading_session(dt: Optional[datetime] = None) -> bool:
    dt = dt or now_vn()
    if dt.weekday() >= 5:
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

# Pivot tay cho VN21; mã khác fallback Darvas
PIVOTS: Dict[str, float] = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# =========================
# Global params
# =========================
MIN_PRICE           = 10.0   # non-penny filter (nghìn VND)
TRADING_MIN_PER_DAY = 270
WICK_LIMIT          = 0.60   # upper wick ≤ 60% thân nến
VOL_MULT_BREAK      = 1.5    # 5m vol per-minute ≥ 1.5× 20D avg per-minute
VOL_MULT_EARLY      = 1.2    # early alert ngưỡng thấp hơn
CLOSE_PCT_FORCE     = 0.01   # +1% trên pivot
SUPPORT_BAND        = 0.02   # ~2% dưới pivot (hỗ trợ vùng mua)
COOLDOWN_SEC        = 9*60   # chống spam theo mã & loại tín hiệu

# Position sizing default (Part 2)
DEFAULT_NAV         = float(os.getenv("VN21_NAV", "300000000.0"))  # 300tr default
RISK_PCT_PER_TRADE  = float(os.getenv("VN21_RISK_PCT", "0.01"))    # rủi ro 1% NAV / lệnh
KELLY_P_WIN         = float(os.getenv("VN21_KELLY_P", "0.45"))      # giả định tạm
KELLY_RR            = float(os.getenv("VN21_KELLY_RR","2.0"))

# =========================
# VNDirect fetchers
# =========================
DCHART_URL = "https://dchart-api.vndirect.com.vn/dchart/history"

def _now_ts() -> int:
    return int(time.time())

def fetch_candles(symbol: str, resolution: str, days: int) -> Optional[pd.DataFrame]:
    try:
        to_ts  = _now_ts()
        from_ts = to_ts - days*86400
        r = requests.get(DCHART_URL, params={"symbol": symbol, "resolution": resolution, "from": from_ts, "to": to_ts}, timeout=15)
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

def fetch_15m(symbol: str, days: int = 20) -> Optional[pd.DataFrame]:
    return fetch_candles(symbol, "15", days)

def fetch_daily(symbol: str, days: int = 220) -> Optional[pd.DataFrame]:
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
    tr = np.maximum(h-l, np.maximum(np.abs(h-c_prev), np.abs(l-c_prev)))
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

# Darvas-like pivot (fallback)
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

def get_pivot(symbol: str, df5: pd.DataFrame) -> Optional[float]:
    if symbol in PIVOTS:
        return PIVOTS[symbol]
    return darvas_pivot(df5)

# Patterns (heuristics)
def pattern_cup_handle(df_d: pd.DataFrame) -> bool:
    if df_d is None or len(df_d) < 80: return False
    c = df_d["c"].astype(float)
    left = c.iloc[-80:-40].max()
    bottom = c.iloc[-40:-25].min()
    right = c.iloc[-25:].max()
    depth = (left - bottom) / max(1e-6, left)
    return (0.08 <= depth <= 0.35) and (right >= left*0.98)

def pattern_double_bottom(df_d: pd.DataFrame) -> bool:
    if df_d is None or len(df_d) < 60: return False
    c = df_d["c"].astype(float)
    a = float(c.iloc[-50:-35].min()); b = float(c.iloc[-25:-10].min())
    return abs(a - b) / max(1e-6, (a+b)/2) <= 0.03

def pattern_flag(df_d: pd.DataFrame) -> bool:
    if df_d is None or len(df_d) < 40: return False
    c = df_d["c"].astype(float)
    surge = c.iloc[-40:-30].pct_change().sum() > 0.05
    pull  = c.iloc[-30:].pct_change().sum()
    return surge and (-0.04 <= pull <= 0.02)

# MTF confirm (Part 1)
def multi_timeframe_confirm(symbol: str, pivot: float) -> bool:
    df15 = fetch_15m(symbol, 10)
    dfd  = fetch_daily(symbol, 35)
    if df15 is None or len(df15) < 3 or dfd is None or len(dfd) < 5:
        return False
    c15 = float(df15["c"].iloc[-1])
    cD  = float(dfd["c"].iloc[-1])
    return (c15 > pivot) and (cD > pivot*0.995)

# Position sizing (Part 2)
def kelly_fraction(p_win: float, rr: float) -> float:
    # Kelly = p - (1-p)/rr
    return max(0.0, min(1.0, p_win - (1.0 - p_win)/max(1e-6, rr)))

def fixed_fraction_size(nav: float, risk_pct: float, entry: float, sl: float) -> int:
    if entry <= 0 or sl <= 0 or entry <= sl:
        return 0
    risk_amount = nav * risk_pct
    per_share   = entry - sl
    if per_share <= 0:
        return 0
    shares = int(risk_amount // per_share)
    return max(shares, 0)

# =========================
# Core decisions
# =========================
def analyze_symbol(symbol: str) -> List[Tuple[str,str]]:
    """
    Trả về list các (kind, message):
      kind ∈ {"CONFIRM","EARLY","SUPPORT","RISK"}
    """
    out: List[Tuple[str,str]] = []

    # Non-penny filter
    last_d = last_close_daily(symbol)
    if last_d is None or last_d < MIN_PRICE:
        return out

    df5 = fetch_5m(symbol, 7)
    if df5 is None or len(df5) < 20:
        return out

    # Lấy 2 nến đã đóng (bỏ nến đang chạy)
    bar2 = df5.iloc[-2]
    bar3 = df5.iloc[-3]
    c1, o1, h1, l1, v1, t1 = float(bar2["c"]), float(bar2["o"]), float(bar2["h"]), float(bar2["l"]), float(bar2["v"]), bar2["dt"]
    c0, o0, h0, l0, v0, t0 = float(bar3["c"]), float(bar3["o"]), float(bar3["h"]), float(bar3["l"]), float(bar3["v"]), bar3["dt"]

    pivot = get_pivot(symbol, df5)
    if pivot is None:
        return out

    apm = avg_per_minute_volume_20d(symbol)
    if apm is None or apm <= 0:
        return out

    # Early / Confirmed
    cond_price_confirm = (c0 > pivot and c1 > pivot) or (c1 >= pivot*(1+CLOSE_PCT_FORCE))
    cond_price_early   = (c1 > pivot) or (c1 >= pivot*(1+CLOSE_PCT_FORCE/2))
    cond_vol_confirm   = v1 >= VOL_MULT_BREAK * apm * 5
    cond_vol_early     = v1 >= VOL_MULT_EARLY * apm * 5
    cond_wick          = upper_wick_ratio(o1, h1, l1, c1) <= WICK_LIMIT
    cond_mtf           = multi_timeframe_confirm(symbol, pivot)

    # Support (về vùng mua)
    support_buy = (pivot*(1-SUPPORT_BAND) <= c1 < pivot)

    # Fake breakout / Risk
    wick_long   = upper_wick_ratio(o1, h1, l1, c1) > 0.8 and c1 < o1
    engulf      = (c1 < o1) and (c0 > o0) and (o1 >= c0) and (c1 <= o0)
    risk_vol    = v1 >= 1.8 * apm * 5
    risk_alert  = (wick_long or engulf) and risk_vol

    # ATR targets/SL
    _atr = atr(df5, 14)
    entry_lo  = round(max(pivot, min(c1, o1))*0.999, 2)
    entry_hi  = round(max(c1, o1), 2)
    t1_price  = round(c1 + 1.0*_atr, 2)
    t2_price  = round(c1 + 2.0*_atr, 2)
    sl_price  = round(pivot*0.97, 2)

    # Patterns (daily)
    dfd = fetch_daily(symbol, 220)
    tags = []
    if pattern_cup_handle(dfd):   tags.append("Cup&Handle")
    if pattern_double_bottom(dfd):tags.append("DoubleBottom")
    if pattern_flag(dfd):         tags.append("Flag")
    if symbol in PIVOTS:          tags.append("CANSLIM")
    if not tags:                  tags.append("Darvas")

    # Position sizing
    size_fixed = fixed_fraction_size(DEFAULT_NAV, RISK_PCT_PER_TRADE, entry_hi, sl_price)
    kelly_f    = kelly_fraction(KELLY_P_WIN, KELLY_RR)
    size_kelly = fixed_fraction_size(DEFAULT_NAV, min(kelly_f, RISK_PCT_PER_TRADE*2.0), entry_hi, sl_price)

    size_note = f"Size≈{size_fixed} (fixed {int(RISK_PCT_PER_TRADE*100)}%), Kelly≈{size_kelly}"

    # Compose messages (priority: Confirmed > Early > Support > Risk)
    if cond_price_confirm and cond_vol_confirm and cond_wick and cond_mtf:
        msg = (f"{symbol} – BUY {entry_lo}-{entry_hi} | T1: {t1_price} | T2: {t2_price} | SL: {sl_price} | "
               f"⚡ Breakout xác nhận MTF (vol {int(v1):,} ≈ {v1/(apm*5):.1f}×, {t1.strftime('%H:%M')}) | "
               f"Model: {'/'.join(tags)} | {size_note}")
        out.append(("CONFIRM", msg))
    elif cond_price_early and cond_vol_early and cond_wick:
        msg = (f"{symbol} – BUY {entry_lo}-{entry_hi} | T1: {t1_price} | T2: {t2_price} | SL: {sl_price} | "
               f"🔔 Early breakout (vol {int(v1):,} ≈ {v1/(apm*5):.1f}×, {t1.strftime('%H:%M')}) | "
               f"Model: {'/'.join(tags)} | {size_note}")
        out.append(("EARLY", msg))

    if support_buy:
        msg = (f"{symbol} – HỖ TRỢ VÙNG MUA quanh {pivot:.2f} "
               f"(close {c1:.2f}, ~{100*(pivot-c1)/pivot:.1f}% dưới) | "
               f"Entry: {entry_lo}-{entry_hi} | SL: {sl_price} | Model: {'/'.join(tags)} | {size_note}")
        out.append(("SUPPORT", msg))

    if risk_alert:
        msg = (f"{symbol} – ⚠️ CẢNH BÁO RỦI RO (đảo chiều/vol bất thường {v1/(apm*5):.1f}×) | "
               f"Close: {c1:.2f} | Pivot: {pivot:.2f}")
        out.append(("RISK", msg))

    return out

# =========================
# Breadth & Sector (Part 5)
# =========================
SECTOR: Dict[str, str] = {
    # Bank
    "BID":"Bank","VCB":"Bank","CTG":"Bank","TCB":"Bank","VPB":"Bank","MBB":"Bank","STB":"Bank",
    "ACB":"Bank","TPB":"Bank","HDB":"Bank","EIB":"Bank","LPB":"Bank",
    # Securities
    "SSI":"Securities","HCM":"Securities","VND":"Securities","VIX":"Securities","SHS":"Securities","MBS":"Securities","FTS":"Securities",
    # Energy/Power
    "PVD":"Energy","PVS":"Energy","BSR":"Energy","PLX":"Energy","GAS":"Energy","POW":"Energy","PPC":"Energy",
    # Property/Industrial
    "VHM":"Property","VIC":"Property","VRE":"Property","KDH":"Property","NLG":"Property","KBC":"Property","GEX":"Industrial","DXG":"Property",
    # Materials
    "HPG":"Materials","HSG":"Materials","NKG":"Materials","KSB":"Materials",
    # Consumer/Tech/Retail
    "VNM":"Consumer","MSN":"Consumer","SAB":"Consumer","FPT":"Tech","MWG":"Retail","DGW":"Retail","FRT":"Retail",
    # Infra/Agri
    "REE":"Industrial","GMD":"Industrial","GVR":"Industrial","VTP":"Industrial","VGI":"Industrial","LTG":"Industrial","PAN":"Industrial"
}

def breadth_summary(tickers: List[str]) -> str:
    above20 = above50 = above200 = total = 0
    for s in tickers:
        dfd = fetch_daily(s, 250)
        if dfd is None or len(dfd) < 200:
            continue
        c = dfd["c"].astype(float)
        ma20  = c.rolling(20).mean().iloc[-1]
        ma50  = c.rolling(50).mean().iloc[-1]
        ma200 = c.rolling(200).mean().iloc[-1]
        last  = float(c.iloc[-1])
        total += 1
        if last > ma20:  above20 += 1
        if last > ma50:  above50 += 1
        if last > ma200: above200 += 1
    if total == 0: return "Breadth: N/A"
    return f"Breadth: >MA20 {above20}/{total}, >MA50 {above50}/{total}, >MA200 {above200}/{total}"

def sector_strength(tickers: List[str]) -> str:
    scores: Dict[str, List[float]] = {}
    for s in tickers:
        dfd = fetch_daily(s, 90)
        if dfd is None or len(dfd) < 60:
            continue
        c = dfd["c"].astype(float)
        base = float(c.iloc[-60]); last = float(c.iloc[-1])
        if base <= 0: continue
        rs = last/base - 1.0
        sec = SECTOR.get(s, "Other")
        scores.setdefault(sec, []).append(rs)
    if not scores: return "Sector: N/A"
    avg = {k: np.mean(v) for k,v in scores.items() if v}
    top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:4]
    return " | ".join([f"{k}:{v*100:.1f}%" for k,v in top])

# =========================
# Dedup & Cooldown (Part 6)
# =========================
LAST_ALERT: Dict[str, float] = {}  # key: sym#kind

def should_alert(key: str) -> bool:
    now = time.time()
    last = LAST_ALERT.get(key, 0)
    if now - last >= COOLDOWN_SEC:
        LAST_ALERT[key] = now
        return True
    return False

def telegram_send(text: str):
    try:
        requests.post(TG_URL, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        logging.exception(f"Telegram error: {e}")

# =========================
# Scan runners
# =========================
def scan_once() -> Dict[str, List[str]]:
    msgs = {"CONFIRM":[], "EARLY":[], "SUPPORT":[], "RISK":[]}

    # Non-penny prefilter
    filtered = []
    for s in DEFAULT_TICKERS:
        last = last_close_daily(s)
        if last is not None and last >= MIN_PRICE:
            filtered.append(s)

    for s in filtered:
        try:
            signals = analyze_symbol(s)
            for kind, msg in signals:
                key = f"{s}#{kind}"
                if should_alert(key):
                    msgs[kind].append(msg)
        except Exception as e:
            logging.exception(f"analyze_symbol({s}) failed: {e}")
    return msgs

# =========================
# FastAPI (Part 7)
# =========================
app = FastAPI(title="VN21 Pro Scanner", version="3.0")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "VN21 Pro Scanner — /healthz /scan /market /config"

@app.get("/healthz", response_class=JSONResponse)
def healthz():
    return {"status":"ok","time": now_vn().isoformat()}

@app.head("/healthz")
def healthz_head():
    return Response(status_code=200)

@app.post("/scan", response_class=JSONResponse)
def scan_api():
    return scan_once()

@app.get("/market", response_class=PlainTextResponse)
def market_brief():
    br  = breadth_summary(VN21_CORE)
    sec = sector_strength(DEFAULT_TICKERS)
    return f"{br}\nSectors: {sec}"

@app.get("/config", response_class=JSONResponse)
def config_api():
    return {
        "min_price_filter": MIN_PRICE,
        "wick_limit": WICK_LIMIT,
        "vol_mult_confirm": VOL_MULT_BREAK,
        "vol_mult_early": VOL_MULT_EARLY,
        "close_pct_force": CLOSE_PCT_FORCE,
        "support_band": SUPPORT_BAND,
        "cooldown_sec": COOLDOWN_SEC,
        "nav": DEFAULT_NAV,
        "risk_pct": RISK_PCT_PER_TRADE,
        "kelly_p": KELLY_P_WIN,
        "kelly_rr": KELLY_RR,
        "vn21_core": VN21_CORE,
        "universe": DEFAULT_TICKERS
    }

# =========================
# Scheduler: quét mỗi 5 phút trong giờ VN
# =========================
def scheduled_job():
    if not in_trading_session():
        return
    out = scan_once()
    chunks = []
    if out["CONFIRM"]:
        chunks += ["⚡ Breakout (Confirmed):", *[f"• {x}" for x in out["CONFIRM"][:8]]]
    if out["EARLY"]:
        chunks += ["🔔 Early breakout:", *[f"• {x}" for x in out["EARLY"][:8]]]
    if out["SUPPORT"]:
        chunks += ["🟦 Hỗ trợ mua:", *[f"• {x}" for x in out["SUPPORT"][:6]]]
    if out["RISK"]:
        chunks += ["⚠️ Rủi ro:", *[f"• {x}" for x in out["RISK"][:6]]]
    if chunks:
        chunks.append("")
        chunks.append(market_brief())
        telegram_send("📈 VN21 — Tín hiệu (5')\n" + "\n".join(chunks))

scheduler = BackgroundScheduler(timezone=str(TZ))
scheduler.add_job(scheduled_job, CronTrigger.from_crontab("*/5 * * * *"))
scheduler.start()

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT","8000")))
