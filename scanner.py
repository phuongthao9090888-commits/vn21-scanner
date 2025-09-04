# scanner.py — VN21 Scanner (full)
# Realtime breakout (VNDirect 5m) + 20D per-minute vol + Wick filter + Darvas Box + ATR
# + Minervini Trend Template + Zanger flag + RS vs VNINDEX
# + Near-support (-2%), Risk warning, Liquidity filter (no penny), Anti-spam, Market hours.

import os, time, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

# ===== Timezone =====
TZ = timezone(timedelta(hours=7))
def now_vn(): return datetime.now(TZ)

# ===== Telegram (đã yêu cầu nhét cứng) =====
TELEGRAM_TOKEN  = "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o"
TELEGRAM_CHAT_ID= "5614513021"
TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
def send_telegram(text: str):
    try:
        requests.post(TG_API, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=8)
    except Exception: pass

# ===== Universe & Pivots (VN21 core) =====
PIVOTS = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0,
}
EXTRA = [  # mở rộng (lọc penny/thanh khoản ở runtime)
  "BID","STB","SHB","ACB","TPB","EIB","LPB","HDB",
  "SSI","HCM","VND","SHS","MBS",
  "PVD","PVS","PLX",
  "GEX","KBC","NLG","DXG",
  "HSG","NKG",
  "FPT","MSN","SAB","DGW","FRT",
  "VIC","GVR"
]
TICKERS = sorted(set(list(PIVOTS.keys()) + EXTRA))

# ===== Rules & thresholds =====
UPPER_WICK_LIMIT  = 0.60
VOL_FACTOR_EARLY  = 1.2
VOL_FACTOR_CONF   = 1.5
BREAK_PCT_EARLY   = 0.005
BREAK_PCT_CONF    = 0.01
NEAR_BUY_BAND     = 0.02
MIN_PRICE_NON_PENNY = 10.0
MIN_AVG_VALUE_VND   = 30_000_000_000  # 30 tỷ/phiên
ATR_WINDOW = 14

# ===== VNDirect endpoints & headers =====
VND_API = "https://dchart-api.vndirect.com.vn/dchart/history"
HDRS = {"User-Agent":"Mozilla/5.0", "Referer":"https://dchart.vndirect.com.vn/"}

# ===== Market hours VN =====
def market_open_now() -> bool:
    t = now_vn().time()
    return (t >= datetime.strptime("09:00","%H:%M").time() and t <= datetime.strptime("11:30","%H:%M").time()) or \
           (t >= datetime.strptime("13:00","%H:%M").time() and t <= datetime.strptime("15:00","%H:%M").time())

# ========== Fetchers ==========
def fetch_5m_df(symbol: str, lookback_days: int = 25) -> pd.DataFrame:
    now = int(time.time()); since = now - lookback_days*24*3600
    try:
        r = requests.get(VND_API, params={"symbol":symbol,"resolution":"5","from":since,"to":now}, headers=HDRS, timeout=10)
        j = r.json() or {}
        if j.get("s")!="ok" or not j.get("t"): return pd.DataFrame(columns=list("ohlcv"))
        df = pd.DataFrame({
            "ts": np.array(j["t"], dtype="int64"),
            "open": j.get("o",[]), "high": j.get("h",[]),
            "low":  j.get("l",[]), "close":j.get("c",[]),
            "volume":j.get("v",[]),
        })
        df["dt"] = pd.to_datetime(df["ts"], unit="s") + pd.Timedelta(hours=7)
        df = df.drop(columns=["ts"]).set_index("dt").sort_index()
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()
    except Exception:
        return pd.DataFrame(columns=list("ohlcv"))

def fetch_daily_df(symbol: str, days: int = 40) -> pd.DataFrame:
    try:
        r = requests.get(VND_API, params={"symbol":symbol,"resolution":"D","from":0,"to":9999999999}, headers=HDRS, timeout=10)
        j = r.json() or {}
        if j.get("s")!="ok" or not j.get("t"): return pd.DataFrame()
        df = pd.DataFrame({
            "ts": np.array(j["t"], dtype="int64"),
            "open": j.get("o",[]), "high": j.get("h",[]),
            "low":  j.get("l",[]), "close":j.get("c",[]),
            "volume":j.get("v",[]),
        })
        df["dt"] = pd.to_datetime(df["ts"], unit="s") + pd.Timedelta(hours=7)
        df = df.drop(columns=["ts"]).set_index("dt").sort_index()
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.tail(days)
    except Exception:
        return pd.DataFrame()

def avg_per_min_volume_20d(symbol: str, minutes_per_day: int = 225) -> float:
    d = fetch_daily_df(symbol, days=30)
    if d.empty: return 0.0
    v20 = d["volume"].tail(20).mean()
    return float(v20) / max(1, minutes_per_day)

def avg_daily_value_vnd(symbol: str, days: int = 20) -> float:
    d = fetch_daily_df(symbol, days=days+5)
    if d.empty: return 0.0
    v = (d["close"].tail(days) * d["volume"].tail(days)) * 1000.0  # ~ VND
    return float(v.mean())

# ========== Indicators / Expert helpers ==========
def sma(s, n): return s.rolling(n, min_periods=max(2, n//3)).mean()
def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=max(2, n//3)).mean()

def _upper_wick_ratio(row)->float:
    body = abs(row["close"]-row["open"])
    if body<=0: return float("inf")
    return max(0.0, row["high"]-max(row["close"],row["open"])) / body

def _atr_val(df: pd.DataFrame, n: int = ATR_WINDOW) -> float:
    if len(df)<n+1: return 0.0
    c,h,l = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n, min_periods=n//2).mean().iloc[-1] or 0.0)

def _darvas_box_top(df: pd.DataFrame, lookback: int = 30) -> float|None:
    if len(df)<lookback: return None
    win = df.tail(lookback); highs = win["high"].values
    idx=None
    for i in range(len(highs)-3,2,-1):
        if highs[i]>=highs[i-1]>=highs[i-2] and highs[i]>=highs[i+1]>=highs[i+2]:
            idx=i; break
    return float(win["high"].iloc[idx]) if idx is not None else None

def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"]*df["volume"]).cumsum(); vv = df["volume"].cumsum().replace(0,np.nan)
    return pv/vv

def high_52w(df_5m: pd.DataFrame) -> float:
    # xấp xỉ theo 5m (ổn cho nhãn)
    if df_5m.empty: return 0.0
    n = min(len(df_5m), 52*48//5)
    return float(df_5m["high"].tail(n).max())

def is_zanger_flag(df: pd.DataFrame, lookback=20, tight=0.06) -> bool:
    win = df.tail(lookback)
    if len(win)<lookback: return False
    rng = float(win["high"].max()-win["low"].min())
    base= float(win["close"].iloc[-2]); 
    if base<=0: return False
    cond_tight = (rng/base) <= tight
    cond_break = win["high"].max() <= df["close"].iloc[-1]
    return cond_tight and cond_break

def minervini_trend_template(df: pd.DataFrame) -> bool:
    c=df["close"]
    if len(c)<220: return False
    ma50,ma150,ma200 = sma(c,50),sma(c,150),sma(c,200)
    if any(pd.isna([ma50.iloc[-1],ma150.iloc[-1],ma200.iloc[-1]])): return False
    cond_stack = c.iloc[-1]>ma50.iloc[-1]>ma150.iloc[-1]>ma200.iloc[-1]
    hi52 = high_52w(df); cond_52w = c.iloc[-1] >= 0.75*hi52
    cond_rising = ma50.diff(5).iloc[-1] > 0
    return bool(cond_stack and cond_52w and cond_rising)

def near_support_zone(price: float, pivot: float, band=NEAR_BUY_BAND)->bool:
    return pivot*(1-band) <= price < pivot

def bearish_reversal_risk(df: pd.DataFrame)->bool:
    if len(df)<2: return False
    c1,o1,h1,l1,v1 = df["close"].iloc[-1], df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["volume"].iloc[-1]
    c2,o2         = df["close"].iloc[-2], df["open"].iloc[-2]
    body = abs(c1-o1)+1e-9; upper = h1-max(c1,o1)
    long_wick = (upper/body)>0.8
    engulf    = (o1>c1) and (o2<c2) and (o1>=c2) and (c1<=o2)
    vol_spike = v1 > 2*df["volume"].rolling(20, min_periods=5).mean().iloc[-1]
    return engulf or (long_wick and vol_spike)

# ========== Core scoring ==========
def score(df: pd.DataFrame, symbol: str, pivot: float)->dict:
    out = {"ok_confirmed":False, "ok_early":False, "why":[], "model_tags":[]}
    if df is None or df.empty or len(df)<40: out["why"].append("no-data"); return out
    last, prev = df.iloc[-1], df.iloc[-2]
    price = float(last["close"])

    # Liquidity / penny filter
    if price < MIN_PRICE_NON_PENNY: out["why"].append("penny"); return out
    if avg_daily_value_vnd(symbol, 20) < MIN_AVG_VALUE_VND: out["why"].append("thin-liquidity"); return out

    # Pivot level (fallback: darvas)
    level = float(pivot) if (pivot and pivot>0) else (_darvas_box_top(df) or 0.0)
    if level<=0: out["why"].append("no-pivot"); return out

    # Volume need (5m) from 20D per-minute
    per_min = avg_per_min_volume_20d(symbol)  # shares/min
    need_5m_early = VOL_FACTOR_EARLY * per_min * 5.0
    need_5m_conf  = VOL_FACTOR_CONF  * per_min * 5.0

    # Price conditions
    price_early = (last["close"] > level) or (last["close"] >= level*(1+BREAK_PCT_EARLY))
    price_conf  = ((prev["close"]>level) and (last["close"]>level)) or (last["close"] >= level*(1+BREAK_PCT_CONF))

    # Wick
    wick_ok = (_upper_wick_ratio(last) <= UPPER_WICK_LIMIT)

    # Early / Confirmed
    vol_ok_early = last["volume"] >= need_5m_early if need_5m_early>0 else False
    vol_ok_conf  = last["volume"] >= need_5m_conf  if need_5m_conf>0  else False

    out["ok_early"]     = bool(price_early and vol_ok_early)
    out["ok_confirmed"] = bool(price_conf  and vol_ok_conf and wick_ok)

    # Model tags (không bắt buộc)
    if minervini_trend_template(df): out["model_tags"].append("Minervini")
    if is_zanger_flag(df):           out["model_tags"].append("ZangerFlag")
    if (price > (_darvas_box_top(df) or level)): out["model_tags"].append("Darvas")

    # ATR-based targets
    atrv = _atr_val(df, ATR_WINDOW)
    if atrv<=0: atrv = max(0.003*price, 0.05)
    entry_lo = max(level, price - 0.25*atrv)
    entry_hi = price + 0.10*atrv
    t1 = price + 1.0*atrv
    t2 = price + 2.0*atrv
    sl = min(level*0.985, price - 1.2*atrv)

    out.update({
        "pivot_used": round(level,2),
        "entry": (round(entry_lo,2), round(entry_hi,2)),
        "t1": round(t1,2), "t2": round(t2,2), "sl": round(sl,2),
        "vol_note": f"{int(last['volume']):,} (need≈{int(need_5m_conf):,}/5m)",
        "price": round(price,2),
        "near_support": near_support_zone(price, level),
        "risk_warn": bearish_reversal_risk(df),
    })
    return out

# ========== Analyze & format ==========
def analyze_symbol(symbol: str, pivot: float)->list[str]:
    df = fetch_5m_df(symbol)
    if df.empty or len(df)<40: return []
    s = score(df, symbol, pivot)
    msgs = []

    # Near support / Risk (nhắc nhẹ)
    notes = []
    if s.get("near_support"): notes.append("🟦 Vùng mua hỗ trợ")
    if s.get("risk_warn"):    notes.append("⚠️ Rủi ro đảo chiều")
    notes_txt = " | ".join(notes) if notes else ""

    model_txt = " / ".join(s.get("model_tags", [])) or "Model"

    # Early
    if s.get("ok_early") and not s.get("ok_confirmed"):
        lo,hi = s["entry"]; when = now_vn().strftime("%H:%M")
        msgs.append(f"{symbol} – BUY {lo}–{hi} | T1: {s['t1']} | T2: {s['t2']} | SL: {s['sl']} | 🔔 Early breakout ({s['vol_note']}, {when}) | {model_txt}" + (f" | {notes_txt}" if notes_txt else ""))

    # Confirmed
    if s.get("ok_confirmed"):
        lo,hi = s["entry"]; when = now_vn().strftime("%H:%M")
        msgs.append(f"{symbol} – BUY {lo}–{hi} | T1: {s['t1']} | T2: {s['t2']} | SL: {s['sl']} | ⚡ Breakout xác nhận ({s['vol_note']}, {when}) | {model_txt}" + (f" | {notes_txt}" if notes_txt else ""))

    # Nếu không có tín hiệu, nhưng có nhắc nhở:
    if not msgs and notes_txt:
        msgs.append(f"{symbol} – {notes_txt}")

    return msgs

# ========== Scanner loop (anti-spam + rate-limit friendly) ==========
def run_scanner():
    last_push = {}  # key=(symbol, kind), val=timestamp
    COOL_EARLY = 600   # 10'
    COOL_CONF  = 900   # 15'
    COOLDOWN   = { "EARLY": COOL_EARLY, "CONF": COOL_CONF, "NOTE": 900 }

    def can_push(sym, kind):
        now_ts = time.time()
        key = (sym, kind)
        if now_ts - last_push.get(key, 0) >= COOLDOWN[kind]:
            last_push[key] = now_ts
            return True
        return False

    send_telegram(f"🚀 VN21 Scanner started {now_vn().strftime('%Y-%m-%d %H:%M:%S')}")

    while True:
        try:
            if market_open_now():
                for i, sym in enumerate(TICKERS):
                    try:
                        piv = PIVOTS.get(sym, 0.0)
                        msgs = analyze_symbol(sym, piv)
                        for m in msgs:
                            kind = "CONF" if "xác nhận" in m else ("EARLY" if "Early" in m else "NOTE")
                            if can_push(sym, kind):
                                send_telegram(m)
                        time.sleep(0.20)  # tránh rate-limit
                    except Exception:
                        time.sleep(0.10)
                time.sleep(20)  # vòng quét tiếp theo
            else:
                time.sleep(60)
        except Exception:
            time.sleep(10)
