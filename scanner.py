# scanner.py — VN21 Scanner (Parts 1–40 merged)
from __future__ import annotations

import os, time, math, json, threading
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd

# =========================
# 0) Config & Constants
# =========================
TZ = timezone(timedelta(hours=7))  # GMT+7
APP_NAME = "vn21-scanner"

# Telegram (có ENV override; mặc định dùng theo bạn yêu cầu)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN",
    "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5614513021")

# VN21 core
VN21_CORE = [
    "VPB","MBB","TCB","CTG","DCM","KDH",
    "HPG","VHM","VIX","DDV","BSR","POW",
    "REE","GMD","VNM","MWG"
]

# Universe mở rộng (không penny – sẽ lọc theo giá runtime)
UNIVERSE_EXTRA = [
    # Bank
    "BID","STB","SHB","ACB","TPB","EIB","LPB","HDB",
    # CK
    "SSI","HCM","VND","VIX","SHS","MBS",
    # Dầu khí
    "PVD","PVS","BSR","PLX","POW","OIL",
    # BĐS/KCN
    "KDH","VHM","GEX","KBC","NLG","DXG","VIC",
    # Thép/VLXD
    "HPG","HSG","NKG","KSB",
    # Công nghệ/tiêu dùng
    "FPT","MWG","VNM","MSN","SAB","DGW","FRT",
    # Khác vốn hóa lớn
    "GVR","REE","GMD","VTP","LTG","PAN"
]

DEFAULT_TICKERS = sorted(list(set(VN21_CORE + UNIVERSE_EXTRA)))

# Pivot tay ưu tiên (VN21). Mã khác dùng Darvas (tự tính).
PIVOTS: Dict[str, float] = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0,
}

# Tham số mô hình
UPPER_WICK_LIMIT = 0.60       # không chấp nhận râu trên >60% thân
VOL_BOOST = 1.5               # vol 5' ≥ 1.5× average
BREAK_PCT = 0.01              # > +1% trên pivot coi như qua
NEAR_BUY_BAND = 0.02          # vùng mua ~ pivot -2%
ATR_WINDOW = 14               # ATR nội bộ (5m) để SL/Targets
MIN_PRICE_NON_PENNY = 10.0    # lọc penny theo runtime (>= 10k)

SELF_PING_URL = os.getenv("SELF_PING_URL",
    "https://vn21-scanner.onrender.com/healthz")
DISABLE_SELF_PING = os.getenv("SELF_PING_DISABLE", "0") == "1"

# =========================
# 1) Utils
# =========================
def now_vn() -> datetime:
    return datetime.now(TZ)

def send_telegram(text: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID,
                                 "text": text, "parse_mode": "HTML"}, timeout=8)
    except Exception:
        pass

def pct(a: float, b: float) -> float:
    if b == 0: return 0.0
    return (a - b) / b

# =========================
# 2) Data – VNDirect try/fallbacks
# =========================
def _vn_time_range_minutes(mins: int = 600) -> Tuple[int, int]:
    to_ts = int(datetime.now(TZ).timestamp())
    fr_ts = to_ts - mins * 60
    return fr_ts, to_ts

def fetch_intraday_5m(symbol: str) -> pd.DataFrame:
    """
    Cố gắng lấy nến 5m. Thứ tự ưu tiên:
    - VNDIRECT finfo intraday (5m)
    - Fallback: tự resample từ 1m nếu có
    - Nếu fail: trả DataFrame rỗng
    """
    fr, to = _vn_time_range_minutes(60*8)  # ~8h
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        # 1) vndirect 5m
        url = ("https://finfo-api.vndirect.com.vn/v4/stock_intraday?"
               f"symbol={symbol}&resolution=5&from={fr}&to={to}")
        r = requests.get(url, headers=headers, timeout=10)
        if r.ok:
            data = r.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                # vndirect key thường: t (epoch), o,h,l,c,v
                for k in ['o','h','l','c','v','t']:
                    if k not in df.columns:
                        raise ValueError("bad schema")
                df = df.rename(columns={'o':'open','h':'high','l':'low',
                                        'c':'close','v':'volume','t':'time'})
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TZ)
                return df[["time","open","high","low","close","volume"]]
    except Exception:
        pass

    # Fallback: rỗng
    return pd.DataFrame(columns=["time","open","high","low","close","volume"])

def fetch_last_quote(symbol: str) -> Optional[Dict]:
    """Lấy snapshot cuối cùng (price, volume, time) – dùng finfo last price."""
    headers = {"User-Agent":"Mozilla/5.0"}
    try:
        url = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?symbol={symbol}&sort=date:desc&size=1"
        r = requests.get(url, headers=headers, timeout=8)
        if r.ok and r.json().get("data"):
            d = r.json()["data"][0]
            # schema thường có closePrice/averagePrice/accumulatedVolume…
            price = float(d.get("closePrice") or d.get("priceClose") or d.get("matchPrice") or 0)
            vol = float(d.get("nmVol") or d.get("accumulatedVolume") or 0)
            t = d.get("date") or d.get("tradingDate")
            if t:
                try:
                    ts = pd.to_datetime(t).tz_localize("UTC").tz_convert(TZ)
                except Exception:
                    ts = now_vn()
            else:
                ts = now_vn()
            return {"price": price, "volume": vol, "time": ts}
    except Exception:
        pass
    return None

# =========================
# 3) Indicators & models
# =========================
def compute_atr(df: pd.DataFrame, n: int = ATR_WINDOW) -> float:
    if len(df) < n+1: return 0.0
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    prev_close = np.roll(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(n).mean().iloc[-1]
    return float(atr or 0.0)

def darvas_pivot(df: pd.DataFrame, lookback: int = 60) -> Optional[float]:
    """Đơn giản: lấy swing high gần nhất làm pivot."""
    if len(df) < lookback: return None
    win = df.tail(lookback)
    # chọn đỉnh mà 3 nến hai bên thấp hơn
    highs = win["high"].to_list()
    idx = None
    for i in range(3, len(highs)-3):
        if highs[i] == max(highs[i-3:i+4]): idx = i
    if idx is None: return None
    return float(win["high"].iloc[idx])

def upper_wick_too_long(candle) -> bool:
    o,h,l,c,v = candle
    body = abs(c - o)
    if body == 0: return True
    upper = h - max(o,c)
    return upper / body > UPPER_WICK_LIMIT

def volume_boost_ok(df: pd.DataFrame) -> Tuple[bool,float,float]:
    """So sánh vol nến mới nhất với trung bình gần (proxy cho 20d per-min)."""
    if len(df) < 40: return (False, 0.0, 0.0)
    last_vol = df["volume"].iloc[-1]
    avg_vol = df["volume"].iloc[-40:-1].mean()  # ~ 3.25h
    ratio = (last_vol / avg_vol) if avg_vol else 0.0
    return (ratio >= VOL_BOOST, float(last_vol), float(avg_vol))

# =========================
# 4) Universe handling
# =========================
def filter_non_penny(tickers: List[str]) -> List[str]:
    out = []
    for s in tickers:
        q = fetch_last_quote(s)
        if q and q["price"] >= MIN_PRICE_NON_PENNY:
            out.append(s)
    # nếu API fail, vẫn trả danh sách gốc để không mất sóng
    return out or tickers

# =========================
# 5) Core signal engine
# =========================
def analyze_symbol(symbol: str, manual_pivot: Optional[float]) -> Optional[str]:
    df = fetch_intraday_5m(symbol)
    if df.empty or len(df) < 5:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"])
    o,h,l,c,v = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"]), float(last["volume"])

    # Pivot
    pivot = manual_pivot or darvas_pivot(df)
    if not pivot or pivot <= 0:
        return None

    # Điều kiện (1): đóng 2 nến trên pivot hoặc >+1% pivot
    cond_a = (prev["close"] > pivot and close > pivot) or (close >= pivot * (1 + BREAK_PCT))

    # Điều kiện (2): vol boost
    vol_ok, last_vol, avg_vol = volume_boost_ok(df)

    # Điều kiện (3): không râu trên quá dài
    wick_ok = not upper_wick_too_long((o,h,l,c,v))

    # ATR để set SL/Target
    atr = max(compute_atr(df), 0.01)
    entry_lo = round(max(pivot * (1 + 0.003), close - 0.15 * atr), 2)
    entry_hi = round(max(close, pivot * (1 + BREAK_PCT)), 2)
    t1 = round(close + 1.0*atr, 2)
    t2 = round(close + 2.0*atr, 2)
    sl = round(close - 1.3*atr, 2)

    model = "Darvas" if manual_pivot is None else "Zanger/CANSLIM"
    when = now_vn().strftime("%H:%M")

    # Kết quả 1: Breakout xác nhận
    if cond_a and vol_ok and wick_ok:
        return (f"{symbol} – BUY {entry_lo}–{entry_hi} | T1: {t1} | T2: {t2} | SL: {sl} | "
                f"⚡ Breakout xác nhận (vol {int(last_vol):,}/{int(avg_vol):,}, {when}) | Model: {model}")

    # Kết quả 2: Về vùng mua hỗ trợ (near pivot -2%)
    if pivot*(1-NEAR_BUY_BAND) <= close <= pivot*(1-0.003):
        note = f"🟢 Về vùng mua quanh {round(pivot*(1-NEAR_BUY_BAND),2)}–{round(pivot,2)}"
        return (f"{symbol} – {note} | Gợi ý: vào từng phần, SL dưới {round(pivot*(1-0.03),2)} | Model: {model}")

    # Kết quả 3: Cảnh báo rủi ro
    # - nến đảo chiều (close<open và râu trên dài) hoặc vol spike bất thường mà không qua pivot
    if (c < o and not wick_ok) or (vol_ok and not cond_a):
        return (f"{symbol} – ⚠️ Rủi ro: nến đảo chiều/vol bất thường gần {round(pivot,2)}. "
                f"Theo dõi chặt, tránh đu giá.")

    return None

# =========================
# 6) Market schedule (VN)
# =========================
def market_open_now(dt: Optional[datetime] = None) -> bool:
    d = dt or now_vn()
    wd = d.weekday()  # 0=Mon .. 6=Sun
    if wd >= 5:  # weekend
        return False
    hhmm = d.strftime("%H%M")
    return ("0855" <= hhmm <= "1135") or ("1255" <= hhmm <= "1505")

# =========================
# 7) Keepalive tự ping (không cần UptimeRobot)
# =========================
def self_ping():
    if DISABLE_SELF_PING: 
        return
    while True:
        try:
            requests.get(SELF_PING_URL, timeout=5)
        except Exception:
            pass
        time.sleep(300)  # 5 phút

# =========================
# 8) Main loop
# =========================
def run_scanner():
    # Tự ping để Render không ngủ
    threading.Thread(target=self_ping, daemon=True).start()

    # Lọc penny theo runtime (có thể bị rỗng nếu API lỗi → fallback toàn list)
    tickers = filter_non_penny(DEFAULT_TICKERS)

    send_telegram(f"🚀 {APP_NAME} started. Theo dõi {len(tickers)} mã. (tz=VN)")
    last_push: Dict[str, float] = {}       # chống spam (mỗi mã 1 lần/10 phút)
    cooldown_sec = 600

    while True:
        try:
            if market_open_now():
                for s in tickers:
                    try:
                        text = analyze_symbol(s, PIVOTS.get(s))
                        if text:
                            last_t = last_push.get(s, 0.0)
                            now_ts = time.time()
                            if now_ts - last_t >= cooldown_sec:
                                send_telegram(text)
                                last_push[s] = now_ts
                    except Exception:
                        # bỏ qua lỗi từng mã
                        pass
                time.sleep(20)  # nhịp quét
            else:
                # ngoài giờ: giảm tần suất nhưng vẫn ping
                time.sleep(60)
        except Exception:
            time.sleep(10)
