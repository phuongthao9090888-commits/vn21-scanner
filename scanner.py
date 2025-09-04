# scanner.py — VN21 Realtime Scanner (poll VNDirect, alert Telegram)
import os, time, json, datetime as dt, requests
from statistics import mean

# ====== ENV ======
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID   = os.getenv("CHAT_ID", "").strip()
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))

# Pivots: lấy từ ENV, nếu không có thì dùng mặc định
DEFAULT_PIVOTS = {
    "VPB": 35.4, "MBB": 28.2, "TCB": 40.1, "CTG": 52.5,
    "DCM": 40.0, "KDH": 36.5, "HPG": 29.5, "VHM": 106.0,
    "VIX": 38.5, "DDV": 31.5, "BSR": 27.5, "POW": 16.5,
    "REE": 65.5, "GMD": 70.0, "VNM": 62.5, "MWG": 80.0,
    # thêm nếu muốn…
}
try:
    PIVOTS = DEFAULT_PIVOTS.copy()
    piv_env = os.getenv("PIVOTS_JSON", "").strip()
    if piv_env:
        PIVOTS.update(json.loads(piv_env))
except Exception:
    PIVOTS = DEFAULT_PIVOTS

# Optional: đặt T1/T2/SL theo mã (nếu không có sẽ tự tính theo % pivot)
try:
    PLAN = json.loads(os.getenv("PLAN_JSON", "{}"))
except Exception:
    PLAN = {}

VN_TZ = dt.timezone(dt.timedelta(hours=int(os.getenv("TZ_OFFSET_HOURS", "7"))))

# ====== Utils ======
def now(): return dt.datetime.now(VN_TZ)

def is_trading_time(ts=None):
    ts = ts or now()
    if ts.weekday() >= 5:  # Sat/Sun
        return False
    m = ts.hour*60 + ts.minute
    return (540 <= m <= 690) or (780 <= m <= 905)  # 09:00–11:30 & 13:00–15:05

def fmt(x): return f"{x:.2f}".rstrip("0").rstrip(".")
def default_targets(pivot):  # T1=+3%, T2=+6%, SL=-3%
    return round(pivot*1.03,2), round(pivot*1.06,2), round(pivot*0.97,2)

def tg_send(text: str):
    if not (BOT_TOKEN and CHAT_ID): return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text}, timeout=10
        )
    except Exception:
        pass

# ====== Data sources ======
# 1) Quote nhanh theo “giá khớp” mới nhất (đủ cho cảnh báo vượt pivot)
def get_quote_vnd(symbol: str):
    url = "https://finfo-api.vndirect.com.vn/v4/stock_prices"
    params = {"symbol": symbol, "sort": "-time", "size": 1}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    arr = (r.json() or {}).get("data") or []
    if not arr: return None
    d = arr[0]
    return {
        "price": float(d.get("matchPrice") or d.get("closePrice") or 0),
        "time": d.get("time"),
        "pct": float(d.get("pctChange") or 0),
        "chg": float(d.get("change") or 0),
    }

# 2) (Tuỳ chọn) 5m candles nếu sau này bạn muốn nâng cấp điều kiện vol/râu nến
DCHART_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
def get_5m_candles(symbol: str, days_back=20):
    t = int(dt.datetime.now(dt.timezone.utc).timestamp())
    frm = t - days_back*86400
    url = "https://dchart-api.vndirect.com.vn/dchart/history"
    r = requests.get(url, params={"symbol": symbol, "resolution": "5", "from": frm, "to": t+600},
                     headers=DCHART_HEADERS, timeout=15)
    r.raise_for_status()
    js = r.json()
    if not js or "t" not in js or not js["t"]: return []
    out = []
    for i in range(len(js["t"])):
        out.append({"t": js["t"][i], "o": js["o"][i], "h": js["h"][i],
                    "l": js["l"][i], "c": js["c"][i], "v": js["v"][i]})
    return out

# ====== Alert logic (pivot-based, gọn nhẹ & nhạy) ======
_last_above = {}  # nhớ trạng thái đã vượt để không spam

def check_and_alert(symbol: str, quote: dict):
    if not quote: return
    px = quote["price"]; ts = quote["time"]
    pivot = PIVOTS.get(symbol.upper())
    if not pivot: return

    t1, t2, sl = ( (PLAN.get(symbol, {}).get("t1"),
                    PLAN.get(symbol, {}).get("t2"),
                    PLAN.get(symbol, {}).get("sl")) )
    if not all([t1, t2, sl]):
        t1, t2, sl = default_targets(pivot)

    # Chỉ cần vượt pivot là cảnh báo (nhẹ – hợp với yêu cầu “nhạy”)
    above = px >= float(pivot)
    prev = _last_above.get(symbol, False)

    if above and not prev:
        msg = (f"⚡ {symbol} BREAKOUT {fmt(px)} ≥ PIVOT {fmt(pivot)}\n"
               f"BUY: {fmt(pivot)}–{fmt(max(px, pivot*1.008))} | "
               f"T1: {fmt(t1)} | T2: {fmt(t2)} | SL: {fmt(sl)}\n"
               f"⏱ {ts}")
        tg_send(msg)
    _last_above[symbol] = above

def loop_poll():
    tickers = list(PIVOTS.keys())
    if not tickers:
        tg_send("⚠️ Không có mã trong PIVOTS_JSON — dừng quét.")
        return

    tg_send(
        "🚀 VN21-Scanner khởi động!\n"
        f"⏱ {now().strftime('%H:%M %d/%m/%Y')}\n"
        f"📈 Theo dõi: {', '.join(tickers)}\n"
        f"⏳ Poll mỗi {POLL_SECONDS}s trong giờ HOSE."
    )

    while True:
        try:
            if not is_trading_time():
                time.sleep(30); continue

            for sym in tickers:
                try:
                    q = get_quote_vnd(sym)
                    check_and_alert(sym, q)
                except Exception:
                    pass

            time.sleep(POLL_SECONDS)
        except Exception:
            time.sleep(POLL_SECONDS)
