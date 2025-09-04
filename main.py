# ===== VN21 Breakout Scanner — Early vs Confirmed =====
import time, requests, datetime as dt
from statistics import mean

# ---------- CONFIG ----------
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID   = "YOUR_CHAT_ID"
TZ = dt.timezone(dt.timedelta(hours=7))
POLL_SECONDS = 20  # quét nhạy vừa phải

TICKERS_PIVOT = {
    "VPB": 35.4, "MBB": 28.2, "TCB": 40.1, "CTG": 52.5,
    "DCM": 40.0, "KDH": 36.5, "HPG": 29.5, "VHM": 106.0,
    "VIX": 38.5, "DDV": 31.5, "BSR": 27.5, "POW": 16.5,
    "REE": 65.5, "GMD": 70.0, "VNM": 62.5, "MWG": 80.0,
}

def now(): return dt.datetime.now(TZ)
def to_unix(t): return int(t.timestamp())

def during_session(t):
    wd = t.weekday()
    if wd >= 5: return False
    hm = t.hour * 60 + t.minute
    return (540 <= hm <= 690) or (780 <= hm <= 900)

def send_tele(text: str):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text},
            timeout=10
        )
        if r.status_code == 200:
            print(f"[Telegram] ✅ Sent: {text[:40]}...")
        else:
            print(f"[Telegram] ❌ Error {r.status_code}: {r.text}")
    except Exception as e:
        print("Telegram error:", e)

def fmt(x): return f"{x:.2f}".rstrip("0").rstrip(".")
def default_targets(pivot): return round(pivot*1.03,2), round(pivot*1.06,2), round(pivot*0.97,2)

# ---------- VNDirect fetch ----------
def fetch_5m(symbol: str, days_back=20):
    t = now()
    frm = to_unix(t - dt.timedelta(days=days_back))
    to  = to_unix(t + dt.timedelta(minutes=10))
    url = "https://dchart-api.vndirect.com.vn/dchart/history"
    r = requests.get(url, params={"symbol": symbol, "resolution": "5", "from": frm, "to": to}, timeout=15)
    if r.status_code == 403:
        print(f"[{symbol}] 🚫 403 Forbidden (VNDirect chặn IP Render).")
        return []
    r.raise_for_status()
    js = r.json()
    if not js or "t" not in js or not js["t"]:
        return []
    return [{"t": js["t"][i], "o": js["o"][i], "h": js["h"][i],
             "l": js["l"][i], "c": js["c"][i], "v": js["v"][i]}
            for i in range(len(js["t"]))]

def no_long_upper_wick(c):
    body = abs(c["c"] - c["o"])
    if body <= 0: return False
    upper = c["h"] - max(c["o"], c["c"])
    return upper <= 0.6 * body

# ---------- SIGNAL CHECK ----------
def volume_benchmark(candles):
    vols = [c["v"] for c in candles[:-1]] or [c["v"] for c in candles]
    return (mean(vols) / 5.0) if vols else 0.0

def price_cond_early(c2, pivot): return (c2["c"] > pivot) or (c2["c"] >= pivot*1.005)
def price_cond_confirmed(c1, c2, pivot): return ((c1["c"] > pivot and c2["c"] > pivot) or (c2["c"] >= pivot*1.01))

def check_signals(candles, pivot):
    if len(candles) < 2: return None, None, {}
    c1, c2 = candles[-2], candles[-1]
    avg_per_min = volume_benchmark(candles)
    early = price_cond_early(c2, pivot) and (c2["v"] >= 1.2 * avg_per_min * 5)
    confirmed = price_cond_confirmed(c1, c2, pivot) and (c2["v"] >= 1.5 * avg_per_min * 5) and no_long_upper_wick(c2)
    return early, confirmed, {"c1": c1, "c2": c2, "avg_per_min": avg_per_min}

def entry_range(pivot, last_close):
    lo = pivot
    hi = max(last_close, pivot*1.008)
    return (round(lo,2), round(hi,2))

def model_for(sym):
    if sym in {"VPB","MBB","TCB","CTG"}: return "CANSLIM"
    if sym in {"KDH","VHM"}: return "Darvas"
    return "Zanger"

# ---------- MAIN ----------
sent_today = {"early": set(), "confirmed": set()}
current_date = now().date()

print("🚀 VN21 Scanner starting on Render...")
send_tele("🚀 VN21 Scanner khởi động (Render 24/7).")

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
                    t1, t2, sl = default_targets(pivot)
                    vol_note = f"vol={int(c2['v']):,} vs avg5m≈{int(info['avg_per_min']*5):,}"
                    ts = dt.datetime.fromtimestamp(c2["t"], TZ).strftime("%H:%M")

                    if early and sym not in sent_today["early"]:
                        msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
                               f"SL: {fmt(sl)} | 🔔 Early breakout ({vol_note}, {ts}) | Model: {model_for(sym)}")
                        send_tele(msg); print("[EARLY]", msg)
                        sent_today["early"].add(sym)

                    if confirmed and sym not in sent_today["confirmed"]:
                        msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
                               f"SL: {fmt(sl)} | ⚡ Breakout xác nhận ({vol_note}, {ts}) | Model: {model_for(sym)}")
                        send_tele(msg); print("[CONFIRMED]", msg)
                        sent_today["confirmed"].add(sym)

                except Exception as e:
                    print(f"[{sym}] error:", e)

        time.sleep(POLL_SECONDS)
    except Exception as e:
        print("Loop error:", e)
        time.sleep(POLL_SECONDS)
