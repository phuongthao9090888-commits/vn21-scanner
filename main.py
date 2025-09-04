# ===== VN21 Breakout Scanner (Render Worker) =====
import os, time, requests, datetime as dt
from statistics import mean
from dateutil import tz

# ---- ENV ----
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID   = os.environ["CHAT_ID"]
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
TZ = tz.tzoffset(None, 7*3600)

TICKERS_PIVOT = {
    "VPB": 35.4, "MBB": 28.2, "TCB": 40.1, "CTG": 52.5,
    "DCM": 40.0, "KDH": 36.5, "HPG": 29.5, "VHM": 106.0,
    "VIX": 38.5, "DDV": 31.5, "BSR": 27.5, "POW": 16.5,
    "REE": 65.5, "GMD": 70.0, "VNM": 62.5, "MWG": 80.0,
}
PLAN = {}  # có thể override T1/T2/SL tại đây

# ---- utils ----
def now(): return dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
def to_unix(t): return int(t.timestamp())
def during_session(t):
    wd = t.weekday()
    if wd >= 5: return False
    hm = t.hour*60 + t.minute
    return (540 <= hm <= 690) or (780 <= hm <= 900)

def send_tele(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text}, timeout=10
        )
    except Exception as e:
        print("Telegram error:", e)

def fmt(x): 
    try: return f"{x:.2f}".rstrip("0").rstrip(".")
    except: return str(x)

def default_targets(pv):
    return round(pv*1.03,2), round(pv*1.06,2), round(pv*0.97,2)

# ---- VNDirect 5m candles (cần User-Agent) ----
HEADERS = {"User-Agent": "Mozilla/5.0 (VN21 Scanner)"}
def fetch_5m(sym, days_back=20):
    t = now()
    frm, to = to_unix(t - dt.timedelta(days=days_back)), to_unix(t + dt.timedelta(minutes=10))
    r = requests.get(
        "https://dchart-api.vndirect.com.vn/dchart/history",
        params={"symbol": sym, "resolution": "5", "from": frm, "to": to},
        headers=HEADERS, timeout=15
    )
    r.raise_for_status()
    js = r.json()
    if not js or "t" not in js or not js["t"]: return []
    return [{"t": js["t"][i], "o": js["o"][i], "h": js["h"][i],
             "l": js["l"][i], "c": js["c"][i], "v": js["v"][i]}
            for i in range(len(js["t"]))]

def no_long_upper_wick(c):
    body = abs(c["c"] - c["o"])
    if body <= 0: return False
    upper = c["h"] - max(c["o"], c["c"])
    return upper <= 0.6*body

def volume_benchmark(cs):
    vols = [c["v"] for c in cs[:-1]] or [c["v"] for c in cs]
    return (mean(vols)/5.0) if vols else 0.0

def price_cond_early(c2, pv): return (c2["c"] > pv) or (c2["c"] >= pv*1.005)
def price_cond_confirmed(c1, c2, pv): return ((c1["c"]>pv and c2["c"]>pv) or (c2["c"]>=pv*1.01))

def check_signals(cs, pv):
    if len(cs) < 2: return None, None, {}
    c1, c2 = cs[-2], cs[-1]
    avg_per_min = volume_benchmark(cs)
    early = price_cond_early(c2, pv) and (c2["v"] >= 1.2*avg_per_min*5)
    confirmed = price_cond_confirmed(c1, c2, pv) and (c2["v"] >= 1.5*avg_per_min*5) and no_long_upper_wick(c2)
    return early, confirmed, {"c1": c1, "c2": c2, "avg_per_min": avg_per_min}

def entry_range(pv, last_close):
    lo, hi = pv, max(last_close, pv*1.008)
    return round(lo,2), round(hi,2)

def model_for(s):
    if s in {"VPB","MBB","TCB","CTG"}: return "CANSLIM"
    if s in {"KDH","VHM"}: return "Darvas"
    return "Zanger"

# ---- main loop ----
sent_today = {"early": set(), "confirmed": set()}
cur_date = now().date()
send_tele("🚀 VN21 Scanner khởi động — chế độ NHẠY VỪA (5m, 20s).")

while True:
    try:
        t = now()
        if t.date() != cur_date:
            sent_today = {"early": set(), "confirmed": set()}
            cur_date = t.date()

        if during_session(t):
            for sym, pv in TICKERS_PIVOT.items():
                try:
                    bars = fetch_5m(sym)
                    if len(bars) < 2: continue
                    early, confirmed, info = check_signals(bars, pv)
                    c2 = info.get("c2", bars[-1])
                    last_close = c2["c"]
                    lo, hi = entry_range(pv, last_close)
                    t1, t2, sl = ( (PLAN[sym]["t1"], PLAN[sym]["t2"], PLAN[sym]["sl"])
                                   if sym in PLAN else default_targets(pv) )
                    vol_note = f"vol={int(c2['v']):,} vs avg5m≈{int(info['avg_per_min']*5):,}"
                    ts = dt.datetime.fromtimestamp(c2["t"], tz=TZ).strftime("%H:%M")

                    if early and sym not in sent_today["early"]:
                        msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
                               f"SL: {fmt(sl)} | 🔔 Early breakout ({vol_note}, {ts}) | Model: {model_for(sym)}")
                        send_tele(msg); print(msg); sent_today["early"].add(sym)

                    if confirmed and sym not in sent_today["confirmed"]:
                        msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
                               f"SL: {fmt(sl)} | ⚡ Breakout xác nhận ({vol_note}, {ts}) | Model: {model_for(sym)}")
                        send_tele(msg); print(msg); sent_today["confirmed"].add(sym)

                except Exception as e:
                    print(f"[{sym}] error:", e)

        time.sleep(POLL_SECONDS)
    except Exception as e:
        print("Loop error:", e)
        time.sleep(POLL_SECONDS)
