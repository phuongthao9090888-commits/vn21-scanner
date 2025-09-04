# scanner.py — VN21 Realtime Scanner (VNDirect) — Heatmap + Advanced Logic
import os, time, json, datetime as dt, requests, math
from statistics import mean

# ====== ENV ======
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID   = os.getenv("CHAT_ID", "").strip()

# tickers: danh sách mã từ heatmap (CSV). Ví dụ: "VPB,MBB,TCB,CTG,DCM,KDH,HPG,VHM,VIX,DDV,BSR,POW,REE,GMD,VNM,MWG,SSI,HCM,VND,PVD,PVS,GEX,KBC"
TICKERS_CSV = os.getenv("TICKERS", "").strip()

# pivot tùy chọn cho 1 phần mã (JSON 1 dòng). Nếu mã không có pivot → auto Darvas.
# Ví dụ: {"VPB":35.4,"MBB":28.2,"POW":16.5}
PIVOTS_JSON = os.getenv("PIVOTS_JSON", "").strip()

# Kế hoạch T1/T2/SL cố định từng mã (nếu không set sẽ tính % theo pivot)
# Ví dụ: {"POW":{"t1":17.5,"t2":19.0,"sl":15.8},"VNM":{"t1":65,"t2":70,"sl":60}}
PLAN_JSON   = os.getenv("PLAN_JSON", "").strip()

# Poll & thời gian
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
VN_TZ = dt.timezone(dt.timedelta(hours=int(os.getenv("TZ_OFFSET_HOURS", "7"))))

# ====== thresholds (tùy chỉnh ở Render → Environment) ======
VOL_MULT_EARLY   = float(os.getenv("VOL_MULT_EARLY",   "1.2"))  # early ≥ 1.2× avg 5m
VOL_MULT_CONFIRM = float(os.getenv("VOL_MULT_CONFIRM", "1.5"))  # confirmed ≥ 1.5× avg 5m
WICK_RATIO_MAX   = float(os.getenv("WICK_RATIO_MAX",   "0.6"))  # upper wick ≤ 60% thân
CONFIRM_BARS     = int(os.getenv("CONFIRM_BARS",       "2"))    # số nến đóng trên pivot
FAKE_LOOKAHEAD   = int(os.getenv("FAKE_LOOKAHEAD",     "3"))    # kiểm tra fake sau breakout
COOLDOWN_MIN     = int(os.getenv("COOLDOWN_MIN",       "20"))   # tránh spam sau fake (phút)

# MA / ATR filters (bật/tắt)
USE_MA_FILTER    = int(os.getenv("USE_MA_FILTER",      "1"))    # 1: bật lọc MA
MA_FAST          = int(os.getenv("MA_FAST",            "20"))   # SMA nhanh (bars 5m)
MA_SLOW          = int(os.getenv("MA_SLOW",            "50"))   # SMA chậm (bars 5m)

USE_ATR_FILTER   = int(os.getenv("USE_ATR_FILTER",     "1"))    # 1: bật ATR sanity
ATR_LEN          = int(os.getenv("ATR_LEN",            "14"))

# Darvas settings
USE_DARVAS       = int(os.getenv("USE_DARVAS",         "1"))
DARVAS_LOOKBACK  = int(os.getenv("DARVAS_LOOKBACK",    "60"))
DARVAS_MIN_SIDE  = int(os.getenv("DARVAS_MIN_SIDE",    "8"))

# Position sizing (gợi ý KHỐI LƯỢNG theo rủi ro % vốn)
RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE",   "0.01"))   # 1% vốn
ACCOUNT_VALUE    = float(os.getenv("ACCOUNT_VALUE",    "30000000"))  # 30tr mặc định
MARGIN_FACTOR    = float(os.getenv("MARGIN_FACTOR",    "1.0"))    # 1.0 = không margin; 1.5 = 1:0.5

# ====== globals ======
def now(): return dt.datetime.now(VN_TZ)

def is_trading_time(ts=None):
    ts = ts or now()
    if ts.weekday() >= 5:  # Sat/Sun
        return False
    m = ts.hour*60 + ts.minute
    # 09:00–11:30 & 13:00–15:05
    return (540 <= m <= 690) or (780 <= m <= 905)

def fmt(x): return f"{x:.2f}".rstrip("0").rstrip(".")

def default_targets(pivot):  # T1=+3%, T2=+6%, SL=-3% từ pivot
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

# ====== Data: VNDirect ======
# 5-minute candles (để vol/ma/atr/darvas)
DCHART_HEADERS = {"User-Agent":"Mozilla/5.0","Accept":"application/json"}
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

# Last quote (nhanh) — dùng để hiển thị giá hiện tại
def get_last_quote(symbol: str):
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

# ====== TA helpers ======
def sma(vals, n):
    if len(vals) < n or n <= 0: return None
    return sum(vals[-n:]) / n

def atr(candles, n=14):
    # Wilder ATR: dùng TR = max(h-l, |h-prev_c|, |l-prev_c|)
    if len(candles) < n+1: return None
    trs = []
    for i in range(1, len(candles)):
        h, l, c_prev = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    if len(trs) < n: return None
    return sum(trs[-n:]) / n

def avg_vol_5m(candles):
    vols = [c["v"] for c in candles[:-1]] or [c["v"] for c in candles]
    return mean(vols) if vols else 0.0

def upper_wick_ok(c, max_ratio=WICK_RATIO_MAX):
    body = abs(c["c"] - c["o"])
    if body <= 0: return False
    upper = c["h"] - max(c["o"], c["c"])
    return upper <= max_ratio * body

def darvas_box(candles, lookback=DARVAS_LOOKBACK, min_side=DARVAS_MIN_SIDE):
    if len(candles) < max(20, min_side+5): return (None, None)
    recent = candles[-lookback:] if len(candles) > lookback else candles[:]
    tail = recent[-max(min_side, len(recent)//4):]
    box_high = max(c["h"] for c in tail)
    box_low  = min(c["l"] for c in tail)
    # box hợp lệ nếu dày > 2% & hẹp tương đối (tránh noise)
    if box_high <= 0 or (box_high - box_low) < 0.02*box_high:
        return (None, None)
    return (round(box_high,2), round(box_low,2))

# ====== Signals ======
def price_confirmed_over_pivot(candles, pivot, bars=CONFIRM_BARS):
    if len(candles) < bars: return False
    last = candles[-bars:]
    return all(c["c"] > pivot for c in last)

def fake_break_after_breakout(candles, pivot, look_ahead=FAKE_LOOKAHEAD):
    # Fake: trong N nến sau breakout, đóng dưới pivot hoặc nến đỏ lớn với vol cao
    if len(candles) < look_ahead+1: return False
    window = candles[-look_ahead:]
    # điều kiện đơn giản: có nến đóng < pivot và biên độ > 0.6% pivot
    for c in window:
        if c["c"] < pivot and (pivot - c["c"]) >= 0.006 * pivot:
            return True
    return False

def position_size(price_entry, stop_loss):
    risk_amount = ACCOUNT_VALUE * RISK_PER_TRADE
    per_share_risk = max(price_entry - stop_loss, 0.01)
    qty = (risk_amount * MARGIN_FACTOR) / per_share_risk
    # làm tròn về bội số 100 (lô lẻ VN có thể 100/10 tuỳ sàn; ta dùng 100)
    lots = max(int(qty // 100), 0)
    return lots * 100

# ====== State ======
_pivots_map = {}
try:
    if PIVOTS_JSON:
        _pivots_map.update(json.loads(PIVOTS_JSON))
except Exception:
    pass

try:
    _plan_map = json.loads(PLAN_JSON) if PLAN_JSON else {}
except Exception:
    _plan_map = {}

# danh sách tickers
TICKERS = [s.strip().upper() for s in TICKERS_CSV.split(",") if s.strip()] or list(_pivots_map.keys())

_last_above = {}         # {sym: bool}
_confirmed_sent = set()  # set(sym)
_fake_cooldown_until = {}# {sym: timestamp}

def model_for(sym):
    # gợi ý mô hình
    if sym in {"VPB","MBB","TCB","CTG"}: return "CANSLIM"
    if sym in {"KDH","VHM"}: return "Darvas"
    return "Zanger"

# ====== Main check per symbol ======
def scan_symbol(sym):
    # 1) load candles
    candles = get_5m_candles(sym, days_back=20)
    if len(candles) < max(ATR_LEN+2, MA_SLOW+2, 10):
        return

    closes = [c["c"] for c in candles]
    vols   = [c["v"] for c in candles]
    last   = candles[-1]
    prev   = candles[-2]

    # 2) pivot: ưu tiên PIVOTS_JSON, nếu không có → Darvas box
    pivot = _pivots_map.get(sym)
    box_hi = box_lo = None
    if pivot is None and USE_DARVAS:
        box_hi, box_lo = darvas_box(candles)
        pivot = box_hi  # breakout trên box high
    if pivot is None:
        return  # không có pivot để so, bỏ qua

    # 3) MA filter
    if USE_MA_FILTER:
        ma_fast = sma(closes, MA_FAST)
        ma_slow = sma(closes, MA_SLOW)
        if not ma_fast or not ma_slow: 
            return
        # Lọc nhẹ: giá hiện tại > MA_FAST và MA_FAST ≥ MA_SLOW (xu hướng lên)
        if not (last["c"] > ma_fast and ma_fast >= ma_slow):
            return

    # 4) ATR sanity (tránh vào khi quá xa SL)
    if USE_ATR_FILTER:
        _atr = atr(candles, ATR_LEN)
        if _atr:
            # nếu khoảng cách entry - pivot > 1.5 ATR -> coi chừng đu đỉnh
            if (last["c"] - pivot) > 1.5 * _atr:
                # có thể bỏ qua hoặc chỉ gửi Early. Ở đây bỏ qua để thận trọng.
                return

    # 5) Volume benchmark 5m
    avg5m = avg_vol_5m(candles)
    vol_ok_early   = last["v"] >= VOL_MULT_EARLY   * avg5m
    vol_ok_confirm = last["v"] >= VOL_MULT_CONFIRM * avg5m

    # 6) Price logic
    above_now = last["c"] >= pivot or last["c"] >= 1.005 * pivot
    confirmed = price_confirmed_over_pivot(candles, pivot, CONFIRM_BARS)

    # 7) Wick filter
    wick_ok = upper_wick_ok(last, WICK_RATIO_MAX)

    # 8) Cooldown (nếu vừa fake)
    cd_until = _fake_cooldown_until.get(sym, 0)
    if time.time() < cd_until:
        return  # đang cooldown, bỏ qua

    # 9) Targets / SL
    if sym in _plan_map:
        t1, t2, sl = _plan_map[sym].get("t1"), _plan_map[sym].get("t2"), _plan_map[sym].get("sl")
        if not all([t1, t2, sl]):
            t1, t2, sl = default_targets(pivot)
    else:
        t1, t2, sl = default_targets(pivot)

    # 10) Entry gợi ý
    lo = pivot
    hi = max(last["c"], pivot*1.008)  # trượt nhẹ 0.8%
    qty = position_size(hi, sl)

    # ===== Early alert =====
    if above_now and vol_ok_early and not _last_above.get(sym, False):
        msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
               f"SL: {fmt(sl)} | 🔔 Early ({int(last['v']):,} vs avg5m≈{int(avg5m):,}) | "
               f"Model: {model_for(sym)} | Qty≈{qty}")
        # nếu có darvas box, thêm note
        if box_hi and box_lo:
            msg += f"\n📦 Darvas: {fmt(box_lo)}–{fmt(box_hi)}"
        tg_send(msg)

    # ===== Confirmed alert =====
    if confirmed and vol_ok_confirm and wick_ok and (sym not in _confirmed_sent):
        msg = (f"{sym} – BUY {fmt(lo)}–{fmt(hi)} | T1: {fmt(t1)} | T2: {fmt(t2)} | "
               f"SL: {fmt(sl)} | ⚡ Confirmed ({int(last['v']):,} vs avg5m≈{int(avg5m):,}) | "
               f"Model: {model_for(sym)} | Qty≈{qty}")
        if box_hi and box_lo:
            msg += f"\n📦 Darvas: {fmt(box_lo)}–{fmt(box_hi)}"
        tg_send(msg)
        _confirmed_sent.add(sym)

    # ===== Fake breakout detection & cooldown =====
    if confirmed and fake_break_after_breakout(candles, pivot, FAKE_LOOKAHEAD):
        mins = COOLDOWN_MIN
        _fake_cooldown_until[sym] = time.time() + mins*60
        tg_send(f"⚠️ {sym} FAKE breakout sau xác nhận → tạm ngưng {mins}′. (Close < pivot hoặc đuối lực)")

    # Update state edge
    _last_above[sym] = above_now

# ====== LOOP ======
def loop_poll():
    if not TICKERS:
        tg_send("⚠️ Không có TICKERS trong ENV, vui lòng set TICKERS=VPB,MBB,TCB,...")
        return

    tg_send(
        "🚀 VN21-Scanner nâng cấp khởi động!\n"
        f"⏱ {now().strftime('%H:%M %d/%m/%Y')}\n"
        f"📈 Theo dõi: {', '.join(TICKERS)}\n"
        f"⏳ Poll mỗi {POLL_SECONDS}s trong giờ HOSE.\n"
        f"🔧 Lọc: VOL({VOL_MULT_EARLY}/{VOL_MULT_CONFIRM}) • Wick≤{int(WICK_RATIO_MAX*100)}% • MA{MA_FAST}/{MA_SLOW} • ATR{ATR_LEN} • Darvas"
    )

    while True:
        try:
            if not is_trading_time():
                time.sleep(30); continue

            for sym in TICKERS:
                try:
                    scan_symbol(sym)
                except Exception as e:
                    # tránh dừng loop vì 1 mã
                    pass

            time.sleep(POLL_SECONDS)
        except Exception:
            time.sleep(POLL_SECONDS)
