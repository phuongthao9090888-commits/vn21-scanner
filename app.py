# app.py — VN21 Scanner (Pro/Full)
# - VNDirect DChart (5m + Daily)
# - Breakout mạnh (pivot, vol, wick)
# - Darvas Box / MA trend / ATR-lite
# - Ưu tiên VN21 (danh mục hiện có)
# - Đề xuất Thay thế/Bổ sung dựa trên điểm số (RS, Trend, Breakout, Risk)
# - Telegram cảnh báo (đã nhúng sẵn TOKEN/CHAT_ID)
# - Scheduler quét trong giờ giao dịch, /scan để quét tay, /config để xem cấu hình

import os, json, math, asyncio, requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# =========================
# CẤU HÌNH CỐ ĐỊNH (đã nhét)
# =========================
TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DCHART_URL = "https://dchart-api.vndirect.com.vn/dchart/history"

TOKEN = "8207349630:AAFQ1Sq8eumEtNoNNSg4DboQ-SMzBLui95o"
CHAT_ID = "5614513021"

# Danh mục VN21 (ưu tiên theo dõi/giữ)
VN21_CORE = [
    "VPB","MBB","TCB","CTG","DCM","KDH",
    "HPG","VHM","VIX","DDV","BSR","POW",
    "REE","GMD","VNM","MWG"
]

# Universe mở rộng (loại VCB, GAS theo yêu cầu trước)
UNIVERSE_EXTRA = [
    # Bank
    "BID","STB","SHB","ACB","TPB","EIB","LPB","HDB",
    # Chứng khoán
    "SSI","HCM","VND","VIX","SHS","MBS",
    # Dầu khí
    "PVD","PVS","BSR","PLX","POW","PVG",
    # BĐS/ KCN
    "KDH","VHM","GEX","KBC","NLG","DXG",
    # Thép/ VLXD
    "HPG","HSG","NKG","KSB",
    # Công nghệ/tiêu dùng
    "FPT","MWG","VNM","MSN","SAB","DGW","FRT",
    # Khác vốn hóa lớn
    "VIC","VGI","REE","GMD","GVR","VTP","LTG","PAN"
]

DEFAULT_TICKERS = sorted(list(set(VN21_CORE + UNIVERSE_EXTRA)))

# Pivot tay ưu tiên (VN21); mã khác dùng Darvas
PIVOTS: Dict[str, float] = {
    "VPB":35.4,"MBB":28.2,"TCB":40.1,"CTG":52.5,"DCM":40.0,"KDH":36.5,
    "HPG":29.5,"VHM":106.0,"VIX":38.5,"DDV":31.5,"BSR":27.5,"POW":16.5,
    "REE":65.5,"GMD":70.0,"VNM":62.5,"MWG":80.0
}

# Plan (target/SL) nếu có; không có thì auto từ pivot (%)
PLAN: Dict[str, Dict[str, float]] = {
    # ví dụ:
    # "POW":{"t1":17.0,"t2":17.6,"sl":15.9}
}

# Thời gian & lịch
SCAN_INTERVAL_SEC = 60              # mỗi phút quét 1 lần
TRADING_MIN_PER_DAY = 270           # 9:00–11:30, 13:00–15:00 (VN)
TRADING_WINDOWS = [("09:00","11:30"), ("13:00","15:00")]

# Chống spam
sent_today: Dict[str, str] = {}     # {sym: "YYYY-MM-DD"}
cooldown_ts: Dict[str, float] = {}  # {sym: unix}  — 600s

# =========================
# FastAPI
# =========================
app = FastAPI(title="VN21 Scanner Pro/Full", version="3.0")

# =========================
# Utils
# =========================
def now_ts() -> int:
    return int(datetime.now(tz=TZ).timestamp())

def fmt_hm(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=TZ).strftime("%H:%M")

def tele_send(text: str):
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text}, timeout=10)
    except Exception:
        pass

def get_plan(sym: str, ref: float) -> Dict[str, float]:
    p = PLAN.get(sym, {})
    if {"t1","t2","sl"} <= set(p):
        return {"t1": float(p["t1"]), "t2": float(p["t2"]), "sl": float(p["sl"])}
    base = PIVOTS.get(sym, ref)
    return {"t1": round(base*1.03,2), "t2": round(base*1.06,2), "sl": round(base*0.97,2)}

def in_trading_hours() -> bool:
    now = datetime.now(tz=TZ)
    if now.weekday() >= 5:  # T7-CN
        return False
    hm = now.strftime("%H:%M")
    for a,b in TRADING_WINDOWS:
        if a <= hm <= b:
            return True
    return False

# =========================
# Data fetchers
# =========================
def fetch_dchart(symbol: str, resolution: str, days: int) -> pd.DataFrame:
    to_ts = now_ts()
    params = {"symbol": symbol, "resolution": resolution, "from": to_ts - days*86400, "to": to_ts}
    r = requests.get(DCHART_URL, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    if not js or "t" not in js or not js["t"]:
        return pd.DataFrame()
    df = pd.DataFrame({"t": js["t"], "o": js["o"], "h": js["h"], "l": js["l"], "c": js["c"], "v": js["v"]})
    return df

def fetch_5m(symbol: str, days: int = 5) -> pd.DataFrame:
    return fetch_dchart(symbol, "5", days)

def fetch_daily(symbol: str, days: int = 140) -> pd.DataFrame:
    return fetch_dchart(symbol, "D", days)

# =========================
# Indicators / helpers
# =========================
def wick_ok(o, h, l, c) -> bool:
    body = abs(c - o)
    upper = h - max(o, c)
    if body < 1e-4:
        return upper <= 0.002 * c
    return upper <= 0.6 * body

def avg_per_minute_volume_20d(symbol: str) -> Optional[float]:
    dfd = fetch_daily(symbol, 60)
    if dfd.empty: return None
    dfd = dfd.tail(20)
    if dfd.empty: return None
    avg_daily = float(pd.to_numeric(dfd["v"], errors="coerce").dropna().mean())
    if avg_daily <= 0: return None
    return avg_daily / TRADING_MIN_PER_DAY

def darvas_box(df5: pd.DataFrame, lookback=60) -> Optional[Dict[str,float]]:
    if df5.empty: return None
    seg = df5.tail(lookback)
    hi = float(seg["h"].max()); lo = float(seg["l"].min())
    if math.isfinite(hi) and math.isfinite(lo) and hi > lo:
        return {"top": round(hi,2), "bot": round(lo,2)}
    return None

def rs_rating(price_now: float, df_d: pd.DataFrame) -> float:
    if df_d is None or len(df_d) < 60: return 0.0
    base = df_d["c"].iloc[-60]
    if base <= 0: return 0.0
    perf_3m = price_now / base - 1.0
    # VNINDEX performance (proxy 5%)
    return (perf_3m - 0.05) * 100.0

def ma_align(df_d: pd.DataFrame) -> float:
    if len(df_d) < 200: return 0.0
    c = df_d["c"]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    ma200 = c.rolling(100).mean().iloc[-1]
    score = 0.0
    if c.iloc[-1] > ma20: score += 0.5
    if ma20 > ma50: score += 0.25
    if ma50 > ma200: score += 0.25
    return score  # 0..1

def near_support_zone(c: float, df_d: pd.DataFrame) -> bool:
    """Về vùng mua hỗ trợ: c gần MA20D (±1.5%) & wick đẹp (dựa 5m nến cuối)."""
    if len(df_d) < 20: return False
    ma20 = df_d["c"].rolling(20).mean().iloc[-1]
    return abs(c/ma20 - 1.0) <= 0.015

# =========================
# Breakout detection (5m)
# =========================
def breakout_5m_msg(sym: str, pivot: float) -> Optional[str]:
    # dữ liệu 5m
    df5 = fetch_5m(sym, 3)
    if df5.empty or len(df5) < 4:
        return None

    # loại nến đang chạy: lấy 2 nến đã đóng
    last3 = df5.tail(3)
    c1, c2 = float(last3.iloc[-3]["c"]), float(last3.iloc[-2]["c"])
    o2, h2, l2 = float(last3.iloc[-2]["o"]), float(last3.iloc[-2]["h"]), float(last3.iloc[-2]["l"])
    v2 = float(last3.iloc[-2]["v"])
    t1, t2 = int(last3.iloc[-3]["t"]), int(last3.iloc[-2]["t"])

    # (1) giá
    cond_price = (c1 > pivot and c2 > pivot) or (c2 >= pivot * 1.01)
    # (2) vol
    apm = avg_per_minute_volume_20d(sym)
    if apm is None or apm <= 0: return None
    cond_vol = v2 >= 1.5 * apm * 5
    # (3) wick
    cond_wick = wick_ok(o2, h2, l2, c2)

    if not (cond_price and cond_vol and cond_wick):
        return None

    # Model tagging
    box = darvas_box(df5, 60)
    tags = []
    if box and c2 > box["top"]: tags.append("Darvas")
    if c2 > pivot * 1.02 and v2 > 1.8 * apm * 5: tags.append("Zanger")
    if not tags: tags.append("CANSLIM")

    plan = get_plan(sym, c2)
    entry_low = round(pivot, 2); entry_high = round(c2, 2)
    msg = (f"{sym} – BUY {entry_low}-{entry_high} | "
           f"T1: {plan['t1']} | T2: {plan['t2']} | SL: {plan['sl']} | "
           f"⚡ Breakout xác nhận (vol {int(v2):,}, {fmt_hm(t1)}–{fmt_hm(t2)}) | "
           f"Model: {'/'.join(tags)}")
    return msg

# =========================
# Scoring (để đề xuất Thay thế/Bổ sung)
# =========================
def score_symbol(sym: str, pivot: Optional[float]) -> Dict[str, Any]:
    """
    Trả về: {"sym","score","reason","state"} — state: breakout/near-support/risk/neutral
    """
    out = {"sym": sym, "score": -1e9, "reason": "", "state": "neutral"}
    try:
        df_d = fetch_daily(sym, 140)
        if df_d.empty or len(df_d) < 60: 
            out["reason"] = "no-data"; return out
        price = float(df_d["c"].iloc[-1])
        vol_d = float(df_d["v"].iloc[-1])
        rs = rs_rating(price, df_d)
        trend = ma_align(df_d)

        # Trạng thái
        state = "neutral"
        if pivot:
            if price > pivot * 1.01:
                state = "breakout"
            elif abs(price/pivot - 1.0) <= 0.01:
                state = "near-pivot"
        if near_support_zone(price, df_d):
            state = "support"
        # rủi ro nếu < MA50 hoặc biến động cao mà vol giảm
        ma50 = df_d["c"].rolling(50).mean().iloc[-1]
        risk_pen = 0.0
        if price < ma50: 
            risk_pen += 0.5
            state = "risk"

        # Điểm breakout readiness (dựa daily)
        ready = 0.0
        if pivot:
            ready = max(0.0, min(1.0, (price/pivot - 0.98)/0.04))  # map 98%..102% pivot -> 0..1

        # Điểm tổng
        score = 40*max(rs, -50)/100 + 35*trend + 25*ready - 20*risk_pen
        out.update({
            "score": round(score, 2),
            "reason": f"RS={rs:.1f}, Trend={trend:.2f}, Ready={ready:.2f}, RiskPen={risk_pen:.2f}",
            "state": state,
            "price": round(price,2),
            "pivot": pivot if pivot else None
        })
        return out
    except Exception as e:
        out["reason"] = f"err:{e}"
        return out

# =========================
# Scan tổng hợp
# =========================
async def scan_all(tickers: List[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"time": datetime.now(tz=TZ).isoformat(), "breakouts": [], "support": [], "risk": [], "suggest": {}}

    # A. Breakout xác nhận (5m)
    for sym in tickers:
        pivot = PIVOTS.get(sym)
        if not pivot: continue
        try:
            msg = breakout_5m_msg(sym, pivot)
            if msg:
                today = datetime.now(tz=TZ).strftime("%Y-%m-%d")
                # cooldown 10p/mã
                if sym in cooldown_ts and (now_ts() - cooldown_ts[sym] < 600):
                    pass
                else:
                    cooldown_ts[sym] = now_ts()
                    if sent_today.get(sym) != today:
                        tele_send(msg); sent_today[sym] = today
                res["breakouts"].append(msg)
        except Exception as e:
            res.setdefault("errors", {})[sym] = f"breakout:{e}"
        await asyncio.sleep(0.3)

    # B. Phân loại hỗ trợ / rủi ro & chấm điểm
    scores: List[Dict[str, Any]] = []
    for sym in tickers:
        sc = score_symbol(sym, PIVOTS.get(sym))
        scores.append(sc)
        st = sc["state"]
        line = f"{sym} @ {sc.get('price')} | {st} | {sc['reason']}"
        if st == "support": res["support"].append(line)
        if st == "risk":    res["risk"].append(line)
        await asyncio.sleep(0.15)

    # C. Đề xuất Thay thế/Bổ sung
    #   - Lấy top trong universe EXTRA (ngoài VN21)
    core_set = set(VN21_CORE)
    extra = [x for x in tickers if x not in core_set]
    sc_map = {s["sym"]: s for s in scores}
    top_extra = sorted([sc_map[s] for s in extra if sc_map[s]["score"] > -1e8], key=lambda x: x["score"], reverse=True)[:5]
    core_rank = sorted([sc_map[s] for s in VN21_CORE if s in sc_map and sc_map[s]["score"]>-1e8], key=lambda x: x["score"])
    suggestions = {"add": [], "replace": []}

    # Bổ sung: lấy ≤3 mã extra có score rất cao (≥ core median + 8)
    if core_rank:
        core_scores = [x["score"] for x in core_rank]
        core_median = np.median(core_scores)
        for cand in top_extra:
            if cand["score"] >= core_median + 8:
                suggestions["add"].append(f"ADD {cand['sym']} (score {cand['score']}) — {cand['reason']}")

        # Thay thế: nếu top_extra > bottom_core + 10 thì đề xuất swap
        bottom_core = core_rank[0]
        best_extra = top_extra[0] if top_extra else None
        if best_extra and best_extra["score"] >= bottom_core["score"] + 10:
            suggestions["replace"].append(
                f"REPLACE {bottom_core['sym']} ⟶ {best_extra['sym']} "
                f"(core {bottom_core['score']} vs extra {best_extra['score']})"
            )

    res["suggest"] = suggestions
    return res

# =========================
# Scheduler
# =========================
def in_windows() -> bool:
    return in_trading_hours()

scheduler = AsyncIOScheduler(timezone=str(TZ))
# chạy mỗi phút trong giờ giao dịch
scheduler.add_job(lambda: asyncio.create_task(scan_all(DEFAULT_TICKERS)),
                  CronTrigger(day_of_week="mon-fri", hour="9-11,13-15", minute="*", second="0"),
                  name="vn21-scan")
scheduler.start()

# =========================
# API
# =========================
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "OK"

@app.get("/config", response_class=JSONResponse)
def cfg():
    return {
        "vn21_core": VN21_CORE,
        "universe": DEFAULT_TICKERS,
        "pivots": PIVOTS,
        "plan": PLAN,
        "interval": SCAN_INTERVAL_SEC
    }

@app.post("/scan", response_class=JSONResponse)
async def scan_now():
    out = await scan_all(DEFAULT_TICKERS)
    # Gửi gọn summary lên Telegram
    if out["breakouts"] or out["suggest"]["add"] or out["suggest"]["replace"]:
        lines = ["📊 VN21 — Tổng hợp:"]
        if out["breakouts"]:
            lines += ["⚡ Breakout:", *[f"• {x}" for x in out["breakouts"][:6]]]
        if out["support"]:
            lines += ["🟦 Về hỗ trợ:", *[f"• {x}" for x in out["support"][:6]]]
        if out["risk"]:
            lines += ["⚠️ Rủi ro:", *[f"• {x}" for x in out["risk"][:4]]]
        if out["suggest"]["add"] or out["suggest"]["replace"]:
            lines += ["🧩 Gợi ý danh mục:"]
            lines += [f"• {x}" for x in out["suggest"]["add"][:3]]
            lines += [f"• {x}" for x in out["suggest"]["replace"][:2]]
        tele_send("\n".join(lines))
    return out

@app.get("/", response_class=PlainTextResponse)
def root():
    return "VN21 Scanner Pro/Full — /healthz /config /scan"
