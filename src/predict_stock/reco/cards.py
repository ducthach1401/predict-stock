"""The recommendation card: a plain structure (JSON), its Vietnamese text, its consistency checks and its translation into engine orders.

Sections (all mandatory for a BUY / WATCH card, otherwise the card is REJECTED with a reason):
  entry      the price zone [low, high], the order type, the tick-rounded prices inside the next session's +-7% band, when the order is cancelled
  exits      SWING: stop, target 1, target 2, reward:risk; INVEST: bear / base / bull price zones and the thesis-break conditions instead of a hard stop
  holding    expected (median), maximum (time-stop), earliest sell date under T+2, INVEST: horizon and the next review date
  confidence calibrated probability, similar past signals with their n, the score's percentile, and how far the probability can be trusted
  rationale  template text built only from numbers the system holds, always with risks and the conditions that void the signal
  sizing     weight, lots of 100, the loss if the stop is hit, caps applied
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from predict_stock.backtest.engine import SignalItem
from predict_stock.backtest.market import MarketRules

ACTIONS = ("BUY", "WATCH", "NO_TRADE")
DISCLAIMER = "Đây là ước lượng thống kê từ mô hình, không phải cam kết hay lời khuyên đầu tư; chỉ dùng cho nghiên cứu và giao dịch giả lập (paper trading)."


# ---- calendar and number formatting ----------------------------------------------------------------------------------------------------------
def session_after(calendar: pd.DatetimeIndex, d, k: int) -> pd.Timestamp:
    """The ``k``-th trading session after ``d``. Beyond the last known session it continues on weekdays (public holidays are not known in advance)."""
    d = pd.Timestamp(d)
    pos = int(calendar.searchsorted(d, side="right"))            # index of the first session strictly after d
    idx = pos + k - 1
    if idx < len(calendar):
        return calendar[idx]
    extra = idx - len(calendar) + 1
    base = calendar[-1] if len(calendar) and calendar[-1] >= d else d
    return pd.Timestamp(np.busday_offset(base.date(), extra, roll="forward"))


def fmt_price(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.0f}".replace(",", ".")


def fmt_pct(x, d: int = 1, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%").replace(".", ",")


def fmt_num(x, d: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace(".", ",")


def fmt_date(d) -> str:
    return "n/a" if d is None else pd.Timestamp(d).strftime("%d/%m/%Y")


def clean(o):
    """JSON-safe copy (numpy -> python, NaN -> None, dates -> ISO)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, date)):
        return str(o)[:10]
    return o


# ---- the card ---------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Card:
    card_id: str
    symbol: str
    instrument_id: int
    strategy: str                 # SWING | INVEST_B1 | INVEST_B2
    sleeve: str
    as_of: str
    valid_until: str
    action: str                   # BUY | WATCH | NO_TRADE
    model_id: int | None
    universe_id: int | None
    model_name: str
    entry: dict = field(default_factory=dict)
    exits: dict = field(default_factory=dict)
    holding: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    rationale: dict = field(default_factory=dict)
    sizing: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    signal: dict = field(default_factory=dict)
    rejected: str | None = None

    def to_dict(self) -> dict:
        return clean(asdict(self))

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=False)

    # ---- checks ---------------------------------------------------------------------------------------------------------------------------------
    def problems(self, rules: MarketRules | None = None) -> list[str]:
        """What is missing or inconsistent. An empty list means the card has all four mandatory parts (entry, target, rationale, holding time) and its prices are valid."""
        out = []
        e, x, h, r = self.entry, self.exits, self.holding, self.rationale
        if not (e.get("zone_low") and e.get("zone_high") and e.get("order_type")):
            out.append("thiếu giá vào lệnh (vùng, loại lệnh)")
        elif e["zone_low"] > e["zone_high"]:
            out.append("vùng vào lệnh đảo ngược (thấp > cao)")
        if self.strategy == "SWING":
            if not (x.get("target1") and x.get("target2") and x.get("stop")):
                out.append("thiếu target / stop-loss")
            elif not (x["stop"] < e.get("reference", x["target1"]) < x["target1"] <= x["target2"]):
                out.append("thứ tự stop < vào < target1 <= target2 bị vi phạm")
        elif not (x.get("scenarios") and x.get("thesis_break")):
            out.append("thiếu kịch bản bear/base/bull hoặc điều kiện phá vỡ luận điểm")
        if not (h.get("expected_sessions") and h.get("max_sessions") and h.get("earliest_sell")):
            out.append("thiếu thời gian nắm giữ (kỳ vọng / tối đa / T+2)")
        if not (r.get("reasons") and r.get("risks") and r.get("invalidation")):
            out.append("thiếu lý do, rủi ro hoặc điều kiện mất hiệu lực")
        if rules is not None:
            for k in ("zone_low", "zone_high", "trigger"):
                v = e.get(k)
                if v is not None and abs(v / rules.tick(v) - round(v / rules.tick(v))) > 1e-6:
                    out.append(f"{k} không nằm trên bước giá")
            lo, hi = e.get("session_floor"), e.get("session_ceiling")
            if lo is not None and hi is not None:
                for k in ("zone_low", "zone_high", "trigger"):
                    v = e.get(k)
                    if v is not None and not (lo - 1e-9 <= v <= hi + 1e-9):
                        out.append(f"{k} nằm ngoài biên độ phiên kế tiếp [{lo:g}, {hi:g}]")
            for k in ("stop", "target1", "target2"):
                v = x.get(k)
                if v is not None and abs(v / rules.tick(v) - round(v / rules.tick(v))) > 1e-6:
                    out.append(f"{k} không nằm trên bước giá")
        return out

    # ---- engine ---------------------------------------------------------------------------------------------------------------------------------
    def to_signal_item(self, weight: float | None = None, group: str | None = None) -> SignalItem:
        """The order the card asks for, exactly as stated: prices, order type, cancel condition, validity, exits, tag. ``weight`` = share of total equity."""
        e, x, s = self.entry, self.exits, self.sizing
        w = s.get("weight") if weight is None else weight
        kind = e["order_type"]
        kw: dict = dict(order=kind, valid_sessions=e.get("validity_sessions", 1), tag=self.card_id, group=group or self.sleeve)
        if kind == "limit":
            kw["limit_price"] = e["zone_high"]
        elif kind == "stop":
            kw.update(trigger_price=e["trigger"], zone_high=e["zone_high"])
        elif kind == "ato":
            kw.update(zone_low=e["zone_low"], zone_high=e["zone_high"])
        if e.get("cancel_if_open_above") is not None:
            kw["cancel_if_open_above"] = e["cancel_if_open_above"]
        if self.strategy == "SWING":
            kw.update(stop_price=x["stop"], target1_price=x["target1"], target_price=x["target2"], target1_fraction=x.get("t1_fraction", 0.5),
                      breakeven_after_t1=bool(x.get("breakeven_after_t1")), max_hold=self.holding["max_sessions"], only_if_flat=True)
        else:
            kw.update(only_if_flat=False)
        return SignalItem(self.instrument_id, float(w), **kw)

    # ---- Vietnamese text ------------------------------------------------------------------------------------------------------------------------
    def text_vi(self) -> str:
        return render_vi(self)


ACTION_VI = {"BUY": "MUA", "WATCH": "THEO DÕI", "NO_TRADE": "KHÔNG GIAO DỊCH"}
ORDER_VI = {"limit": "lệnh giới hạn (LO) chờ điều chỉnh", "stop": "lệnh breakout (mua khi giá vượt mức kích hoạt)", "ato": "lệnh ATO (mua tại giá mở cửa)"}


def render_vi(c: Card) -> str:
    e, x, h, k, r, s, v = c.entry, c.exits, c.holding, c.confidence, c.rationale, c.sizing, c.validation
    L = [f"THẺ KHUYẾN NGHỊ — {c.symbol} — {c.strategy.replace('_', ' ')} — {ACTION_VI.get(c.action, c.action)}",
         f"Ngày tạo {fmt_date(c.as_of)} · hiệu lực đến hết {fmt_date(c.valid_until)} · mô hình {c.model_name}"
         + (f" (#{c.model_id})" if c.model_id else "") + f" · universe #{c.universe_id}"]
    if v.get("model_status"):
        L.append(f"⚠ {v['model_status']}")
    if c.rejected:
        L += ["", f"KHÔNG PHÁT HÀNH: {c.rejected}"]
        return "\n".join(L)
    L += ["", "1. GIÁ VÀO LỆNH"]
    zone = f"{fmt_price(e['zone_low'])} – {fmt_price(e['zone_high'])}"
    L.append(f"   Vùng vào: {zone} (giá tham chiếu phiên kế tiếp {fmt_price(e.get('reference_close'))}; biên độ {fmt_price(e.get('session_floor'))} – {fmt_price(e.get('session_ceiling'))}).")
    L.append(f"   Loại lệnh: {ORDER_VI.get(e['order_type'], e['order_type'])}" + (f", kích hoạt tại {fmt_price(e['trigger'])}" if e.get("trigger") else "") + ".")
    for cond in e.get("cancel_conditions", []):
        L.append(f"   Hủy nếu: {cond}")
    for t in e.get("tranches", []):
        L.append(f"   Đợt {t['n']}: {fmt_pct(t['share'], 0)} khối lượng — {t['when']}")
    L.append("")
    if c.strategy == "SWING":
        L.append("2. TARGET & CẮT LỖ")
        L.append(f"   Giá vào tham chiếu {fmt_price(x['reference'])} · Stop-loss {fmt_price(x['stop'])} ({fmt_pct((x['stop'] / x['reference'] - 1), 1, True)})")
        L.append(f"   Target 1: {fmt_price(x['target1'])} ({fmt_pct(x['target1'] / x['reference'] - 1, 1, True)}), xác suất chạm ước lượng {fmt_pct(x.get('p_touch_target1'), 0)} — bán {fmt_pct(x['t1_fraction'], 0)} vị thế"
                 + (", dời stop về giá vào" if x.get("breakeven_after_t1") else ""))
        L.append(f"   Target 2: {fmt_price(x['target2'])} ({fmt_pct(x['target2'] / x['reference'] - 1, 1, True)}), xác suất chạm trước stop (đã hiệu chỉnh) {fmt_pct(x.get('p_touch_target2'), 0)}")
        L.append(f"   R:R tới target 2 = {fmt_num(x['rr_target2'])} (tới target 1 = {fmt_num(x['rr_target1'])}); ngưỡng tối thiểu {fmt_num(x['min_rr'])}.")
        for n in x.get("notes", []):
            L.append(f"   Lưu ý: {n}")
    else:
        L.append("2. KỊCH BẢN & ĐIỀU KIỆN PHÁ VỠ LUẬN ĐIỂM (không có stop cứng)")
        sc = x["scenarios"]
        for name, key in (("Bear", "bear"), ("Base", "base"), ("Bull", "bull")):
            L.append(f"   {name} (q{ {'bear': 10, 'base': 50, 'bull': 90}[key] }) sau {h['horizon_sessions']} phiên: {fmt_pct(sc[key]['return'], 1, True)} → khoảng giá {fmt_price(sc[key]['price'])}")
        L.append("   Điều kiện phá vỡ luận điểm (rà soát và cân nhắc thoát khi xảy ra):")
        for t in x["thesis_break"]:
            L.append(f"   - {t}")
        L.append(f"   Mức tham chiếu rủi ro (kịch bản bear, KHÔNG phải lệnh stop): {fmt_price(x.get('bear_reference'))}")
    L += ["", "3. THỜI GIAN NẮM GIỮ"]
    L.append(f"   Kỳ vọng (trung vị) {h['expected_sessions']} phiên" + (f", 75% trường hợp trong {h['p75_sessions']} phiên" if h.get("p75_sessions") else "")
             + f"; tối đa {h['max_sessions']} phiên ({'time-stop' if c.strategy == 'SWING' else 'rà soát lại luận điểm'}); sớm nhất bán được (T+2): {fmt_date(h['earliest_sell'])}.")
    if h.get("next_review"):
        L.append(f"   Ngày rà soát/tái cơ cấu kế tiếp: {fmt_date(h['next_review'])}.")
    L += ["", "4. ĐỘ TIN CẬY"]
    L.append(f"   Xác suất mô hình (đã hiệu chỉnh): {fmt_pct(k.get('p_model'), 0)} · xác suất HIỂN THỊ: {k.get('p_display_text', 'n/a')}")
    L.append(f"   Điểm xếp hạng {fmt_num(k.get('score'), 3)} — hạng {k.get('rank')}/{k.get('n_universe')} (phân vị {fmt_pct(k.get('score_percentile'), 0)} trong rổ hôm nay).")
    sm = k.get("similar") or {}
    if sm and "beat_median_rate" in sm:
        L.append(f"   Các lần chọn top-K trước đây của mô hình (n = {sm['n']}, chồng lấp): lợi suất dương {fmt_pct(sm.get('win_rate'), 0)}, vượt trung vị {fmt_pct(sm.get('beat_median_rate'), 0)}."
                 + (f" ⚠ n nhỏ (< {sm['n_min']}), không đáng tin." if sm.get("small_n") else ""))
    elif sm:
        L.append(f"   Nhóm tín hiệu tương tự trong quá khứ (n = {sm['n']}): thắng {fmt_pct(sm.get('win_rate'), 0)}, chạm target 1 {fmt_pct(sm.get('t1_rate'), 0)}, chạm stop {fmt_pct(sm.get('stop_rate'), 0)}, hết hạn {fmt_pct(sm.get('timeout_rate'), 0)}."
                 + (f" ⚠ n nhỏ (< {sm['n_min']}), kết quả không đáng tin." if sm.get("small_n") else ""))
    L.append(f"   Xếp hạng độ tin cậy hiển thị: {k.get('grade')} — {k.get('grade_reason')}")
    L += ["", "5. LÝ DO"]
    for t in r.get("reasons", []):
        L.append(f"   • {t}")
    L.append("   Rủi ro / lập luận ngược:")
    for t in r.get("risks", []):
        L.append(f"   ▸ {t}")
    L.append("   Tín hiệu mất hiệu lực khi:")
    for t in r.get("invalidation", []):
        L.append(f"   ✗ {t}")
    L += ["", "6. QUẢN TRỊ VỐN"]
    shares = f"{s['shares']:,}".replace(",", ".")
    tr = s.get("tranche")
    L.append(f"   Tỷ trọng {fmt_pct(s['weight'], 2)} vốn (≈ {shares} cổ phiếu = {int(s['shares'] // 100)} lô, vốn tham chiếu {s['capital'] / 1e9:.1f} tỷ)"
             + (f" — đợt {tr['n']}/{tr['of']}, tỷ trọng mục tiêu cuối {fmt_pct(tr['final_weight'], 2)}" if tr else "") + ".")
    if s.get("risk_if_stop") is not None:
        L.append(f"   Nếu chạm stop mất ≈ {fmt_pct(s['risk_pct_capital'], 2)} vốn (mục tiêu rủi ro {fmt_pct(s['risk_target'], 2)}).")
    if s.get("risk_if_bear_pct_capital") is not None:
        L.append(f"   Nếu rơi về kịch bản bear mất ≈ {fmt_pct(s['risk_if_bear_pct_capital'], 2)} vốn (ước lượng thống kê, không có stop cứng).")
    for n in s.get("caps_applied", []):
        L.append(f"   Giới hạn áp dụng: {n}")
    L += ["", DISCLAIMER]
    return "\n".join(L)
