"""Card builders: SWING and INVEST. The rules (entry style, stop / target multiples, resistance cap, R:R gate, sizing) are the configured ones; every number in a
card is computed from data the system holds at the as-of close. A card that cannot satisfy a rule is NOT issued: it comes back as NO_TRADE with the reason."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predict_stock.backtest.market import MarketRules
from predict_stock.config import InvestPreset, RecoConfig
from predict_stock.reco.cards import (
    DISCLAIMER, Card, fmt_date, fmt_num, fmt_pct, fmt_price, session_after,
)

# ---- labels (Vietnamese) for the model's features; values are formatted by kind -----------------------------------------------------------------
FEATURE_VI = {
    "ret_1": ("lợi suất 1 phiên", "pct"), "ret_2": ("lợi suất 2 phiên", "pct"), "ret_3": ("lợi suất 3 phiên", "pct"), "ret_5": ("lợi suất 5 phiên", "pct"), "ret_10": ("lợi suất 10 phiên", "pct"),
    "rsi_14": ("RSI(14)", "num"), "macd_pct": ("MACD (% giá)", "pct2"), "macd_hist_pct": ("histogram MACD (% giá)", "pct2"), "atr_pct_14": ("ATR(14) so với giá", "pct"),
    "bb_pctb_20": ("vị trí trong dải Bollinger(20)", "num"), "don_hi_dist_20": ("khoảng cách tới đỉnh 20 phiên", "pct"), "don_lo_dist_20": ("khoảng cách tới đáy 20 phiên", "pct"),
    "zscore_20": ("z-score giá 20 phiên", "num"), "gap_1": ("khoảng trống mở cửa", "pct"), "vol_spike_20": ("khối lượng / trung bình 20 phiên", "x"),
    "rs_5": ("sức mạnh tương đối 5 phiên so với chỉ số", "pct"), "rs_10": ("sức mạnh tương đối 10 phiên so với chỉ số", "pct"),
    "ret_5_csrank": ("thứ hạng lợi suất 5 phiên trong rổ", "rank"), "ret_10_csrank": ("thứ hạng lợi suất 10 phiên trong rổ", "rank"), "rsi_14_csrank": ("thứ hạng RSI trong rổ", "rank"),
    "zscore_20_csrank": ("thứ hạng z-score trong rổ", "rank"), "vol_spike_20_csrank": ("thứ hạng khối lượng trong rổ", "rank"), "rs_5_csrank": ("thứ hạng sức mạnh tương đối 5 phiên", "rank"),
    "mom_3m": ("động lượng 3 tháng", "pct"), "mom_6m": ("động lượng 6 tháng (bỏ 1 tháng cuối)", "pct"), "mom_12m": ("động lượng 12 tháng (bỏ 1 tháng cuối)", "pct"),
    "vol_63": ("biến động 63 phiên (năm hóa)", "pct"), "vol_126": ("biến động 126 phiên (năm hóa)", "pct"), "dbeta_126": ("beta giảm giá 126 phiên", "num"),
    "dd_252": ("mức sụt giảm từ đỉnh 252 phiên", "pct"), "sma_ratio_200": ("giá so với SMA200", "pct"), "liq_trend_60_252": ("xu hướng thanh khoản 60/252 phiên", "pct"),
    "mom_6m_csrank": ("thứ hạng động lượng 6 tháng trong rổ", "rank"), "mom_12m_csrank": ("thứ hạng động lượng 12 tháng trong rổ", "rank"),
    "vol_126_csrank": ("thứ hạng biến động (thấp = tốt)", "rank"), "dd_252_csrank": ("thứ hạng mức sụt giảm (nông = tốt)", "rank"), "sma_ratio_200_csrank": ("thứ hạng giá so với SMA200", "rank"),
    "liq_trend_60_252_csrank": ("thứ hạng xu hướng thanh khoản", "rank"),
}


def fmt_feature(name: str, value) -> str:
    label, kind = FEATURE_VI.get(name.lstrip("-"), (name, "num"))
    if value is None or not np.isfinite(value):
        return f"{label}: n/a"
    txt = {"pct": fmt_pct(value, 1), "pct2": fmt_pct(value, 2), "num": fmt_num(value, 2), "x": f"{fmt_num(value, 1)} lần", "rank": fmt_num(value, 2)}[kind]
    return f"{label} = {txt}"


def label_of(name: str) -> str:
    return FEATURE_VI.get(name.lstrip("-"), (name, ""))[0]


# ---- inputs -------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    """The market context of one name at the as-of close (all from data known at that close)."""
    symbol: str
    instrument_id: int
    as_of: pd.Timestamp
    calendar: pd.DatetimeIndex
    close: float
    atr: float                      # absolute ATR(14)
    floor: float                    # next session's band, from the reference (previous close)
    ceiling: float
    hi20: float | None = None
    hi60: float | None = None
    hi252: float | None = None
    sma50: float | None = None
    sma200: float | None = None
    rsi14: float | None = None
    vol_spike: float | None = None
    rs5: float | None = None
    rs10: float | None = None
    regime_off: bool | None = None  # index below its SMA200
    feats: dict = field(default_factory=dict)


@dataclass
class Pred:
    score: float
    rank: int
    n_universe: int
    model_id: int | None
    model_name: str
    universe_id: int | None
    proba_raw: float | None = None
    proba: float | None = None      # calibrated
    q10: float | None = None
    q50: float | None = None
    q90: float | None = None
    hold_median: float | None = None
    hold_p75: float | None = None
    hold_n: int | None = None
    contributions: list = field(default_factory=list)      # [[name, value, contribution], ...] sorted by |contribution|


def nearest(rules: MarketRules, x: float) -> float:
    t = rules.tick(x)
    return float(round(x / t) * t)


def reject(base: dict, why: str) -> Card:
    return Card(**base, action="NO_TRADE", rejected=why)


def _base(ctx: Ctx, strategy: str, sleeve: str, pred: Pred, cfg: RecoConfig, validity: int, verdict: dict | None) -> dict:
    return dict(card_id=f"{strategy}:{ctx.symbol}:{ctx.as_of.date()}", symbol=ctx.symbol, instrument_id=ctx.instrument_id, strategy=strategy, sleeve=sleeve,
                as_of=str(ctx.as_of.date()), valid_until=str(session_after(ctx.calendar, ctx.as_of, validity).date()), model_id=pred.model_id, universe_id=pred.universe_id,
                model_name=pred.model_name, validation=validation_block(verdict), signal={"score": pred.score, "rank": pred.rank, "n_universe": pred.n_universe})


def validation_block(verdict: dict | None) -> dict:
    """What the pre-registered validation said about the model behind the card. A card never hides it."""
    if not verdict:
        return {"paper_trading_only": True, "model_status": "Mô hình chưa có kết quả kiểm chứng trong hệ thống.", "passed": None, "disclaimer": DISCLAIMER}
    failed = [c for c in verdict.get("criteria", []) if not c.get("ok")]
    txt = ("Mô hình đã đạt các tiêu chí kiểm chứng tiền đăng ký trên walk-forward." if verdict.get("passed") else
           f"Mô hình CHƯA vượt baseline sau chi phí trong kiểm chứng walk-forward tiền đăng ký ({len(failed)}/{len(verdict.get('criteria', []))} tiêu chí không đạt): "
           "khuyến nghị chỉ mang tính nghiên cứu / giao dịch giả lập.")
    return {"paper_trading_only": True, "model_status": txt, "passed": bool(verdict.get("passed")), "failed_criteria": [c.get("name") for c in failed], "disclaimer": DISCLAIMER}


# ---- confidence ----------------------------------------------------------------------------------------------------------------------------------------
def grade_evidence(evidence: dict | None, cfg: RecoConfig) -> tuple[str, str]:
    """How much the displayed probability can be trusted, from PAST out-of-sample predictions only."""
    if not evidence or (evidence.get("n") or 0) < cfg.evidence_min_rows:
        return "CHƯA ĐỦ BẰNG CHỨNG", f"chưa có đủ dự đoán ngoài mẫu trong quá khứ để kiểm chứng xác suất (cần ≥ {cfg.evidence_min_rows} dòng)"
    auc, gap = evidence.get("auc"), evidence.get("calibration_gap")
    if auc is None or not np.isfinite(auc) or auc < cfg.evidence_auc_ok:
        return "THẤP", f"AUC ngoài mẫu trong quá khứ chỉ {fmt_num(auc, 2)} (< {fmt_num(cfg.evidence_auc_ok, 2)}): xác suất chưa phân biệt được tín hiệu tốt và xấu"
    if gap is not None and abs(gap) > 0.05:
        return "TRUNG BÌNH", f"AUC {fmt_num(auc, 2)} nhưng xác suất công bố lệch {fmt_pct(gap, 1, True)} so với thực tế trong quá khứ"
    return "KHÁ", f"AUC {fmt_num(auc, 2)} và xác suất công bố bám sát thực tế trong quá khứ (lệch {fmt_pct(gap, 1, True)})"


def confidence_block(pred: Pred, similar: dict | None, evidence: dict | None, cfg: RecoConfig, need_probability: bool = True) -> dict:
    grade, why = grade_evidence(evidence, cfg)
    if not need_probability:
        grade = "KHÔNG ÁP DỤNG"
        why = ("INVEST không công bố xác suất chạm mức giá; tham chiếu duy nhất là các lần chọn top-K trước đây của mô hình" +
               (f" (n = {similar['n']}: lợi suất dương {fmt_pct(similar.get('win_rate'), 0)}, vượt trung vị {fmt_pct(similar.get('beat_median_rate'), 0)}, lợi suất trung bình {fmt_pct(similar.get('mean_return'), 1, True)}; "
                "các quan sát chồng lấp nhiều nên số trường hợp độc lập ít hơn n rất nhiều)" if similar and similar.get("n") else "; chưa có lần chọn nào đã có kết quả thực tế"))
    sm = None
    if similar and similar.get("n"):
        sm = {**similar, "n_min": cfg.similar_min_n, "small_n": similar["n"] < cfg.similar_min_n}
    p_disp, kind = None, "none"
    if need_probability and pred.proba is not None:
        if grade in ("KHÁ", "TRUNG BÌNH"):
            p_disp, kind = pred.proba, "model"
        elif sm and not sm["small_n"]:
            p_disp, kind = sm.get("win_rate"), "historical"
        else:
            p_disp, kind = (sm or {}).get("win_rate"), "historical_low_n" if sm else "none"
    pct = 1.0 - (pred.rank - 1) / max(pred.n_universe - 1, 1)
    text = {"model": f"{fmt_pct(p_disp, 0)} (mô hình đã hiệu chỉnh)", "historical": f"{fmt_pct(p_disp, 0)} (tỷ lệ thắng lịch sử của nhóm tương tự, n = {sm['n'] if sm else 0}; xác suất mô hình không được dùng làm con số chính)",
            "historical_low_n": f"{fmt_pct(p_disp, 0)} (tỷ lệ lịch sử, n nhỏ — chỉ để tham khảo)", "none": "không hiển thị con số xác suất"}[kind]
    if not need_probability:
        text = "không áp dụng (INVEST không dùng xác suất chạm target)"
    return {"p_model": pred.proba, "p_raw": pred.proba_raw, "p_display": p_disp, "display_kind": kind, "p_display_text": text, "score": pred.score, "rank": pred.rank,
            "n_universe": pred.n_universe, "score_percentile": pct, "similar": sm, "grade": grade, "grade_reason": why, "evidence": evidence}


# ---- SWING -----------------------------------------------------------------------------------------------------------------------------------------------
def swing_levels(ctx: Ctx, cfg: RecoConfig, rules: MarketRules) -> dict | str:
    """Entry zone, reference entry, stop, targets from the configured rules. Returns a dict, or the reason it cannot be issued."""
    c = cfg.swing
    close, atr = ctx.close, ctx.atr
    style = c.entry_style
    if style == "auto":
        style = "breakout" if (ctx.hi20 and close >= ctx.hi20 * (1 - c.breakout_within_pct)) else "pullback"
    trigger = None
    if style == "pullback":
        lo, hi = rules.round_down(close - c.pullback_atr[0] * atr), rules.round_down(close - c.pullback_atr[1] * atr)
        order, cancel = "limit", None
    elif style == "breakout":
        if not ctx.hi20:
            return "thiếu đỉnh 20 phiên để đặt lệnh breakout"
        trigger = rules.round_up(ctx.hi20 + c.breakout_buffer_atr * atr)
        lo, hi = trigger, max(trigger, rules.round_down(trigger + c.breakout_zone_atr * atr))
        order, cancel = "stop", hi
    elif style == "ato":
        lo, hi = rules.round_down(close * (1 + c.ato_zone_pct[0])), rules.round_down(close * (1 + c.ato_zone_pct[1]))
        order, cancel = "ato", None
    else:
        return f"kiểu vào lệnh không hợp lệ: {style}"
    lo, hi = max(lo, ctx.floor), min(hi, ctx.ceiling)
    if trigger is not None and trigger > ctx.ceiling:
        return f"mức kích hoạt breakout {fmt_price(trigger)} vượt giá trần phiên kế tiếp {fmt_price(ctx.ceiling)}"
    if lo > hi or hi <= ctx.floor:
        return f"vùng vào lệnh nằm ngoài biên độ phiên kế tiếp [{fmt_price(ctx.floor)}, {fmt_price(ctx.ceiling)}]"
    ref = nearest(rules, (lo + hi) / 2)
    ref = min(max(ref, lo), hi)
    stop = rules.round_down(ref - c.stop_atr * atr)
    t1, t2 = rules.round_up(ref + c.target1_atr * atr), rules.round_up(ref + c.target2_atr * atr)
    notes, capped = [], False
    if stop <= 0 or stop >= ref:
        return "mức stop-loss không hợp lệ"
    if c.use_resistance and ctx.hi60 and ctx.hi60 > ref:
        cap = rules.round_down(ctx.hi60 - rules.tick(ctx.hi60))
        if cap <= ref + rules.tick(ref):
            return f"vùng vào sát kháng cự gần nhất {fmt_price(ctx.hi60)} (đỉnh {c.resistance_lookback} phiên): không còn dư địa tăng"
        if t2 > cap:
            t2, capped = cap, True
            notes.append(f"target 2 bị kéo xuống ngay dưới kháng cự {fmt_price(ctx.hi60)} (đỉnh {c.resistance_lookback} phiên); xác suất mô hình áp dụng cho mức {fmt_num(c.target2_atr, 1)} ATR, không phải mức này")
        t1 = min(t1, cap)
    if t1 <= ref:
        return "target 1 không nằm trên giá vào"
    t1 = min(t1, t2)
    risk = ref - stop
    rr2, rr1 = (t2 - ref) / risk, (t1 - ref) / risk
    return {"style": style, "order": order, "trigger": trigger, "zone_low": lo, "zone_high": hi, "cancel": cancel, "reference": ref, "stop": stop, "target1": t1, "target2": t2,
            "rr1": rr1, "rr2": rr2, "capped": capped, "notes": notes}


def swing_sizing(lv: dict, cfg: RecoConfig, rules: MarketRules, name_room: float | None = None) -> dict | str:
    """Weight from the risk per trade: the loss if the stop is hit is ``risk_per_trade`` of capital, then capped per name (and by what is left of the
    name's total cap across sleeves), rounded down to whole lots of 100."""
    c, cap_total = cfg.swing, cfg.portfolio.capital
    stop_pct = (lv["reference"] - lv["stop"]) / lv["reference"]
    w_risk = c.risk_per_trade / stop_pct
    w, caps = w_risk, []
    if w > c.max_weight:
        w, _ = c.max_weight, caps.append(f"trần {fmt_pct(c.max_weight, 1)} vốn cho mỗi mã (theo rủi ro sẽ là {fmt_pct(w_risk, 1)})")
    if name_room is not None and w > name_room:
        w = max(name_room, 0.0)
        caps.append(f"trần {fmt_pct(cfg.portfolio.max_weight_name, 0)} tổng tỷ trọng một mã trên mọi sleeve (còn {fmt_pct(name_room, 1)})")
    price = lv["zone_high"]
    shares = rules.lots(w * cap_total / price)
    if shares < rules.lot_size:
        return "vốn phân bổ không đủ một lô 100 cổ phiếu"
    weight = shares * price / cap_total
    risk = shares * (lv["reference"] - lv["stop"])
    return {"weight": weight, "shares": int(shares), "lots": int(shares // rules.lot_size), "capital": cap_total, "value": shares * price, "risk_if_stop": risk,
            "risk_pct_capital": risk / cap_total, "risk_target": c.risk_per_trade, "caps_applied": caps, "sizing_price": price}


def swing_texts(ctx: Ctx, pred: Pred, lv: dict, conf: dict, cfg: RecoConfig, verdict: dict | None, rules: MarketRules) -> dict:
    """Reasons, risks and invalidation conditions, from the numbers the system holds (nothing else)."""
    reasons: list[str] = []
    top = [c for c in pred.contributions if c[2] is not None][:4]
    if top:
        reasons.append(f"Mô hình xếp {ctx.symbol} hạng {pred.rank}/{pred.n_universe} hôm nay (điểm {fmt_num(pred.score, 3)}). Các yếu tố đóng góp lớn nhất (TreeSHAP): "
                       + "; ".join(f"{fmt_feature(n, v)} → {'nâng' if c > 0 else 'giảm'} điểm {fmt_num(abs(c), 3)}" for n, v, c in top) + ".")
    ctxt = []
    if ctx.sma200 and ctx.close:
        ctxt.append(f"giá {'trên' if ctx.close >= ctx.sma200 else 'dưới'} SMA200 ({fmt_price(ctx.sma200)}) {fmt_pct(abs(ctx.close / ctx.sma200 - 1), 1)}")
    if ctx.sma50 and ctx.close:
        ctxt.append(f"{'trên' if ctx.close >= ctx.sma50 else 'dưới'} SMA50 ({fmt_price(ctx.sma50)})")
    if ctx.rsi14 is not None and np.isfinite(ctx.rsi14):
        ctxt.append(f"RSI(14) = {fmt_num(ctx.rsi14, 0)}")
    if ctx.vol_spike is not None and np.isfinite(ctx.vol_spike):
        ctxt.append(f"khối lượng {fmt_num(ctx.vol_spike, 1)} lần trung bình 20 phiên")
    if ctx.rs5 is not None and np.isfinite(ctx.rs5):
        ctxt.append(f"sức mạnh tương đối 5 phiên so với chỉ số {fmt_pct(ctx.rs5, 1, True)}" + (f", 10 phiên {fmt_pct(ctx.rs10, 1, True)}" if ctx.rs10 is not None and np.isfinite(ctx.rs10) else ""))
    if ctx.regime_off is not None:
        ctxt.append("chỉ số tham chiếu " + ("đang DƯỚI SMA200 (thị trường yếu)" if ctx.regime_off else "đang trên SMA200"))
    if ctxt:
        reasons.append("Bối cảnh: " + "; ".join(ctxt) + ".")
    reasons.append(f"Cách vào: {lv['style']} — vùng {fmt_price(lv['zone_low'])}–{fmt_price(lv['zone_high'])}; stop {fmt_price(lv['stop'])} = {fmt_num(cfg.swing.stop_atr, 1)} ATR dưới giá vào; "
                   f"ATR(14) = {fmt_price(ctx.atr)} ({fmt_pct(ctx.atr / ctx.close, 1)} giá).")
    risks: list[str] = []
    grade = conf["grade"]
    if grade in ("THẤP", "CHƯA ĐỦ BẰNG CHỨNG"):
        risks.append(f"Xác suất mô hình chưa đáng tin ({conf['grade_reason']}); dùng tỷ lệ lịch sử của nhóm tương tự làm tham chiếu.")
    sm = conf.get("similar")
    if sm and sm.get("small_n"):
        risks.append(f"Nhóm tín hiệu tương tự chỉ có n = {sm['n']} (< {sm['n_min']}): tỷ lệ thắng lịch sử không đáng tin.")
    if verdict and not verdict.get("passed"):
        risks.append("Mô hình chưa vượt các baseline (giữ đều cả rổ, momentum) sau chi phí trong kiểm chứng walk-forward; một tín hiệu đơn lẻ có thể chỉ là nhiễu.")
    if ctx.rsi14 is not None and np.isfinite(ctx.rsi14) and ctx.rsi14 >= 70:
        risks.append(f"RSI(14) = {fmt_num(ctx.rsi14, 0)} ở vùng quá mua: dễ có nhịp điều chỉnh trước khi đi tiếp.")
    if ctx.regime_off:
        risks.append("Chỉ số tham chiếu dưới SMA200: xác suất thất bại của các lệnh long ngắn hạn thường cao hơn trong thị trường yếu.")
    if lv.get("capped"):
        risks.append("Target 2 bị giới hạn bởi kháng cự gần: dư địa lợi nhuận thấp hơn so với 2 ATR mà mô hình được huấn luyện.")
    if ctx.atr / ctx.close >= 0.04:
        risks.append(f"ATR = {fmt_pct(ctx.atr / ctx.close, 1)} giá (biến động lớn): stop 1 ATR dễ bị quét.")
    risks.append(f"T+{rules.settlement_days}: cổ phiếu mua hôm nay không bán được trong {rules.settlement_days} phiên đầu, kể cả khi giá xuyên stop — rủi ro gap thực tế lớn hơn mức stop đã nêu.")
    inval = [f"Đóng cửa dưới stop-loss {fmt_price(lv['stop'])} (hoặc thủng stop trong phiên khi đã bán được).",
             f"Sau {cfg.swing.max_hold} phiên chưa chạm target thì thoát theo time-stop.",
             "Sau khi đạt target 1 mà giá quay về giá vào: thoát phần còn lại (stop dời về hòa vốn)." if cfg.swing.breakeven_after_t1 else "Điểm xếp hạng của mã rơi khỏi nhóm dẫn đầu ở phiên rà soát kế tiếp."]
    if lv["order"] == "stop":
        inval.append(f"Hủy lệnh nếu mở cửa cao hơn {fmt_price(lv['zone_high'])} (không đuổi giá) hoặc sau {cfg.swing.validity_sessions} phiên chưa khớp.")
    else:
        inval.append(f"Hủy lệnh sau {cfg.swing.validity_sessions} phiên chưa khớp" + ("." if lv["order"] == "limit" else " hoặc nếu giá mở cửa ngoài vùng vào."))
    return {"reasons": reasons, "risks": risks[:6], "invalidation": inval, "facts": [list(c) for c in pred.contributions[:5]]}


def build_swing_card(ctx: Ctx, pred: Pred, similar: dict | None, evidence: dict | None, verdict: dict | None, cfg: RecoConfig, rules: MarketRules, *,
                     watch: bool = False, name_room: float | None = None) -> Card:
    c = cfg.swing
    base = _base(ctx, "SWING", "swing", pred, cfg, c.validity_sessions, verdict)
    if not (np.isfinite(ctx.close) and ctx.close > 0 and np.isfinite(ctx.atr) and ctx.atr > 0):
        return reject(base, "thiếu giá hoặc ATR hợp lệ")
    lv = swing_levels(ctx, cfg, rules)
    if isinstance(lv, str):
        return reject(base, lv)
    if lv["rr2"] < c.min_rr:
        return reject(base, f"R:R tới target 2 = {fmt_num(lv['rr2'])} thấp hơn ngưỡng {fmt_num(c.min_rr)}")
    sz = swing_sizing(lv, cfg, rules, name_room)
    if isinstance(sz, str):
        return reject(base, sz)
    conf = confidence_block(pred, similar, evidence, cfg)
    sm = conf.get("similar") or {}
    action = "WATCH" if (watch or (cfg.gate_on_verdict and verdict is not None and not verdict.get("passed"))) else "BUY"
    exp = int(min(round(pred.hold_median), c.expected_hold_cap)) if pred.hold_median else None
    holding = {"expected_sessions": exp, "p75_sessions": int(round(pred.hold_p75)) if pred.hold_p75 else None, "similar_n": pred.hold_n, "max_sessions": c.max_hold,
               "earliest_sell": str(session_after(ctx.calendar, ctx.as_of, 1 + rules.settlement_days).date()),
               "note": "sớm nhất bán được nếu khớp ngay phiên kế tiếp; nếu khớp muộn hơn thì cộng thêm"}
    cancel_conditions = [f"hết hiệu lực sau {c.validity_sessions} phiên (đến hết {fmt_date(base['valid_until'])})"]
    if lv["cancel"] is not None:
        cancel_conditions.insert(0, f"mở cửa cao hơn {fmt_price(lv['cancel'])} thì hủy, không đuổi giá")
    elif lv["order"] == "ato":
        cancel_conditions.insert(0, f"giá mở cửa ngoài vùng {fmt_price(lv['zone_low'])}–{fmt_price(lv['zone_high'])} thì không khớp")
    else:
        cancel_conditions.insert(0, "lệnh giới hạn không bao giờ khớp cao hơn giá giới hạn; nếu giá không điều chỉnh về vùng thì không mua")
    entry = {"style": lv["style"], "order_type": lv["order"], "order_type_vi": {"limit": "LO chờ điều chỉnh", "stop": "breakout", "ato": "ATO"}[lv["order"]], "zone_low": lv["zone_low"],
             "zone_high": lv["zone_high"], "trigger": lv["trigger"], "reference": lv["reference"], "reference_close": ctx.close, "session_floor": ctx.floor, "session_ceiling": ctx.ceiling,
             "cancel_if_open_above": lv["cancel"], "cancel_conditions": cancel_conditions, "validity_sessions": c.validity_sessions, "tick": rules.tick(lv["zone_high"])}
    exits = {"reference": lv["reference"], "stop": lv["stop"], "target1": lv["target1"], "target2": lv["target2"], "t1_fraction": c.t1_fraction, "breakeven_after_t1": c.breakeven_after_t1,
             "rr_target1": lv["rr1"], "rr_target2": lv["rr2"], "min_rr": c.min_rr, "p_touch_target2": None if lv["capped"] else conf["p_display"], "p_touch_target2_model": None if lv["capped"] else pred.proba,
             "p_touch_target1": sm.get("t1_rate"), "p_touch_target1_note": "tỷ lệ lịch sử của nhóm tương tự với rào 1 ATR trước 1 ATR trong 10 phiên", "notes": lv["notes"],
             "resistance": ctx.hi60, "atr": ctx.atr}
    card = Card(**base, action=action, entry=entry, exits=exits, holding=holding, confidence=conf, sizing=sz,
                rationale=swing_texts(ctx, pred, lv, conf, cfg, verdict, rules))
    card.signal.update({"close": ctx.close, "atr": ctx.atr, "atr_pct": ctx.atr / ctx.close, "rsi14": ctx.rsi14, "vol_spike": ctx.vol_spike, "hi20": ctx.hi20, "hi60": ctx.hi60,
                        "q10_5d": pred.q10, "q50_5d": pred.q50, "q90_5d": pred.q90, "regime_off": ctx.regime_off})
    return card


# ---- INVEST ----------------------------------------------------------------------------------------------------------------------------------------------
def invest_thesis_break(ctx: Ctx, preset: InvestPreset, rs_floor: float, top_k: int) -> list[str]:
    out = []
    if ctx.sma200:
        out.append(f"Giá đóng cửa dưới SMA200 ({fmt_price(ctx.sma200)}).")
    out.append(f"Sức mạnh tương đối suy giảm: thứ hạng động lượng 6 tháng trong rổ rơi dưới {fmt_num(rs_floor, 2)} (hiện {fmt_num(ctx.feats.get('mom_6m_csrank'), 2)}).")
    if ctx.hi252:
        out.append(f"Sụt giảm quá {fmt_pct(preset.drawdown_break, 0)} từ đỉnh 252 phiên (đỉnh {fmt_price(ctx.hi252)} → dưới {fmt_price(ctx.hi252 * (1 - preset.drawdown_break))}).")
    else:
        out.append(f"Sụt giảm quá {fmt_pct(preset.drawdown_break, 0)} từ đỉnh 252 phiên.")
    out.append(f"Mã rơi khỏi nhóm {top_k} mã đầu bảng xếp hạng ở kỳ tái cơ cấu kế tiếp.")
    return out


def build_invest_card(ctx: Ctx, pred: Pred, preset_key: str, preset: InvestPreset, similar: dict | None, evidence: dict | None, verdict: dict | None, cfg: RecoConfig,
                      rules: MarketRules, *, weight: float, next_review: pd.Timestamp, top_k: int, rs_floor: float, watch: bool = False, name_room: float | None = None,
                      tranche: tuple[int, int, float] | None = None) -> Card:
    c = cfg.invest
    strategy, sleeve = f"INVEST_{preset_key.upper()}", f"invest_{preset_key}"
    base = _base(ctx, strategy, sleeve, pred, cfg, c.validity_sessions, verdict)
    if not (np.isfinite(ctx.close) and ctx.close > 0):
        return reject(base, "thiếu giá hợp lệ")
    if None in (pred.q10, pred.q50, pred.q90):
        return reject(base, "thiếu phân vị kịch bản bear/base/bull")
    lo, hi = max(rules.round_down(ctx.close * (1 + c.entry_zone_pct[0])), ctx.floor), min(rules.round_down(ctx.close * (1 + c.entry_zone_pct[1])), ctx.ceiling)
    if lo > hi or hi <= ctx.floor:
        return reject(base, f"vùng vào lệnh nằm ngoài biên độ phiên kế tiếp [{fmt_price(ctx.floor)}, {fmt_price(ctx.ceiling)}]")
    ref = min(max(nearest(rules, (lo + hi) / 2), lo), hi)
    px = lambda q, up: (rules.round_up if up else rules.round_down)(ref * (1 + q))
    scen = {"bear": {"quantile": 0.10, "return": pred.q10, "price": max(px(pred.q10, False), rules.tick(1.0))}, "base": {"quantile": 0.50, "return": pred.q50, "price": nearest(rules, ref * (1 + pred.q50))},
            "bull": {"quantile": 0.90, "return": pred.q90, "price": px(pred.q90, True)}}
    total_w = weight
    if name_room is not None and total_w > name_room:
        total_w = max(name_room, 0.0)
    w = min(total_w, c.max_weight)
    caps = []
    if w < weight:
        caps.append(f"trần tỷ trọng ({fmt_pct(min(c.max_weight, name_room if name_room is not None else 1), 1)}) thấp hơn tỷ trọng mục tiêu {fmt_pct(weight, 1)}")
    shares = rules.lots(w * cfg.portfolio.capital / hi)
    if shares < rules.lot_size:
        return reject(base, "vốn phân bổ không đủ một lô 100 cổ phiếu")
    conf = confidence_block(pred, similar, evidence, cfg, need_probability=False)
    sm = conf.get("similar") or {}
    n_tr = preset.tranches
    tranches = [{"n": t + 1, "share": 1.0 / n_tr, "when": ("ngay khi khuyến nghị có hiệu lực" if t == 0 else f"{t * preset.tranche_spacing} phiên sau đợt 1, nếu mã vẫn thuộc top-{top_k}")} for t in range(n_tr)]
    thesis = invest_thesis_break(ctx, preset, rs_floor, top_k)
    action = "WATCH" if (watch or (cfg.gate_on_verdict and verdict is not None and not verdict.get("passed"))) else "BUY"
    entry = {"style": "limit", "order_type": "limit", "order_type_vi": "LO chờ điều chỉnh", "zone_low": lo, "zone_high": hi, "trigger": None, "reference": ref, "reference_close": ctx.close,
             "session_floor": ctx.floor, "session_ceiling": ctx.ceiling, "cancel_if_open_above": None, "validity_sessions": c.validity_sessions, "tick": rules.tick(hi), "tranches": tranches,
             "cancel_conditions": [f"hết hiệu lực sau {c.validity_sessions} phiên (đến hết {fmt_date(base['valid_until'])}) nếu chưa khớp", "lệnh giới hạn không bao giờ khớp cao hơn giá giới hạn; không đuổi giá"]}
    exits = {"reference": ref, "scenarios": scen, "thesis_break": thesis, "bear_reference": scen["bear"]["price"], "stop": None,
             "note": "INVEST không dùng stop cứng: rà soát khi một điều kiện phá vỡ luận điểm xảy ra; mức bear chỉ là tham chiếu rủi ro"}
    holding = {"expected_sessions": preset.horizon, "max_sessions": int(round(preset.horizon * c.time_review_multiple)), "horizon_sessions": preset.horizon,
               "earliest_sell": str(session_after(ctx.calendar, ctx.as_of, 1 + rules.settlement_days).date()), "next_review": str(pd.Timestamp(next_review).date()),
               "note": "kỳ vọng = horizon của nhãn; không có ước lượng thời gian nắm giữ riêng cho INVEST"}
    reasons = []
    top = [t for t in pred.contributions if t[2] is not None][:5]
    reasons.append(f"{ctx.symbol} xếp hạng {pred.rank}/{pred.n_universe} theo điểm {fmt_num(pred.score, 3)} ({pred.model_name.split('_')[2] if pred.model_name.count('_') >= 2 else pred.model_name})"
                   + (". Đóng góp lớn nhất: " + "; ".join(f"{fmt_feature(n, v)} ({'+' if cc > 0 else '-'}{fmt_num(abs(cc), 3)})" for n, v, cc in top) if top else "") + ".")
    tx = []
    if ctx.sma200:
        tx.append(f"giá {'trên' if ctx.close >= ctx.sma200 else 'dưới'} SMA200 {fmt_pct(abs(ctx.close / ctx.sma200 - 1), 1)}")
    if ctx.feats.get("dd_252") is not None:
        tx.append(f"đang cách đỉnh 252 phiên {fmt_pct(abs(ctx.feats['dd_252']), 1)}")
    if ctx.feats.get("mom_6m") is not None:
        tx.append(f"động lượng 6 tháng {fmt_pct(ctx.feats['mom_6m'], 1, True)}")
    if ctx.feats.get("vol_126") is not None:
        tx.append(f"biến động 126 phiên {fmt_pct(ctx.feats['vol_126'], 0)}/năm")
    if ctx.regime_off is not None:
        tx.append("chỉ số tham chiếu " + ("DƯỚI SMA200" if ctx.regime_off else "trên SMA200"))
    if tx:
        reasons.append("Bối cảnh: " + "; ".join(tx) + ".")
    reasons.append(f"Kịch bản {preset.horizon} phiên: bear {fmt_pct(pred.q10, 1, True)}, base {fmt_pct(pred.q50, 1, True)}, bull {fmt_pct(pred.q90, 1, True)} (phân vị 10/50/90 của mô hình).")
    risks = ["Kịch bản bear/base/bull là phân vị thống kê, không phải dự báo: trong kiểm chứng ngoài mẫu chúng không tốt hơn một phân vị hằng số."]
    if verdict and not verdict.get("passed"):
        risks.append("Chiến lược INVEST chưa vượt danh mục giữ đều cả rổ sau chi phí trong kiểm chứng walk-forward (khoảng tin cậy Sharpe bao trùm 0).")
    if sm.get("small_n") or not sm:
        risks.append("Chưa có đủ dữ liệu lịch sử độc lập để nói về tỷ lệ thắng của nhóm tín hiệu này (nhãn 63/126 phiên chồng lấp mạnh).")
    if ctx.regime_off:
        risks.append("Chỉ số tham chiếu dưới SMA200: giai đoạn thị trường yếu thường làm mọi cổ phiếu cùng giảm.")
    if len(risks) < 2:
        risks.append(f"T+{rules.settlement_days}: không bán được trong {rules.settlement_days} phiên đầu.")
    sizing = {"weight": shares * hi / cfg.portfolio.capital, "target_weight": weight, "shares": int(shares), "lots": int(shares // rules.lot_size), "capital": cfg.portfolio.capital, "value": shares * hi,
              "risk_if_stop": None, "risk_if_bear": shares * max(ref - scen["bear"]["price"], 0), "risk_pct_capital": None, "caps_applied": caps, "sizing_price": hi}
    sizing["risk_if_bear_pct_capital"] = sizing["risk_if_bear"] / cfg.portfolio.capital
    if tranche:
        sizing["tranche"] = {"n": tranche[0], "of": tranche[1], "final_weight": tranche[2]}
    card = Card(**base, action=action, entry=entry, exits=exits, holding=holding, confidence=conf, sizing=sizing,
                rationale={"reasons": reasons, "risks": risks, "invalidation": thesis + [f"Hủy lệnh nếu sau {c.validity_sessions} phiên chưa khớp."],
                                                                       "facts": [list(t) for t in pred.contributions[:5]]})
    card.signal.update({"close": ctx.close, "q10": pred.q10, "q50": pred.q50, "q90": pred.q90, "regime_off": ctx.regime_off, **{k: v for k, v in ctx.feats.items() if k in ("mom_6m_csrank", "dd_252", "sma_ratio_200")}})
    return card
