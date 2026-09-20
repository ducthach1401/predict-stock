"""Time-series features. Every ``compute`` receives ONE instrument's bars (rows = sessions it traded) and may use
row t and earlier rows only. Ratios everywhere (prices are vendor back-adjusted, so levels are not comparable
over time); the single level-based column is ``liq_logvalue_*`` and is documented as such.

SWING : ret, rsi, macd, atr, bollinger_pctb, donchian, zscore, gap, volume_spike, rel_strength
INVEST: momentum_skip, volatility, downside_beta, drawdown, sma_ratio, liquidity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_stock.features.registry import RegistryError, TimeSeriesFeature, register_feature


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (EMA with alpha = 1/n); defined after n observations."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(bars: pd.DataFrame) -> pd.Series:
    pc = bars["close"].shift()
    parts = pd.concat([bars["high"] - bars["low"], (bars["high"] - pc).abs(), (bars["low"] - pc).abs()], axis=1)
    return parts.max(axis=1, skipna=True).where(bars["high"].notna() & bars["low"].notna())


def atr(bars: pd.DataFrame, n: int) -> pd.Series:
    return wilder(true_range(bars), n)


def _positive_int(params: dict, key: str) -> None:
    if not isinstance(params[key], int) or params[key] < 1:
        raise RegistryError(f"{key} must be a positive integer, got {params[key]!r}")


def _positive_ints(params: dict, key: str) -> None:
    vals = params[key]
    if not vals or any((not isinstance(x, int)) or x < 1 for x in vals):
        raise RegistryError(f"{key} must be a non-empty list of positive integers, got {vals!r}")


# ======================================================================== SWING ===
@register_feature
class Returns(TimeSeriesFeature):
    name, version, group = "ret", 1, "swing"
    DEFAULT_PARAMS = {"windows": [1, 2, 3, 5, 10]}

    def validate(self): _positive_ints(self.params, "windows")
    def columns(self): return [f"ret_{n}" for n in self.params["windows"]]
    def warmup(self): return max(self.params["windows"]) + 1

    def compute(self, bars, bench):
        c = bars["close"]
        return pd.DataFrame({f"ret_{n}": c / c.shift(n) - 1 for n in self.params["windows"]})


@register_feature
class RSI(TimeSeriesFeature):
    name, version, group = "rsi", 1, "swing"
    DEFAULT_PARAMS = {"window": 14}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): return [f"rsi_{self.params['window']}"]
    def warmup(self): return self.params["window"] + 1

    def compute(self, bars, bench):
        n = self.params["window"]
        d = bars["close"].diff()
        gain, loss = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi = rsi.where(loss != 0, pd.Series(np.where(gain > 0, 100.0, 50.0), index=rsi.index))   # no losses: 100 (50 if flat)
        return pd.DataFrame({f"rsi_{n}": rsi.where(gain.notna() & loss.notna())})


@register_feature
class MACD(TimeSeriesFeature):
    name, version, group = "macd", 1, "swing"
    DEFAULT_PARAMS = {"fast": 12, "slow": 26, "signal": 9}

    def validate(self):
        for k in ("fast", "slow", "signal"):
            _positive_int(self.params, k)
        if self.params["fast"] >= self.params["slow"]:
            raise RegistryError("fast must be shorter than slow")

    def columns(self): return ["macd_pct", "macd_hist_pct"]
    def warmup(self): return self.params["slow"] + self.params["signal"]

    def compute(self, bars, bench):
        p, c = self.params, bars["close"]
        macd = c.ewm(span=p["fast"], adjust=False, min_periods=p["fast"]).mean() - c.ewm(span=p["slow"], adjust=False, min_periods=p["slow"]).mean()
        sig = macd.ewm(span=p["signal"], adjust=False, min_periods=p["signal"]).mean()
        return pd.DataFrame({"macd_pct": macd / c, "macd_hist_pct": (macd - sig) / c})


@register_feature
class ATRPct(TimeSeriesFeature):
    name, version, group = "atr", 1, "swing"
    DEFAULT_PARAMS = {"window": 14}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): return [f"atr_pct_{self.params['window']}"]
    def warmup(self): return self.params["window"] + 1

    def compute(self, bars, bench):
        n = self.params["window"]
        return pd.DataFrame({f"atr_pct_{n}": atr(bars, n) / bars["close"]})


@register_feature
class BollingerPctB(TimeSeriesFeature):
    name, version, group = "bollinger_pctb", 1, "swing"
    DEFAULT_PARAMS = {"window": 20, "k": 2.0}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): return [f"bb_pctb_{self.params['window']}"]
    def warmup(self): return self.params["window"]

    def compute(self, bars, bench):
        n, k, c = self.params["window"], self.params["k"], bars["close"]
        mid, sd = c.rolling(n).mean(), c.rolling(n).std()
        width = (2 * k * sd).replace(0, np.nan)
        return pd.DataFrame({f"bb_pctb_{n}": (c - (mid - k * sd)) / width})


@register_feature
class DonchianBreakout(TimeSeriesFeature):
    """Distance of the close from the PRIOR n-session high / low (row t is excluded from the channel, so a
    positive ``don_hi_dist`` means today's close broke out)."""

    name, version, group = "donchian", 1, "swing"
    DEFAULT_PARAMS = {"window": 20}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): n = self.params["window"]; return [f"don_hi_dist_{n}", f"don_lo_dist_{n}"]
    def warmup(self): return self.params["window"] + 1

    def compute(self, bars, bench):
        n, c = self.params["window"], bars["close"]
        hi = bars["high"].shift(1).rolling(n).max()
        lo = bars["low"].shift(1).rolling(n).min()
        return pd.DataFrame({f"don_hi_dist_{n}": c / hi - 1, f"don_lo_dist_{n}": c / lo - 1})


@register_feature
class ZScore(TimeSeriesFeature):
    name, version, group = "zscore", 1, "swing"
    DEFAULT_PARAMS = {"window": 20}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): return [f"zscore_{self.params['window']}"]
    def warmup(self): return self.params["window"]

    def compute(self, bars, bench):
        n, c = self.params["window"], bars["close"]
        sd = c.rolling(n).std().replace(0, np.nan)
        return pd.DataFrame({f"zscore_{n}": (c - c.rolling(n).mean()) / sd})


@register_feature
class Gap(TimeSeriesFeature):
    """Today's open vs yesterday's close. NaN on bars whose open was declared unreliable (repair rule)."""

    name, version, group = "gap", 1, "swing"
    def columns(self): return ["gap_1"]
    def warmup(self): return 1
    def compute(self, bars, bench): return pd.DataFrame({"gap_1": bars["open"] / bars["close"].shift() - 1})


@register_feature
class VolumeSpike(TimeSeriesFeature):
    """Volume today over the median of the PRIOR n sessions."""

    name, version, group = "volume_spike", 1, "swing"
    DEFAULT_PARAMS = {"window": 20}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): return [f"vol_spike_{self.params['window']}"]
    def warmup(self): return self.params["window"] + 1

    def compute(self, bars, bench):
        n, v = self.params["window"], bars["volume"]
        med = v.shift(1).rolling(n).median().replace(0, np.nan)
        return pd.DataFrame({f"vol_spike_{n}": v / med})


@register_feature
class RelativeStrength(TimeSeriesFeature):
    """n-session return of the stock minus that of the benchmark (NaN where the benchmark has no history)."""

    name, version, group = "rel_strength", 1, "swing"
    DEFAULT_PARAMS = {"windows": [5, 10]}

    def validate(self): _positive_ints(self.params, "windows")
    def columns(self): return [f"rs_{n}" for n in self.params["windows"]]
    def warmup(self): return max(self.params["windows"]) + 1

    def compute(self, bars, bench):
        c = bars["close"]
        out = {}
        for n in self.params["windows"]:
            out[f"rs_{n}"] = (c / c.shift(n) - 1) - (bench / bench.shift(n) - 1) if bench is not None else np.nan
        return pd.DataFrame(out, index=bars.index)


# ======================================================================= INVEST ===
@register_feature
class MomentumSkip(TimeSeriesFeature):
    """Momentum over m months that skips the most recent month: close[t-skip] / close[t-skip-m*days] - 1."""

    name, version, group = "momentum_skip", 1, "invest"
    DEFAULT_PARAMS = {"months": [3, 6, 12], "skip": 21, "days_per_month": 21}

    def validate(self):
        _positive_ints(self.params, "months"); _positive_int(self.params, "skip"); _positive_int(self.params, "days_per_month")

    def columns(self): return [f"mom_{m}m" for m in self.params["months"]]
    def warmup(self): return self.params["skip"] + max(self.params["months"]) * self.params["days_per_month"] + 1

    def compute(self, bars, bench):
        p, c = self.params, bars["close"]
        recent = c.shift(p["skip"])
        return pd.DataFrame({f"mom_{m}m": recent / c.shift(p["skip"] + m * p["days_per_month"]) - 1 for m in p["months"]})


@register_feature
class Volatility(TimeSeriesFeature):
    name, version, group = "volatility", 1, "invest"
    DEFAULT_PARAMS = {"windows": [63, 126], "annualize": 252}

    def validate(self): _positive_ints(self.params, "windows")
    def columns(self): return [f"vol_{n}" for n in self.params["windows"]]
    def warmup(self): return max(self.params["windows"]) + 1

    def compute(self, bars, bench):
        r = bars["close"].pct_change()
        f = np.sqrt(self.params["annualize"])
        return pd.DataFrame({f"vol_{n}": r.rolling(n).std() * f for n in self.params["windows"]})


@register_feature
class DownsideBeta(TimeSeriesFeature):
    """Beta to the benchmark measured only on the benchmark's down days over the last ``window`` sessions
    (cov(r, r_b | r_b < 0) / var(r_b | r_b < 0), from rolling sums). NaN with fewer than ``min_obs`` down days."""

    name, version, group = "downside_beta", 1, "invest"
    DEFAULT_PARAMS = {"window": 126, "min_obs": 20}

    def validate(self): _positive_int(self.params, "window"); _positive_int(self.params, "min_obs")
    def columns(self): return [f"dbeta_{self.params['window']}"]
    def warmup(self): return self.params["window"] + 1

    def compute(self, bars, bench):
        n, k = self.params["window"], self.params["min_obs"]
        if bench is None:
            return pd.DataFrame({f"dbeta_{n}": np.nan}, index=bars.index)
        x, y = bench.pct_change(), bars["close"].pct_change()          # x: benchmark, y: stock
        m = ((x < 0) & x.notna() & y.notna()).astype(float)
        xm, ym = x.where(m > 0, 0.0), y.where(m > 0, 0.0)
        cnt, sx, sy = m.rolling(n).sum(), xm.rolling(n).sum(), ym.rolling(n).sum()
        sxy, sxx = (xm * ym).rolling(n).sum(), (xm * xm).rolling(n).sum()
        cov = sxy / cnt - sx * sy / cnt**2
        var = sxx / cnt - (sx / cnt) ** 2
        beta = (cov / var.where(var > 1e-14)).where(cnt >= k)
        return pd.DataFrame({f"dbeta_{n}": beta})


@register_feature
class Drawdown(TimeSeriesFeature):
    name, version, group = "drawdown", 1, "invest"
    DEFAULT_PARAMS = {"window": 252}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): return [f"dd_{self.params['window']}"]
    def warmup(self): return self.params["window"]

    def compute(self, bars, bench):
        n, c = self.params["window"], bars["close"]
        return pd.DataFrame({f"dd_{n}": c / c.rolling(n).max() - 1})


@register_feature
class SMARatio(TimeSeriesFeature):
    name, version, group = "sma_ratio", 1, "invest"
    DEFAULT_PARAMS = {"window": 200}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): return [f"sma_ratio_{self.params['window']}"]
    def warmup(self): return self.params["window"]

    def compute(self, bars, bench):
        n, c = self.params["window"], bars["close"]
        return pd.DataFrame({f"sma_ratio_{n}": c / c.rolling(n).mean() - 1})


@register_feature
class Liquidity(TimeSeriesFeature):
    """DEPRECATED, kept so that feature sets built with it stay reproducible (use ``liquidity`` v2).

    ``liq_logvalue_n``: log of the median traded value (close x volume) over n sessions. It is LEVEL-based: the close is
    the vendor's back-adjusted price, which embeds corporate actions that happen AFTER the date, and the volume
    adjustment is unknown, so the level at t carries information about the future. The look-ahead audit cannot see
    this (the adjusted series is one download). On the real data this column had by far the largest rank IC of the
    INVEST set (-0.11 vs <0.05 for the others), which is why v2 exists.
    ``liq_zero_share_n``: share of zero-volume sessions in the window."""

    name, version, group = "liquidity", 1, "invest"
    DEFAULT_PARAMS = {"window": 60}

    def validate(self): _positive_int(self.params, "window")
    def columns(self): n = self.params["window"]; return [f"liq_logvalue_{n}", f"liq_zero_share_{n}"]
    def warmup(self): return self.params["window"]

    def compute(self, bars, bench):
        n = self.params["window"]
        value = bars["close"] * bars["volume"]
        med = value.rolling(n).median()
        return pd.DataFrame({f"liq_logvalue_{n}": np.log(med.where(med > 0)),
                             f"liq_zero_share_{n}": (bars["volume"] == 0).astype(float).where(bars["volume"].notna()).rolling(n).mean()})


@register_feature
class LiquidityScaleFree(TimeSeriesFeature):
    """Liquidity without price levels, so it is point-in-time even though prices are back-adjusted.

    ``liq_trend_n_m``: log(median traded value over n sessions / median over m sessions). Both medians carry the same
    adjustment factor unless a corporate action falls inside the window, so the factor cancels: it measures whether
    trading is drying up (< 0) or picking up (> 0). ``liq_zero_share_n``: share of zero-volume sessions in the window.
    What is NOT here is the cross-sectional LEVEL of liquidity (how liquid vs other stocks): that needs unadjusted
    prices or shares outstanding, which the data does not have."""

    name, version, group = "liquidity", 2, "invest"
    DEFAULT_PARAMS = {"short": 60, "long": 252}

    def validate(self):
        _positive_int(self.params, "short"); _positive_int(self.params, "long")
        if self.params["short"] >= self.params["long"]:
            raise RegistryError("short must be shorter than long")

    def columns(self): return [f"liq_trend_{self.params['short']}_{self.params['long']}", f"liq_zero_share_{self.params['short']}"]
    def warmup(self): return self.params["long"]

    def compute(self, bars, bench):
        n, m = self.params["short"], self.params["long"]
        value = bars["close"] * bars["volume"]
        a, b = value.rolling(n).median(), value.rolling(m).median()
        trend = np.log((a / b).where((a > 0) & (b > 0)))
        return pd.DataFrame({f"liq_trend_{n}_{m}": trend,
                             f"liq_zero_share_{n}": (bars["volume"] == 0).astype(float).where(bars["volume"].notna()).rolling(n).mean()})
