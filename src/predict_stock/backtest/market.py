"""Vietnam market rules for the simulator. Every number comes from ``config.market`` (principle 8; the defaults are the
brief's and have NOT been re-verified against the current HOSE/HNX rulebook or a broker's fee schedule).

* lot size (100 shares), tick size by price band, daily price band, fee per side, tax on sells, slippage, T+2;
* only long positions exist anywhere in the engine;
* ``ceiling`` / ``floor`` prices from the previous close: the ceiling is rounded DOWN to a tick and the floor UP, so the
  band is never wider than the rule.

PRICE BASIS: the price series are the vendor's back-adjusted ones (Phase 0), because unadjusted prices are not available.
Returns are right; a tick grid or a lot applied to an adjusted price is only approximately what the exchange would
have applied at the time. The exchange per date is unknown for LARGE50 (some names traded on HNX/UPCoM earlier, with
±10% / ±15% bands), so the band is INFERRED from each instrument's own trailing return history (see ``infer_bands``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from predict_stock.config import MarketConfig

_EPS = 1e-9


@dataclass(frozen=True)
class MarketRules:
    lot_size: int
    fee_rate: float
    sell_tax_rate: float
    slippage_rate: float
    settlement_days: int
    tick_lower: tuple[float, ...]      # price lower bounds (VND)
    tick_size: tuple[float, ...]       # tick for each bound
    bands: tuple[float, ...]           # allowed daily bands, ascending (e.g. 7%, 10%, 15%)
    lock_tolerance: float = 0.001      # a price within this fraction of the ceiling/floor counts as AT it

    @classmethod
    def from_config(cls, m: MarketConfig) -> "MarketRules":
        ticks = sorted(m.tick_size_vnd)
        return cls(m.lot_size, m.fee_rate, m.sell_tax_rate, m.slippage_rate, m.settlement_days,
                   tuple(float(t[0]) for t in ticks), tuple(float(t[1]) for t in ticks), tuple(sorted(m.price_limit_pct.values())))

    def with_costs(self, fee: float | None = None, tax: float | None = None, slippage: float | None = None) -> "MarketRules":
        from dataclasses import replace
        return replace(self, fee_rate=self.fee_rate if fee is None else fee, sell_tax_rate=self.sell_tax_rate if tax is None else tax,
                       slippage_rate=self.slippage_rate if slippage is None else slippage)

    def scaled_costs(self, k: float) -> "MarketRules":
        """All trading costs multiplied by ``k`` (0 = frictionless), everything else unchanged."""
        return self.with_costs(self.fee_rate * k, self.sell_tax_rate * k, self.slippage_rate * k)

    # ---- ticks ---------------------------------------------------------------------------------
    def tick(self, price):
        """Tick size at ``price`` (scalar or array)."""
        p = np.asarray(price, dtype=float)
        idx = np.searchsorted(np.asarray(self.tick_lower), p, side="right") - 1
        out = np.asarray(self.tick_size)[np.clip(idx, 0, len(self.tick_size) - 1)]
        return float(out) if np.ndim(price) == 0 else out

    def round_down(self, price):
        t = self.tick(price)
        return np.floor(np.asarray(price, dtype=float) / t + _EPS) * t if np.ndim(price) else float(np.floor(price / t + _EPS) * t)

    def round_up(self, price):
        t = self.tick(price)
        return np.ceil(np.asarray(price, dtype=float) / t - _EPS) * t if np.ndim(price) else float(np.ceil(price / t - _EPS) * t)

    # ---- band ----------------------------------------------------------------------------------
    def ceiling(self, prev_close, band):
        return self.round_down(np.asarray(prev_close, dtype=float) * (1 + np.asarray(band)))

    def floor(self, prev_close, band):
        return self.round_up(np.asarray(prev_close, dtype=float) * (1 - np.asarray(band)))

    # ---- costs -----------------------------------------------------------------------------------
    def buy_price(self, ref: float) -> float:
        """Adverse slippage on a market buy, rounded UP to the tick."""
        return self.round_up(ref * (1 + self.slippage_rate))

    def sell_price(self, ref: float) -> float:
        """Adverse slippage on a market sell, rounded DOWN to the tick."""
        return self.round_down(ref * (1 - self.slippage_rate))

    def buy_fee(self, value: float) -> float:
        return value * self.fee_rate

    def sell_costs(self, value: float) -> tuple[float, float]:
        """(fee, tax) on a sell of ``value``."""
        return value * self.fee_rate, value * self.sell_tax_rate

    def lots(self, qty: float) -> int:
        """Whole lots contained in ``qty`` shares, as a share count."""
        return int(qty // self.lot_size) * self.lot_size

    def buy_quantity(self, budget: float, price: float) -> int:
        """Largest whole-lot quantity whose cost (price x qty x (1 + fee)) fits ``budget``."""
        if price <= 0 or budget <= 0:
            return 0
        return self.lots(budget / (price * (1 + self.fee_rate)) + _EPS)


def infer_bands(close: pd.DataFrame, rules: MarketRules, window: int = 252, exchange_band: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily price band per (date, instrument), known at the START of each date (it uses returns up to the previous close).

    Where the exchange is known (``exchange_band`` not NaN) that band is used. Otherwise the band is the smallest allowed
    band (7/10/15%) that contains the largest absolute one-session return seen over the trailing ``window`` sessions,
    with a 0.5 percentage point tolerance for adjusted-price rounding; the lowest allowed band when there is no history.
    Being trailing, it lags a change of exchange by up to ``window`` sessions (conservative when moving to a wider band
    is not yet visible, harmless when moving to a narrower one)."""
    ret = close.pct_change(fill_method=None).abs()
    biggest = ret.rolling(window, min_periods=1).max().shift(1)          # information up to yesterday's close
    bands = np.asarray(rules.bands)
    idx = np.searchsorted(bands, biggest.to_numpy(float) - 0.005, side="left")
    idx = np.where(np.isnan(biggest.to_numpy(float)), 0, np.clip(idx, 0, len(bands) - 1))
    out = pd.DataFrame(bands[idx], index=close.index, columns=close.columns)
    if exchange_band is not None:
        out = exchange_band.reindex_like(out).combine_first(out)
    return out
