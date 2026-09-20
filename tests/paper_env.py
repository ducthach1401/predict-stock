"""A small simulated market for the end-to-end demo of the paper bot: random-walk bars for a set of stocks, a universe snapshot file, final models registered."""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from predict_stock.data.dnse_client import DailyBar


def random_walk(start: date, n: int, base: float, seed: int, vol: float = 0.018) -> list[DailyBar]:
    rng = np.random.default_rng(seed)
    bars, d, px = [], start, base
    while len(bars) < n:
        if d.weekday() < 5:
            r = rng.normal(0.0002, vol)
            o = px * (1 + rng.normal(0, 0.004))
            c = px * (1 + r)
            h, l = max(o, c) * (1 + abs(rng.normal(0, 0.006))), min(o, c) * (1 - abs(rng.normal(0, 0.006)))
            tick = 10.0 if c < 10_000 else 50.0 if c < 50_000 else 100.0
            q = lambda x: Decimal(str(round(round(x / tick) * tick, 2)))
            bars.append(DailyBar(d, q(o), q(max(h, o, c)), q(min(l, o, c)), q(c), int(rng.integers(400_000, 3_000_000))))
            px = float(q(c))
        d += timedelta(days=1)
    return bars


def market(symbols: list[str], start: date, n: int, seed: int = 0) -> dict[str, list[DailyBar]]:
    return {s: random_walk(start, n, base=20_000.0 + 3_000.0 * (k % 7), seed=seed + k) for k, s in enumerate(symbols)}


def write_snapshot(path: Path, symbols: list[str]) -> None:
    path.write_text("symbol\n" + "\n".join(symbols) + "\n")
