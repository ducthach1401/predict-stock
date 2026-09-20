"""Expected holding time from SIMILAR past signals.

"Similar" = the same bucket of the calibrated probability. For each bucket the table keeps the median and the 75th percentile of the
sessions until the triple barrier resolved (``tb_time``: the session of the first touch of the target or the stop, or the horizon when
neither was touched, which is where a time-stop would close the position) and the number of past signals ``n`` it rests on.
It is built from the validation rows of a fold only (their labels were known before the test window), so it is point-in-time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HoldTable:
    edges: list[float]        # upper bounds of all buckets except the last
    median: list[float]
    p75: list[float]
    n: list[int]
    timed_out: list[float]    # share of the bucket's signals that reached the horizon with no barrier touched
    overall: dict

    def bucket(self, proba: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges, float), np.asarray(proba, float), side="right")

    def lookup(self, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = self.bucket(proba)
        return (np.asarray(self.median)[b], np.asarray(self.p75)[b], np.asarray(self.n)[b])

    def to_dict(self) -> dict:
        return {"edges": self.edges, "median": self.median, "p75": self.p75, "n": self.n, "timed_out": self.timed_out, "overall": self.overall}

    @classmethod
    def from_dict(cls, d: dict) -> "HoldTable":
        return cls(**d)


def _stats(hold: np.ndarray, horizon: int) -> tuple[float, float, int, float]:
    return float(np.median(hold)), float(np.percentile(hold, 75)), int(len(hold)), float(np.mean(hold >= horizon))


def fit_hold_table(proba: np.ndarray, hold: np.ndarray, horizon: int, buckets: int = 10, min_n: int = 100) -> HoldTable:
    """Buckets by quantiles of ``proba`` (ties collapse them); a bucket with fewer than ``min_n`` signals is merged into its smaller neighbour."""
    p, h = np.asarray(proba, float), np.asarray(hold, float)
    ok = np.isfinite(p) & np.isfinite(h)
    p, h = p[ok], h[ok]
    if len(p) == 0:
        raise ValueError("no signals to build the holding-time table from")
    edges = sorted(set(float(e) for e in np.quantile(p, np.linspace(0, 1, buckets + 1)[1:-1])))
    while True:
        b = np.searchsorted(np.asarray(edges, float), p, side="right")
        counts = np.bincount(b, minlength=len(edges) + 1)
        small = [i for i, c in enumerate(counts) if c < min_n]
        if not small or len(edges) == 0:
            break
        i = min(small, key=lambda k: counts[k])
        # merge bucket i into a neighbour: drop the edge on the side of the smaller neighbour
        left = counts[i - 1] if i > 0 else np.inf
        right = counts[i + 1] if i < len(edges) else np.inf
        edges.pop(i - 1 if left <= right else i)
    b = np.searchsorted(np.asarray(edges, float), p, side="right")
    rows = [_stats(h[b == i], horizon) for i in range(len(edges) + 1)]
    med, p75, n, to = (list(x) for x in zip(*rows))
    m, q, cnt, t = _stats(h, horizon)
    return HoldTable(edges, med, p75, [int(x) for x in n], to, {"median": m, "p75": q, "n": cnt, "timed_out": t})
