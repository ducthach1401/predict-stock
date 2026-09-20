"""Plugin registry for features and labels, with versions.

Adding a feature = adding a class decorated with ``@register_feature`` (a new *version* of an existing name is
a new class); nothing else changes: no schema, no other feature. Feature sets and label specs are DATA
(config/feature_sets.yaml -> the ``feature_sets`` / ``label_specs`` tables) that pin ``(name, version, params)``,
so a set built today keeps meaning the same thing after new versions appear.

Kinds
* ``ts``    per-instrument time series: sees only that instrument's bars (and the benchmark) up to and
            including row t, never a later row. Everything is point-in-time by construction and checked
            by ``features/audit.py``.
* ``cs``    cross-sectional: sees the values of ts columns for the *members of the universe on that date* only.
* ``panel`` external data by date (fundamentals); off unless a provider is configured.

Nothing that identifies a stock (symbol, id, name, exchange) may be a feature column.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pandas as pd

FORBIDDEN_COLUMN_PARTS = ("symbol", "ticker", "instrument", "exchange", "isin", "_id", "sector_code")
RESERVED_COLUMNS = {"trade_date", "instrument_id"}


class RegistryError(ValueError):
    pass


def _check_columns(kind: str, key: tuple, cols: list[str]) -> None:
    for c in cols:
        low = c.lower()
        if c in RESERVED_COLUMNS or low == "id" or any(p in low for p in FORBIDDEN_COLUMN_PARTS):
            raise RegistryError(f"{kind} {key}: column {c!r} would identify an instrument or clash with a key column")
    if len(set(cols)) != len(cols):
        raise RegistryError(f"{kind} {key}: duplicate column names {cols}")


class _Plugin:
    name: ClassVar[str]
    version: ClassVar[int]
    group: ClassVar[str] = "common"  # swing | invest | common
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.DEFAULT_PARAMS)
        if unknown:
            raise RegistryError(f"{self.name} v{self.version}: unknown params {sorted(unknown)}; allowed {sorted(self.DEFAULT_PARAMS)}")
        self.params: dict[str, Any] = {**self.DEFAULT_PARAMS, **params}
        self.validate()

    def validate(self) -> None:  # override to check parameter values
        pass

    @property
    def key(self) -> tuple[str, int]:
        return (self.name, self.version)

    def columns(self) -> list[str]:
        raise NotImplementedError


class Feature(_Plugin):
    kind: ClassVar[Literal["ts", "cs", "panel"]]

    def warmup(self) -> int:
        """Sessions of the instrument's own history needed before the output is fully defined."""
        return 0


class TimeSeriesFeature(Feature):
    kind = "ts"

    def compute(self, bars: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
        """``bars``: one instrument's rows (only sessions it traded), index = date, columns open/high/low/close/volume.
        ``bench``: benchmark ``close`` aligned to the same index (values known at each date), or None.
        Returns a frame on the same index with ``columns()``. Row t may depend on rows <= t only."""
        raise NotImplementedError


class CrossSectionalFeature(Feature):
    kind = "cs"

    def inputs(self) -> list[str]:
        raise NotImplementedError

    def compute(self, inputs: dict[str, pd.DataFrame], mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """``inputs``: wide frames (date x instrument) of ts columns; ``mask``: True where the instrument is a
        universe member on that date. Row t may depend on row t only."""
        raise NotImplementedError


class PanelFeature(Feature):
    kind = "panel"

    def compute(self, dates: pd.DatetimeIndex, instrument_ids: list[int], provider: Any) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


class LabelBuilder(_Plugin):
    """Labels look FORWARD by design; ``horizon()`` states how far (sessions), and the audit checks a label at t
    is unchanged when data beyond t + horizon is removed."""

    def horizon(self) -> int:
        raise NotImplementedError

    def compute(self, panel: Any, mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Wide frames (date x instrument) keyed by column name, including ``*_end`` date columns (datetime64)
        so that later phases can purge/embargo by label end date."""
        raise NotImplementedError


_FEATURES: dict[tuple[str, int], type[Feature]] = {}
_LABELS: dict[tuple[str, int], type[LabelBuilder]] = {}


def register_feature(cls: type[Feature]) -> type[Feature]:
    key = (cls.name, cls.version)
    if key in _FEATURES and _FEATURES[key] is not cls:
        raise RegistryError(f"feature {key} is already registered")
    if getattr(cls, "kind", None) not in ("ts", "cs", "panel"):
        raise RegistryError(f"feature {key}: kind must be ts, cs or panel")
    if not isinstance(cls.version, int) or cls.version < 1:
        raise RegistryError(f"feature {key}: version must be a positive integer")
    _FEATURES[key] = cls
    return cls


def register_label(cls: type[LabelBuilder]) -> type[LabelBuilder]:
    key = (cls.name, cls.version)
    if key in _LABELS and _LABELS[key] is not cls:
        raise RegistryError(f"label {key} is already registered")
    if not isinstance(cls.version, int) or cls.version < 1:
        raise RegistryError(f"label {key}: version must be a positive integer")
    _LABELS[key] = cls
    return cls


def unregister(kind: str, name: str, version: int) -> None:  # for tests that register throw-away plugins
    (_FEATURES if kind == "feature" else _LABELS).pop((name, version), None)


def get_feature(name: str, version: int) -> type[Feature]:
    try:
        return _FEATURES[(name, version)]
    except KeyError:
        have = sorted(v for n, v in _FEATURES if n == name)
        raise RegistryError(f"unknown feature {name!r} v{version}" + (f" (registered versions: {have})" if have else "")) from None


def get_label(name: str, version: int) -> type[LabelBuilder]:
    try:
        return _LABELS[(name, version)]
    except KeyError:
        have = sorted(v for n, v in _LABELS if n == name)
        raise RegistryError(f"unknown label {name!r} v{version}" + (f" (registered versions: {have})" if have else "")) from None


def list_features() -> list[tuple[str, int, str, str]]:
    return sorted((n, v, c.kind, c.group) for (n, v), c in _FEATURES.items())


def list_labels() -> list[tuple[str, int]]:
    return sorted(_LABELS)


# ---- specs: the DATA that pins plugins ---------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def spec_hash(spec: Any) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    version: int
    features: tuple[tuple[str, int, tuple[tuple[str, Any], ...]], ...]  # (feature, version, sorted params)
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "FeatureSetSpec":
        feats = []
        for f in d.get("features") or []:
            if "name" not in f or "version" not in f:
                raise RegistryError(f"feature set {name!r}: every feature needs name and version, got {f}")
            feats.append((f["name"], int(f["version"]), tuple(sorted((f.get("params") or {}).items()))))
        if not feats:
            raise RegistryError(f"feature set {name!r} has no features")
        return cls(name, int(d["version"]), tuple(feats), d.get("description", ""))

    def to_spec(self) -> dict:
        return {"features": [{"name": n, "version": v, "params": dict(p)} for n, v, p in self.features]}

    def instantiate(self) -> list[Feature]:
        objs = [get_feature(n, v)(**dict(p)) for n, v, p in self.features]
        seen: dict[str, str] = {}
        for o in objs:
            _check_columns("feature", o.key, o.columns())
            for c in o.columns():
                if c in seen:
                    raise RegistryError(f"feature set {self.name!r}: column {c!r} produced by both {seen[c]} and {o.name} v{o.version}")
                seen[c] = f"{o.name} v{o.version}"
        produced = set(seen)
        for o in objs:
            if o.kind == "cs":
                missing = [c for c in o.inputs() if c not in produced]
                if missing:
                    raise RegistryError(f"feature set {self.name!r}: {o.name} ranks {missing}, which no feature of the set produces")
        return objs

    def columns(self) -> list[str]:
        return [c for o in self.instantiate() for c in o.columns()]


@dataclass(frozen=True)
class LabelSpecSpec:
    name: str
    version: int
    labels: tuple[tuple[str, int, tuple[tuple[str, Any], ...]], ...]
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "LabelSpecSpec":
        labs = []
        for f in d.get("labels") or []:
            if "name" not in f or "version" not in f:
                raise RegistryError(f"label spec {name!r}: every label needs name and version, got {f}")
            labs.append((f["name"], int(f["version"]), tuple(sorted((f.get("params") or {}).items()))))
        if not labs:
            raise RegistryError(f"label spec {name!r} has no labels")
        return cls(name, int(d["version"]), tuple(labs), d.get("description", ""))

    def to_spec(self) -> dict:
        return {"labels": [{"name": n, "version": v, "params": dict(p)} for n, v, p in self.labels]}

    def instantiate(self) -> list[LabelBuilder]:
        objs = [get_label(n, v)(**dict(p)) for n, v, p in self.labels]
        seen = set()
        for o in objs:
            _check_columns("label", o.key, o.columns())
            for c in o.columns():
                if c in seen:
                    raise RegistryError(f"label spec {self.name!r}: duplicate column {c!r}")
                seen.add(c)
        return objs

    def horizon(self) -> int:
        return max(o.horizon() for o in self.instantiate())

    def columns(self) -> list[str]:
        return [c for o in self.instantiate() for c in o.columns()]
