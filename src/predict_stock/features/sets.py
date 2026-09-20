"""Feature sets and label specs are DATA (config/feature_sets.yaml), pinned by (name, version, params).

`sync_definitions` copies them into the ``feature_sets`` / ``label_specs`` tables. A stored version is immutable:
changing what a set contains means a new version. New features never require a schema change."""
from __future__ import annotations

from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from predict_stock.db.models import FeatureSet, LabelSpec
from predict_stock.features import cs as _cs, fundamentals as _fund, labels as _labels, ts as _ts  # noqa: F401  (register plugins)
from predict_stock.features.registry import FeatureSetSpec, LabelSpecSpec, RegistryError, spec_hash


def _split_key(key: str, d: dict) -> tuple[str, int]:
    """``swing:1`` -> ("swing", 1); a plain ``swing`` takes its version from the ``version`` field."""
    name, _, ver = str(key).partition(":")
    version = int(d["version"])
    if ver and int(ver) != version:
        raise RegistryError(f"{key!r}: the key says version {ver} but the entry says version {version}")
    return name, version


def load_definitions(path: str | Path) -> tuple[dict[tuple[str, int], FeatureSetSpec], dict[tuple[str, int], LabelSpecSpec]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fsets, lspecs = {}, {}
    for key, d in (raw.get("feature_sets") or {}).items():
        name, version = _split_key(key, d)
        if (name, version) in fsets:
            raise RegistryError(f"feature set {name}:{version} is defined twice")
        fsets[(name, version)] = FeatureSetSpec.from_dict(name, d)
    for key, d in (raw.get("label_specs") or {}).items():
        name, version = _split_key(key, d)
        if (name, version) in lspecs:
            raise RegistryError(f"label spec {name}:{version} is defined twice")
        lspecs[(name, version)] = LabelSpecSpec.from_dict(name, d)
    for spec in fsets.values():
        spec.instantiate()   # fail early: unknown feature, bad params, clashing columns
    for spec in lspecs.values():
        spec.instantiate()
    return fsets, lspecs


def sync_definitions(session: Session, fsets, lspecs) -> dict[str, int]:
    """Idempotent. Returns counts of rows created. A stored (name, version) whose content differs is an error."""
    created = {"feature_sets": 0, "label_specs": 0}
    for (name, version), spec in fsets.items():
        row = session.scalars(select(FeatureSet).where(FeatureSet.name == name, FeatureSet.version == version)).first()
        body = spec.to_spec()
        if row is None:
            session.add(FeatureSet(name=name, version=version, spec=body, description=spec.description,
                                   code_ref="predict_stock.features"))
            created["feature_sets"] += 1
        elif spec_hash(row.spec) != spec_hash(body):
            raise RegistryError(f"feature set {name} v{version} already exists with different content; bump the version")
    for (name, version), spec in lspecs.items():
        row = session.scalars(select(LabelSpec).where(LabelSpec.name == name, LabelSpec.version == version)).first()
        body = spec.to_spec()
        if row is None:
            session.add(LabelSpec(name=name, version=version, horizon_days=spec.horizon(), spec=body, description=spec.description))
            created["label_specs"] += 1
        elif spec_hash(row.spec) != spec_hash(body):
            raise RegistryError(f"label spec {name} v{version} already exists with different content; bump the version")
    session.flush()
    return created
