"""Model registry: statuses candidate | shadow | champion | retired, the allowed transitions, one champion per strategy, and the log of every change.

  candidate --start-shadow--> shadow --promote--> champion --(replaced / rollback)--> retired --rollback--> champion
  candidate/shadow --reject--> retired

A challenger can only become champion through `shadow`; the only way back from `retired` to `champion` is a rollback to the champion it replaced."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from predict_stock.db.models import Model, ModelStatusLog
from predict_stock.db.session import session_scope

STATUSES = ("candidate", "shadow", "champion", "retired")
STRATEGIES = ("swing", "invest_b1", "invest_b2")
ALLOWED = {("candidate", "shadow"), ("candidate", "retired"), ("shadow", "champion"), ("shadow", "retired"), ("champion", "retired"), ("retired", "champion")}


class LifecycleError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRef:
    id: int
    name: str
    version: int
    strategy: str | None
    status: str
    artifact_path: str
    sha256: str
    trained_until: date | None


def _ref(m: Model) -> ModelRef:
    return ModelRef(m.id, m.name, m.version, m.strategy, m.status, m.artifact_path, m.artifact_sha256, m.trained_until)


def log_change(session: Session, model_id: int, frm: str | None, to: str, actor: str, reason: str | None, details: dict | None = None) -> None:
    session.add(ModelStatusLog(model_id=model_id, from_status=frm, to_status=to, actor=actor, reason=reason, details=details))


def get(engine: Engine, model_id: int) -> ModelRef:
    with session_scope(engine) as s:
        m = s.get(Model, model_id)
        if m is None:
            raise LifecycleError(f"model #{model_id} does not exist")
        return _ref(m)


def champion(engine: Engine, strategy: str) -> ModelRef | None:
    with session_scope(engine) as s:
        rows = list(s.scalars(select(Model).where(Model.strategy == strategy, Model.status == "champion")))
        if len(rows) > 1:
            raise LifecycleError(f"{strategy}: {len(rows)} champions (must be exactly one): {[r.id for r in rows]}")
        return _ref(rows[0]) if rows else None


def with_status(engine: Engine, strategy: str, status: str) -> list[ModelRef]:
    with session_scope(engine) as s:
        return [_ref(m) for m in s.scalars(select(Model).where(Model.strategy == strategy, Model.status == status).order_by(Model.id))]


def set_status(engine: Engine, model_id: int, to: str, *, actor: str, reason: str, details: dict | None = None) -> ModelRef:
    """One status change with its checks; promoting a shadow model retires the current champion of the strategy in the same transaction."""
    if to not in STATUSES:
        raise LifecycleError(f"unknown status {to!r}")
    with session_scope(engine) as s:
        m = s.get(Model, model_id, with_for_update=True)
        if m is None:
            raise LifecycleError(f"model #{model_id} does not exist")
        first = (m.status, to) == ("candidate", "champion") and m.strategy and s.scalar(select(func.count()).select_from(Model).where(Model.strategy == m.strategy, Model.status == "champion")) == 0
        if (m.status, to) not in ALLOWED and not first:                       # a candidate skips the shadow stage only to become the FIRST champion of a strategy that has none
            raise LifecycleError(f"a {m.status} model cannot become {to}: allowed moves are {sorted(f'{a}->{b}' for a, b in ALLOWED)} (a candidate becomes champion directly only when the strategy has none)")
        if to in ("shadow", "champion") and not m.strategy:
            raise LifecycleError(f"model #{model_id} has no strategy: it cannot serve one")
        if to == "champion":
            for old in s.scalars(select(Model).where(Model.strategy == m.strategy, Model.status == "champion", Model.id != m.id).with_for_update()):
                log_change(s, old.id, "champion", "retired", actor, f"replaced by #{m.id}: {reason}", details)
                old.status = "retired"
        log_change(s, m.id, m.status, to, actor, reason, details)
        m.status = to
        s.flush()
        return _ref(m)


def history(engine: Engine, model_id: int | None = None, strategy: str | None = None) -> list[dict]:
    with session_scope(engine) as s:
        q = select(ModelStatusLog, Model.name, Model.version, Model.strategy).join(Model, Model.id == ModelStatusLog.model_id).order_by(ModelStatusLog.id)
        if model_id is not None:
            q = q.where(ModelStatusLog.model_id == model_id)
        if strategy is not None:
            q = q.where(Model.strategy == strategy)
        return [{"model_id": r.model_id, "model": f"{n} v{v}", "strategy": st, "from": r.from_status, "to": r.to_status, "at": r.changed_at, "actor": r.actor, "reason": r.reason, "details": r.details}
                for r, n, v, st in s.execute(q).all()]


def previous_champion(engine: Engine, strategy: str) -> ModelRef | None:
    """The champion that the current one replaced (the most recent retirement from 'champion' that is not the current champion)."""
    cur = champion(engine, strategy)
    with session_scope(engine) as s:
        q = (select(ModelStatusLog, Model).join(Model, Model.id == ModelStatusLog.model_id)
             .where(Model.strategy == strategy, ModelStatusLog.from_status == "champion", ModelStatusLog.to_status == "retired").order_by(ModelStatusLog.id.desc()))
        for log, m in s.execute(q).all():
            if cur is None or m.id != cur.id:
                return _ref(m)
    return None
