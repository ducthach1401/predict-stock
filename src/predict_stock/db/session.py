from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from predict_stock.config import Role, db_url


def make_engine(role: Role = "app", *, test: bool = False) -> Engine:
    # UTC everywhere: the server runs with --default-time-zone=+00:00 as well.
    return create_engine(db_url(role, test=test), pool_pre_ping=True, future=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
