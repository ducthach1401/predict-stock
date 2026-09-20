"""Alembic environment.

The URL comes from the environment (migrator role), never from alembic.ini.
Target the test database with:  alembic -x db=test upgrade head
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from predict_stock.config import db_url
from predict_stock.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url():
    x = context.get_x_argument(as_dictionary=True)
    return db_url("migrator", test=x.get("db") == "test")


def run_migrations_offline() -> None:
    context.configure(
        url=_url().render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
