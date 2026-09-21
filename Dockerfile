# predict-stock: research + paper-trading (paper mode only; no code here places a real order).
# One image serves everything: `python -m predict_stock <group> <command>`, the scheduler, alembic.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TZ=Asia/Ho_Chi_Minh HOME=/tmp MPLCONFIGDIR=/tmp/mpl

# libgomp: LightGBM; tzdata: Asia/Ho_Chi_Minh; the MySQL 8 client tools (mysqldump / mysql) come from the same image as the server so backups and restores match it
# (the MariaDB client does not accept the dump options the backup job uses).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libncurses6 tzdata ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=mysql:8.0 /usr/bin/mysql /usr/bin/mysqldump /usr/bin/

WORKDIR /app

# dependencies first (cached until pyproject.toml changes), then the source
COPY pyproject.toml requirements.lock ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt && cat /tmp/requirements.txt
# requirements.lock = `pip freeze` of the environment the models were built in: the image installs exactly those versions (pandas / numpy / LightGBM change results)
RUN pip install -c requirements.lock -r /tmp/requirements.txt

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY config ./config
COPY data ./data
COPY docker/bootstrap.sh ./docker/bootstrap.sh
RUN pip install --no-deps -e . && mkdir -p /app/artifacts /app/reports /app/backups /app/logs && chmod -R a+rwX /app/artifacts /app/reports /app/backups /app/logs

# the config and the universe files are bind-mounted by docker-compose so they can be edited without rebuilding
CMD ["python", "-m", "predict_stock", "paper", "scheduler"]
