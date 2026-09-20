# Usage:  make setup   (first time)   ·   make data   (idempotent: safe to run repeatedly)
PY ?= .venv/bin/python
ALEMBIC ?= .venv/bin/alembic
EFFECTIVE_DATE ?= $(shell date +%F)

.PHONY: setup db-up migrate universe data backfill quality features datasets backtest swing invest reco paper backup restore-test crontab test test-live

setup: db-up migrate universe

db-up:
	docker-compose up -d

migrate:
	$(ALEMBIC) upgrade head

# Seed universe (LARGE50). Re-applying the same file is a no-op.
universe:
	$(PY) -m predict_stock universe apply --universe LARGE50 --file data/universe/large50_membership.csv \
		--effective-date $(EFFECTIVE_DATE) --create --name "Top 50 by traded value" --no-backfill

# Everything data-related, idempotent: backfill new members, incremental ingest (+calendar), gap candidates,
# quality checks, persisted findings, docs/DATA_QUALITY.md. Exit code 1 if a step failed or an error is unexplained.
data: migrate
	$(PY) -m predict_stock data

backfill:
	$(PY) -m predict_stock backfill

quality:
	$(PY) -m predict_stock quality run --details 20

# Phase 3: store the feature sets / label specs, then build (or reproduce) the datasets. Identical inputs -> identical
# files: a second run reports REUSED. Each build proves the features do not use data after t (look-ahead audit).
features:
	$(PY) -m predict_stock features sync

datasets: features
	$(PY) -m predict_stock dataset build --feature-set swing:1  --label-spec swing:1
	$(PY) -m predict_stock dataset build --feature-set invest:2 --label-spec invest:1

# Phase 4: baselines on the DEVELOPMENT period (the last 12 months stay held out): before/after costs, cost sensitivity, sub-periods,
# walk-forward folds, noise floor; writes docs/BASELINES.md + charts and stores the results in `experiments`.
backtest: datasets
	$(PY) -m predict_stock backtest run

# Phase 5: SWING model: pre-registration, one Optuna study, walk-forward on the development period, comparison with the baselines over the same window,
# report docs/SWING.md, models + predictions in the database. The held-out period is NOT touched (`swing oos --final` only after a PASS).
swing: datasets
	$(PY) -m predict_stock swing run

# Phase 6: INVEST B1 / B2: pre-registration, one grid, walk-forward, four candidates vs the baselines with bootstrap intervals; docs/INVEST_B1.md, docs/INVEST_B2.md.
# The held-out period is NOT touched (`invest oos --final` only after a PASS).
invest: datasets
	$(PY) -m predict_stock invest run

# Phase 7: today's recommendation cards (BUY / WATCH into `recommendations`) and the backtest of the cards exactly as issued -> docs/RECOMMENDATIONS.md.
reco:
	$(PY) -m predict_stock reco generate
	$(PY) -m predict_stock reco backtest

# Phase 8: the daily paper-trading job (paper mode only), backup with retention, restore test, cron entries (printed, not installed).
paper:
	$(PY) -m predict_stock paper run

backup:
	$(PY) -m predict_stock paper backup --prune

restore-test:
	$(PY) -m predict_stock paper restore-test

crontab:
	$(PY) -m predict_stock paper crontab

test:
	$(PY) -m pytest

test-live:
	$(PY) -m pytest -m live -o addopts=""
