# Usage:  make setup   (first time)   ·   make data   (idempotent: safe to run repeatedly)
PY ?= .venv/bin/python
ALEMBIC ?= .venv/bin/alembic
EFFECTIVE_DATE ?= $(shell date +%F)

.PHONY: setup db-up migrate universe data backfill quality features datasets test test-live

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

test:
	$(PY) -m pytest

test-live:
	$(PY) -m pytest -m live -o addopts=""
