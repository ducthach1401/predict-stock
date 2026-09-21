#!/usr/bin/env bash
# First-time set-up on an EMPTY database, run inside the app container by deploy.sh: schema, universe, data, features, datasets, baselines, models, first champions, paper record.
# Every step is idempotent, so a run that was interrupted can simply be started again. Training takes a while (Optuna study + walk-forward for SWING and INVEST).
set -euo pipefail
PS="python -m predict_stock"
step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

step "database schema (alembic upgrade head)"; alembic upgrade head
step "universe LARGE50";  $PS universe apply --universe LARGE50 --file data/universe/large50_membership.csv --effective-date "$(date +%F)" --create --name "Top 50 by traded value" --no-backfill
step "market data (backfill + ingest + quality)"; $PS data
step "feature sets and datasets"
$PS features sync
$PS dataset build --feature-set swing:1 --label-spec swing:1
$PS dataset build --feature-set invest:2 --label-spec invest:1
step "baselines (development period)";  $PS backtest run
step "SWING model (walk-forward, held-out year not touched)"; $PS swing run
step "INVEST B1 / B2 models"; $PS invest run
step "first champions";  $PS lifecycle init-champions
step "paper record";     if $PS paper status | grep -q "NOT INITIALISED"; then $PS paper init; else echo "already initialised"; fi
echo; echo "=== bootstrap finished ($(date +%H:%M:%S))"
