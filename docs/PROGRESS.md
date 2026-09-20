# Progress

## Phase 1 — Data layer  ·  status: acceptance met (2026-09-20), **not committed**

Foundation pieces needed by Phase 1 (repo skeleton, docker-compose MySQL, Alembic, config) were
built as part of it; there is no separate Phase 0.

### Acceptance criteria and evidence

| # | Criterion | Result |
|---|---|---|
| 1 | MySQL 8 via docker-compose; `alembic upgrade head` on empty DB; downgrade → upgrade works | `upgrade`, `downgrade base`, `upgrade` all ran; `alembic check` = no drift. Test asserts models == migrations, utf8mb4, InnoDB |
| 2 | Real ingest from DNSE; re-run creates 0 new rows and writes nothing | DEMO universe: 10,855 stock bars + VNINDEX 2,138 + VN30 1,556. Re-run: fetched 91, inserted 0, updated 0, unchanged 91. 0 duplicate PKs. Test compares values, `fetched_at` and `run_id` before/after a re-run |
| 3 | DNSE re-adjustment detected → full re-fetch, no spliced series | Test: history ×0.8 + new bars → symbol flagged, all rows rewritten. Tolerance boundary tested |
| 4 | `get_members(universe, as_of)` correct at any date | Tests: from-inclusive / to-exclusive, leavers/joiners, leave-and-rejoin, 57-member universe (no size assumption), overlap/bad-date rejection, idempotent reload |
| 5 | Quality report runs on real data, findings reported as measured | See "Data findings" |
| 6 | pytest green; no secrets in git; app role is DML-only | 48 passed (+2 live, pass with `-m live`). `.env` ignored; no DB password found in any file that would be tracked. `predict_app` gets `denied` on CREATE/DROP/ALTER (tested) |

### What exists

* `src/predict_stock/config.py` + `config/default.yaml` — typed config; VN market constraints
  (lot 100, fee 0.15 %/side, sell tax 0.1 %, slippage 0.1 %, T+2, ±7/10/15 % bands, HOSE tick table).
  **These are the brief's defaults, not re-verified against current regulations.**
* `db/models.py`, `alembic/versions/0001_initial_schema.py` — `instruments`, `universe_membership`,
  `ohlcv_daily`, `trading_days`, `job_runs`. Two MySQL users: `predict_app` (DML) and
  `predict_migrator` (DDL, scoped to the two databases).
* `data/dnse_client.py` — all API assumptions, exact Decimal price conversion, throttle + retry.
* `data/ingest.py` — idempotent, incremental with overlap + drift detection, no partial bars, one
  transaction per symbol, failures recorded but isolated.
* `data/quality.py` — read-only checks; nothing is auto-fixed.
* `universe.py` — point-in-time membership, CSV loader. No "VN30" or "30" anywhere in code.
* `runs.py` — every job stores config snapshot, seed, git commit (+dirty flag) in `job_runs`.

### Data findings (from the real DEMO load, 2018-01-02 → 2026-09-18)

* Calendar: 2,171 trading days (stock consensus).
* 11 `ohlc_inconsistent` errors: 8 stock bars + 3 VNINDEX bars, all mid-2019 (source data; listed in `DNSE_API_NOTES.md`).
* 66 `missing_trading_day` warnings: VNINDEX and VN30 each miss 33 days that all stocks have.
* 5 same-date split rows (2022-12-27) merged — an inference, flagged in every run's `client_warnings`.
* No `big_move`, `zero_volume`, `extra_day` or `stale_series` findings on the 5 stocks.

### Assumptions (details in `docs/DNSE_API_NOTES.md`)

1. The chart-api endpoint is undocumented; unit/timezone/adjustment behaviour is from probing.
2. Stock history is back-adjusted (inferred); volume adjustment unknown.
3. Split-session rows are fragments of one session (inferred).
4. Real rate limit unknown; client throttles to ≥ 0.2 s/request.
5. Membership interval semantics: `effective_from <= d < effective_to`.

### Open items / caveats

* **No real index membership yet.** DNSE has no constituent list. The `DEMO` universe (5 large
  caps, fixed) is a pipeline fixture: any backtest on it has survivorship bias. Need a
  review-by-review CSV for the target index (format: `data/universe/README.md`) before Phase 2
  results mean anything.
* Adjusted prices are not point-in-time (see `DNSE_API_NOTES.md`); Phase 4 needs a plan for tick
  size / price-band checks.
* Exchange per symbol is not stored; the price band defaults to HOSE.
* Future holidays are unknown; the calendar only contains observed days.
* `--refetch` is required to extend history to an earlier `--start` on an already-loaded symbol.
* Nothing is committed to git yet (the repo has no commits); commit when you say so.

### How to run

```bash
cp .env.example .env        # replace the change_me values
docker-compose up -d        # MySQL on 127.0.0.1:3307 (docker-compose v1.29 works here)
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/alembic upgrade head
.venv/bin/python -m predict_stock load-membership data/universe/demo_membership.csv
.venv/bin/python -m predict_stock ingest --universe DEMO
.venv/bin/python -m predict_stock quality --universe DEMO --details 20
.venv/bin/pytest            # add  -m live  to hit the real endpoint
```
