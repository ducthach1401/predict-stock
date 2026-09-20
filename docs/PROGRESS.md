# Progress

## Phase 0 — API verification  ·  done (2026-09-20)
Findings in `docs/DNSE_API_NOTES.md`: public chart-api endpoint works without a key but is undocumented;
prices are back-adjusted (inferred); **no constituent-list endpoint** (so `sync_universe` cannot read the
basket from DNSE and imports CSV instead); several data quirks (split-session rows, index series missing
days, mid-2019 OHLC glitches, exchange-dependent price bands).

## Phase 1 — Database & flexible universe  ·  acceptance met (2026-09-20), uncommitted

Scope adapted from the brief: the seed universe is **LARGE50** (50 large/liquid stocks, kept as it was)
instead of a VN30 CSV.

### Acceptance

| Criterion | Evidence |
|---|---|
| Migration up/down works | `0001 <-> 0002` round-trip test with rows in the old schema (data preserved both ways, DB left at head). Real DB migrated: 111,037 bars, close/volume checksums identical before/after, 52 instruments, 55 memberships. `alembic check` = no drift (also a test). Backup taken before touching the real DB |
| Point-in-time tests: member added/removed mid-period, rename, exchange move | `tests/test_universe.py`: add/remove, leave-and-rejoin, rename (same instrument, no membership churn, old ticker before the date), HNX→HOSE (exchange by date, one continuous membership), suspension (member but not tradable, half-open), delisting, weights by date, `get_instruments_between` |
| Idempotent upsert | `tests/test_universe_sync.py` (re-apply same file/date, later date, with rename/exchange/status rows → no rows, no log entries); `tests/test_ingest.py` (re-run writes nothing: values, `fetched_at`, `run_id` unchanged; config snapshot deduplicated). Real data: re-applying the LARGE50 seed → "already up to date"; re-ingest → 676 bars, 0 written |
| New universe (e.g. VN100) by data only | `test_new_universe_from_a_csv_needs_no_code_change`: 100-row CSV → `apply --create`, no code touched. Creation must be explicit (`--create`), a typo in the code is an error |
| Mutation check | flipping the half-open comparison or the weight comparison makes the relevant tests fail (verified, then reverted) |

Suite at the end of Phase 1: 100 passed + 2 live tests. Real-data regression after the migration: quality report
identical to before (65 / 120 / 109 / 10 / 2 findings).

### Brief item by item

1. **SQLAlchemy 2 + Alembic** — every change is a migration with upgrade/downgrade. DECIMAL prices, DATE trade
   dates, UTC timestamps (`UTC_TIMESTAMP()` defaults; server on +00:00) — asserted by tests. No binary columns
   anywhere (tested); `models`/`datasets` hold path + sha256.
2. **Tables** — all groups exist (A instruments/symbol & status history/universes/membership/change log ·
   B price_bar/revisions/corporate_actions/adjustment_factors + view `v_price_adjusted`/trading_calendar/data_ingest_runs ·
   C feature_sets/label_specs/datasets · D models/model_metrics/experiments/predictions/monitoring_metrics ·
   E recommendations/outcomes/paper_orders/paper_positions/portfolio_snapshots · F config_snapshots/job_runs/alerts).
   `instrument_status_history` was added for suspension/delisting. `recommendations` cannot be written without
   entry, target, stop, holding period, exit conditions and rationale (NOT NULL + CHECKs, tested).
   **Groups C–F are schema only: nothing writes to them yet.**
3. **Half-open intervals + `get_members(universe_code, as_of_date)`** — `valid_from <= d AND (valid_to IS NULL OR d < valid_to)`;
   returns the ticker valid on that date. `get_member_records` adds instrument id / exchange / weight,
   `tradable_only` drops suspended/delisted.
4. **Training vs trading** — `universe.training_code` / `trading_code` in config; `training_members`, `trading_members`,
   `trading_outside_training`.
5. **sync_universe** — `universe apply --file --effective-date [--dry-run] [--create]` (job `sync_universe`, tracked in
   `job_runs`): diff, close old rows, open new, write `universe_change_log` (with run id and file sha256). Handles rename,
   exchange move, suspension, delisting. DNSE cannot supply the basket (Phase 0), so it is CSV only.
6. **Seed** — LARGE50 (migrated from the earlier load, re-appliable from `data/universe/large50_membership.csv`).

### Deviations and honest limits

* **"price_bar = raw, immutable" is not achievable with DNSE data**: only vendor-adjusted prices are served. `price_basis`
  says so; a bar is rewritten only when the vendor re-adjusts, and the old values go to `price_bar_revisions` first.
  `corporate_actions` and `adjustment_factors` (versioned) are empty; the view applies factors only to `price_basis='raw'` rows.
* **LARGE50 has no exchange history** (unknown, left NULL): several members appear to have traded on HNX/UPCoM earlier
  (see `DNSE_API_NOTES.md`), which changes the price band. Fill it with `exchange` values in a snapshot / `instrument exchange`.
* Membership overlaps cannot be prevented by MySQL; `apply` never creates them and `universe check` detects them.
* Suspension/delisting are only recorded when a snapshot or `instrument status` says so; nothing infers them from prices.
* LARGE50 ranking is by traded value (liquidity), from a candidate pool written from memory; backtests on it are optimistic.
* A destructive migration cannot be rolled back atomically (MySQL DDL is not transactional): `0002` was rehearsed on the test
  DB (upgrade → downgrade → upgrade with data) before the real one, and the real DB was dumped first.
* `v_price_adjusted` uses a correlated subquery: 0.12 s for 111k rows today; not tested at much larger scale.

### Data findings (unchanged; real data 2018-01-02 → 2026-09-18, 2,171 trading days)
See the table in `DNSE_API_NOTES.md`: 65 `ohlc_inconsistent` (62 stock + 3 VNINDEX, mostly Jul–Sep 2019),
109 `big_move` (103 in 7 symbols, consistent with earlier HNX/UPCoM listing), 10 `extra_day`, 120 `missing_trading_day`
(+66 index), 2 `zero_volume`; 45 symbols had the 2022-12-27 split-session rows merged.

### Next / open
* Adjusted prices are not point-in-time (tick size / price band need actual prices): still open for Phase 4.
* Flagged OHLC bars: characterised in Phase 2, treatment left to the feature layer.
* Commit: Phases 1 and 2 are uncommitted (the repo has one commit, 3055f61).

### How to run
```bash
cp .env.example .env && docker-compose up -d           # MySQL on 127.0.0.1:3307
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/alembic upgrade head
.venv/bin/python -m predict_stock universe apply --universe LARGE50 --file data/universe/large50_membership.csv \
    --effective-date 2026-09-18 --create --name "Top 50 by traded value"
.venv/bin/python -m predict_stock universe members --universe LARGE50 --as-of 2024-01-02 [--tradable]
.venv/bin/python -m predict_stock universe check
.venv/bin/python -m predict_stock ingest                # training + trading universes from config
.venv/bin/python -m predict_stock quality --details 20
.venv/bin/python -m predict_stock instrument rename --old AAA --new AAX --effective-date 2026-10-01
.venv/bin/pytest                                        # add  -m live  to hit the real endpoint
```

## Phase 2 — Data ingestion & data quality  ·  acceptance met (2026-09-20), uncommitted

### Acceptance

| Criterion | Evidence |
|---|---|
| `make data` twice in a row creates no duplicate rows | On the real DB, run 1 vs run 2: **every data table identical** (row counts and content hashes of 13 tables incl. `price_bar` 111,037 rows, `data_quality_findings`, `alerts`), report identical byte for byte, both exit 0. Also equal to the state *before* run 1. Test: `tests/test_pipeline.py` (same comparison on 3 instruments, plus 0 duplicate primary keys) |
| Quality report has no serious unexplained error | `docs/DATA_QUALITY.md`: verdict **PASS**, 0 unexplained errors. The 65 error-level findings (all `ohlc_inconsistent`) are covered by 7 known issues with measured evidence (see below). The gate is enforced: a test proves the pipeline exits 1 while an error is unexplained and 0 once explained |
| A test per kind of data defect | `tests/test_quality_checks.py`: OHLC (5 variants), non-positive price, price unit (3), zero volume, repeated bar, volume spike (+ needs history), return outlier / big move / ordinary limit move, missing / extra / stale days, no data, duplicates (DB refuses a 2nd row; vendor duplicates reported), severity catalogue. Each planted defect must be found by its own check and a clean twin series must stay clean |
| Backfill time of a new instrument measured and recorded | Real endpoint, test DB (`pytest -m live -s`): **SAB 1,046 ms · BCM 1,268 ms · BVH 1,151 ms** for the full history (~2,160 sessions, fetch + write). Stored per instrument in `data_ingest_runs.duration_ms`. Optional 1H for SAB: 3,722 bars in 2.3 s. Whole `make data` on 52 instruments, nothing new: ~21 s |

Suite: **196 tests** pass (+ 4 live: `make test-live`).

### What was built (brief item by item)

1. **Ingest job** — daily OHLCV for the training universe + VNINDEX/VN30 (>= 5.7 years for every stock, 8.7 for most; VN30 index only exists from
   2020-05-11). Incremental, batched `INSERT ... ON DUPLICATE KEY UPDATE`, only new/changed rows written. `data_ingest_runs` now records source,
   resolution, parameters, mode, counts, **status (ok / blocked / failed), error, duration** — failures get a row too.
   Optional 1H bars: `ingest.intraday.enabled` (default **false**), table `price_bar_intraday` (UTC bar start), same incremental/idempotent logic,
   no revision history. Not enabled on the real DB (only exercised on the test DB with real data).
2. **Backfill** — `backfill` job (and automatically after `universe apply`, `--no-backfill` to skip): instruments with no bars get their full
   history, timed. Fewer than `ingest.min_sessions` (500) sessions → **blocked with the reason** (`data_ingest_runs.status='blocked'`, an alert,
   excluded from `ready_members`); readiness is derived from the bars, so it lifts itself and the alert is acknowledged.
3. **Calendar** — `trading_calendar` synced from stock consensus (`calendar sync|gaps`); `get_working_dates` is unavailable without an API key.
4. **Quality checks** — 14 checks (catalogue at the top of `data/quality.py`), findings persisted in `data_quality_findings` (upsert, `open` /
   `explained` / `resolved`, no duplicates), known issues in `data_quality_known_issues` loaded from `data/quality_known_issues.csv`, Markdown
   report `docs/DATA_QUALITY.md` (deterministic; file untouched when nothing changed).
5. **Price adjustment** — DNSE prices are back-adjusted (Phase 0 inference), so nothing is adjusted. Instead: gap **detection that only reports**
   (`corporate_actions` status `candidate` + warn alert, never applied), `adjustments apply` for operator-confirmed events (new *version* of
   `adjustment_factors`, idempotent), `adjustments set-basis --basis raw` to make the view `v_price_adjusted` apply them. Tested end to end.

### Data findings on the real data (2018-01-02 → 2026-09-18, 52 instruments, 111,037 bars)

* **Errors: 65, all `ohlc_inconsistent`, all explained** (`data/quality_known_issues.csv`): 57 bars in 34 instruments between 2019-07-15 and 2019-09-04, the
  bar's *open* outside [low, high] in all 57 and the close in none, 53 of them on five dates with 4–16 unrelated symbols each (a feed defect on those days);
  3 VNINDEX bars (open = 949.48 three days running); 5 isolated bars. "Explained" means characterised with evidence and given a treatment
  (`keep_flagged_clip_envelope_downstream`, `exclude_open`); **root causes are not confirmed** and the explanations say so. Raw values are unchanged.
* **Warnings: 262, open and not explained** (they do not block): `missing_trading_day` 120, `big_move` 109, `volume_spike` 14, `extra_day` 10,
  `return_outlier` 6, `zero_volume` 2, `repeated_bar` 1. Patterns are in `DNSE_API_NOTES.md` (earlier HNX/UPCoM bands; COVID-crash days).
* **Corporate-action candidates: 0.** One false positive (GVR 2020-03-17) appeared and was traced to five missing sessions, not an event; the detector now
  compounds the band over missing sessions (+ regression test). Its alert is closed automatically when a candidate goes stale.

### Limits and assumptions (read before trusting the data)

* **"Adjusted correctly for every instrument" cannot be certified.** Adjustment is inferred: no gap beyond the price band that an unadjusted
  stock dividend/split would leave. Detection is blind to events smaller than the band (small cash dividends), and blind below ~16% where the exchange
  history is unknown (widest band used). Volume adjustment is unknown. Bars are stored `vendor_adjusted`.
* **Explained is not fixed:** the 65 defective bars stay in the data; whoever builds features must clip high/low to the envelope and ignore `open`
  on them (Phase 3 decision).
* The 262 warnings are unexplained on purpose: I only explained what the data could support.
* `ingest.min_sessions = 500` and the other quality thresholds (z 8, volume x30, price range 500 – 5,000,000 VND) were calibrated on this data,
  not taken from a standard.
* 1H bars exist only from 2023-09-21 and are not revision-tracked.
* Calendar weekday gaps look like Vietnamese holidays but were not compared with an official list.
* One-off: the alert produced by the pre-fix detector was acknowledged with a direct SQL update (the code now does this by itself).

### How to run
```bash
make setup      # first time: MySQL, migrations, LARGE50 seed
make data       # idempotent; exit code 1 if a step failed or an error-level finding is unexplained
make quality    # checks + report only
make test       # make test-live  for the real endpoint
python -m predict_stock quality known-issues --file data/quality_known_issues.csv
python -m predict_stock adjustments detect | apply --file confirmed.csv | set-basis --symbol X --basis raw
python -m predict_stock backfill | calendar gaps | ingest --intraday
```

## Phase 3 — Features, labels, datasets  ·  acceptance met (2026-09-20), uncommitted

Details, catalogue and rules: [docs/FEATURES.md](FEATURES.md).

### Acceptance

| Criterion | Evidence |
|---|---|
| Feature at t does not use data after t (cut the future, features unchanged) | `tests/test_lookahead.py`: all shipped sets (`swing:1`, `invest:1`, `invest:2`) x 3 random panels with gaps and membership changes x 7 cut dates: every value at or before the cut identical after removing the future. Also whole dataset rows (`assemble_frame`) at 3 cuts, and "appending future bars does not change history". **The audit is itself tested**: 7 deliberate time-series leaks and 2 cross-sectional leaks are all caught, a clean control passes, and a label that looks past its declared horizon is caught. Every real build re-runs the audit (manifest: `lookahead_audit.passed`) |
| Same config → same dataset hash | Real data: SWING `sha256 6ec43485…` and INVEST `e2019b08…`; a rebuild into a different name/file (and, in tests, after deleting the row and the file) gives the **same sha256, content hash and inputs hash**. A second `dataset build` prints `REUSED`, writes nothing, adds no row. Changed input data → version 2, version 1 kept |
| New member mid-period and removed member | `tests/test_membership_features.py` + `tests/test_dataset.py`: a joiner has no rows before `valid_from` and full features on day one (history from before it joined); ranks before it joins equal those of a universe in which it never existed; a removed stock stops on `valid_to - 1`, its ranks vanish after, yet its last rows are labelled with prices after removal and it still counts in the label ranks of that day. Also delisted (no bars), suspended sessions, new listing (warm-up), and, in the dataset, `ret_5_csrank` equals the percentile rank recomputed from that day's rows |

Suite: **334 tests** pass (+ 4 live), of which 138 are new in this phase.

### What was built

1. **Registry** — plugin classes registered by decorator under `(name, version)`; sets/specs are YAML data pinned to versions and stored in `feature_sets` / `label_specs` (immutable per version).
   Adding a feature needs **no schema change** (a test compares `SHOW TABLES` before and after). Identifier-like column names are rejected.
2. **SWING features** (10 plugins + ranks), **INVEST features** (6 plugins), `FundamentalProvider` interface (off; a set that needs it fails loudly).
3. **Labels** — `fwd_rank_return` (3/5 for SWING, 21/63/126 for INVEST) and `triple_barrier` (ATR barriers, time barrier, time-to-touch, end date). Every case (target, stop, both on one day, gap-through, time-out, suspension, incomplete window) has a test.
4. **Cross-sectional ranks** only over the members on that date; no id/symbol feature (registry + dataset tests).
5. **`build_dataset`** — Parquet + manifest in `datasets` (universe, dates, feature set, label spec, hash, path) + config snapshot + git commit; CLI `features list|sync`, `dataset build|list|verify`, `make datasets`.

### Real data (LARGE50, 2018-01-02 → 2026-09-18)

| dataset | rows | instruments | first → last decision date | features | label columns | build |
|---|---|---|---|---|---|---|
| `LARGE50__swing1__swing1` | 105,633 | 50 | 2018-02-28 → 2026-09-18 | 23 | 10 | 12 s (incl. audit) |
| `LARGE50__invest2__invest1` | 93,683 | 50 | 2019-02-13 → 2026-09-18 | 16 | 9 | 9 s (incl. audit) |

* Policies applied: 62 defective stock bars repaired (exactly the Phase 2 count), 10 off-calendar bars dropped.
* NaN: `rs_5/rs_10` 26% and `dbeta_126` 18% (VN30 exists only from 2020-05-11); labels are NaN only where the window does not fit. The last 5 (SWING) / 126 (INVEST) sessions have unknown labels by design.
* SWING triple barrier: −1 55%, 0 15%, +1 30% (target 2 ATR is farther than stop 1 ATR; this is not a prediction target balance, do not read it as skill).
* Smoke test for leakage on the real data (mean daily rank IC of each feature against the forward-return rank; **not** a claim of predictive power): SWING max |IC| 0.043 (short-term reversal shows as small negative IC on `ret_1..3`); INVEST `invest:2` max |IC| 0.060 (`mom_12m` vs 126-session rank).

### A leak the audit could not see, found by that smoke test

The first INVEST set had `liq_logvalue_60` (log median close × volume). Its IC against the 63-session rank was **−0.108**, three times the next feature. The cause is
structural: DNSE prices are back-adjusted for corporate actions that happen *after* the date, so a price **level** at t carries information about the future
(the audit cannot see it: the adjusted series is one download). `liquidity` v2 uses ratios only (`liq_trend_60_252`); `invest:1` is kept in the YAML marked deprecated
(versions are immutable), its dataset was deleted, and `invest:2` is the shipped set. Consequence: the cross-sectional level of liquidity is not offered.
`liq_zero_share_60` is constantly 0 in this universe of large caps (no information here; it may matter for less liquid universes).

### Limits and assumptions

* **Not point-in-time by nature of the data:** back-adjusted price levels (only ratios are used), and the volume-adjustment status is unknown.
* The audit proves absence of look-ahead **inside the feature code**, not in the data. A fundamentals provider that ignores `as_of` would not be caught (there is none yet).
* Labels enter at the **close of the decision date**; realistic execution (next open, T+2, price limits, fees, slippage) is Phase 4's job.
* Rows with `rs_*` / `dbeta_*` NaN before 2020-05-11: set `features.benchmark_fallback_symbol: VNINDEX` to fill them (rescaled at the splice); off by default because it mixes two indices.
* 50 stocks × ~2,100 sessions is a small sample: ranks over 50 names are coarse, and overlapping forward windows make consecutive labels highly correlated (the `*_end` columns exist so Phase 4 can purge/embargo).
* The universe is LARGE50 as of today applied backwards (Phase 1 caveat): labels and ranks over it are optimistic.
* Dataset files live in `artifacts/datasets/` (git-ignored); the DB holds path, sha256 and manifest, never the data.

### How to run
```bash
make datasets                                   # features sync + build swing:1 and invest:2 (REUSED when nothing changed)
python -m predict_stock features list           # plugins, sets, warm-ups
python -m predict_stock dataset list | verify --name LARGE50__swing1__swing1
python -m predict_stock dataset build --feature-set swing:1 --label-spec swing:1 --start 2020-01-01 --data-cutoff 2025-12-31
```

