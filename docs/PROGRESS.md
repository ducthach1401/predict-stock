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

## Phase 4 — Backtest engine & baselines  ·  acceptance met (2026-09-20), uncommitted

Engine rules and how a strategy plugs in: [docs/BACKTEST.md](BACKTEST.md). Baseline results: [docs/BASELINES.md](BASELINES.md) (tables + charts).

### Acceptance

| Criterion | Evidence |
|---|---|
| Unit tests for fee, tax, lot, tick, price band, T+2, limit fill, stop-first | `tests/test_backtest_market.py` (42: tick table, rounding directions, ceiling/floor, fee/tax/slippage, lots, band inference) and `tests/test_backtest_engine.py` (39 scenarios, each hand-worked): fills at the next open, never the same session; cash flow with fee and whole lots; round trip pays fee both ways + tax on sells only + slippage; T+2 (also for full rebalances, time-stops and stops); buy at the ceiling / sell at the floor blocked; limit buy fills only if the low reaches it, at the better of limit and open, valid for N sessions; **both levels in one bar → stop first** (and `target_first` on request); gap-through exits at the open; suspension; cash limits; rebalance threshold; determinism; the future cannot change the past |
| `make backtest` runs the baselines | `make backtest` (builds/reuses the datasets, then 9 baselines + a 20-seed noise floor): **2 min 8 s** on the real data |
| Report saved in the DB and docs/ | 10 rows in `experiments` (`baseline:<key>` + `baseline:noise_floor`: metrics, config hash, equity-curve artifact path + sha256; re-running reuses rows), `docs/BASELINES.md`, `docs/img/baselines_equity.png`, `docs/img/baselines_cost_sensitivity.png`, equity CSVs under `artifacts/backtests/` |

Suite: **460 tests** pass (+ 4 live), of which 126 are new in this phase.

### What was built

1. **Engine** — event-driven daily simulator, long-only: signal at close t, fill at the open of t+1; limit orders with fill-rate records; stop / target / time-stop from each session's high/low, **stop first** when both fall in one bar, gap-through at the open; T+2; lots of 100; tick size by price; price band (known exchange, else inferred PIT); fee + tax on sells + slippage; suspension and locked limits; every blocked attempt is counted by reason.
2. **Portfolio** — top-K, equal / inverse-volatility weights, cap per instrument (excess redistributed or left in cash), rebalance threshold (relative to the target position).
3. **Walk-forward** — expanding/rolling folds, embargo gap, purging by label end date (`purge_train`), held-out final period excluded from every fold and every development run.
4. **Baselines** — equal-weight universe, short-term momentum, short-term mean reversion, momentum 6-12 months (+ inverse-vol variant), buy & hold VNINDEX and VN30, and a **random top-K noise floor over 20 seeds** (weekly and monthly). No baseline parameter was tuned (K = 10 and the schedules are conventions).
5. **Metrics** — CAGR, Sharpe, Sortino, MDD, Calmar, turnover, win rate, profit factor, expectancy, exposure; before/after costs (a second run with all costs at 0), sensitivity to costs ×0 / 0.5 / 1 / 2 / 3, up / sideways / down periods (ex-post, VNINDEX) and calendar years, walk-forward folds, VN30-era like-for-like table; equity + drawdown and cost-sensitivity charts (dataviz palette, validated: worst adjacent CVD ΔE 9.1; 3 of 4 hues are < 3:1 on the surface, so direct labels and the tables carry the values).

### Development-period results (2019-03-01 → 2025-09-18, net of costs, LARGE50)

| strategy | CAGR | Sharpe | MDD | turnover/yr | CAGR before costs |
|---|---|---|---|---|---|
| Equal-weight universe | 22.2% | 1.01 | -48% | 0.2 | 22.3% |
| Momentum 6-12 months | 22.5% | 0.91 | -58% | 3.0 | 25.5% |
| Momentum 6-12 months, inverse-vol | 21.2% | 0.90 | -57% | 3.2 | 24.2% |
| Buy & hold VN30 (from 2020-05-11) | 17.6% | 0.91 | -43% | 0 | 17.7% |
| Short-term momentum (10 d, weekly) | 16.5% | 0.72 | -45% | 25.6 | **39.1%** |
| Buy & hold VNINDEX | 8.4% | 0.52 | -40% | 0 | 8.4% |
| Short-term mean reversion (weekly) | -16.0% | -0.61 | -78% | 28.1 | 1.9% |
| Random top-10, monthly (median of 20 seeds) | 11.9% | 0.59 (p95 0.78) | | 9.4 | 19.5% |
| Random top-10, weekly (median of 20 seeds) | -12.7% | -0.46 (p95 -0.34) | | 40.8 | 15.0% |

What the numbers say (and do not):
* **Nothing beats holding the whole universe equal-weight after costs** (Sharpe 1.01); the 6-12-month momentum baselines match it on return and are worse on risk. That is the bar a model has to clear, and it is inflated by the universe look-ahead (below).
* **Costs decide the short-horizon strategies**: short-term momentum earns 39% a year before costs and 16.5% after (turnover 26×/yr, about 22 points of drag); at 3× the costs its Sharpe is negative, while 6-12-month momentum stays positive (0.77). Weekly random and weekly mean reversion lose money after costs.
* The baselines' **absolute levels are not achievable in real time**: the universe was picked with today's information (equal-weight LARGE50 makes 22% vs 8% for the price index VNINDEX), prices are dividend-adjusted, the benchmarks are not.
* The differences between the top rows are within noise (standard error of a Sharpe ratio over ~6.5 years ≈ 0.4); the test windows of the walk-forward folds swing from Sharpe +2.6 to -1.1 for the same strategy.

### Bugs and design mistakes found while building it (kept here on purpose)

1. **Rebalance threshold was absolute (fraction of equity)**: with 50 names each weight is 2%, so no top-up ever crossed a 2% band; exposure sank from 95% to 80% and equal-weight showed 18.6% instead of 22.2%. Found by checking the exposure, not the return. It is now relative to the target position (regression test).
2. A window starting mid-history crashed on a read-only reference-price array (found by test).
3. The first purging test expected `horizon - 1` purged samples; a label that ends **on** the first test session already used that session, so `horizon` samples go.
4. Short-term momentum looked too good (39% gross vs an information coefficient near zero). Cross-checked with an independent vectorised open-to-open calculation, a reversed portfolio (5.8%) and random baselines: the engine agrees with the independent calculation; the cause is the data (universe chosen with hindsight) and it disappears after costs. No engine change.
5. Chart labels of two lines finishing together overlapped; labels are now spread.

### Limits and assumptions

* LARGE50 look-ahead / survivorship; adjusted prices (lots, ticks, bands approximate in level); price band **inferred** (no exchange history); 59 bars whose open was unreliable cannot fill market orders that day.
* T+2 for the whole period (earlier cycles may have been longer; not verified); sellable for the whole session two sessions after purchase; sale proceeds reusable at once; costs are the brief's defaults, not re-verified; no market-impact model (negligible at 1 bn VND, not at larger sizes).
* The baselines trade at the next open only: limit orders, stops and targets are engine features that are tested but not exercised on real data yet.
* Buy & hold benchmarks are price indices (no dividends) and not directly tradable; their costs are one round trip.
* **The held-out final period (2025-09-19 → 2026-09-18) has not been touched** by any figure: `backtest oos --final` evaluates it once (recorded in `experiments`, a second call exits with code 3); it is meant to be run once for the final model and all baselines together in Phase 5.

### How to run
```bash
make backtest                                   # datasets (reused) + baselines + report + DB, about 2 minutes
python -m predict_stock backtest list           # the baselines
python -m predict_stock backtest run --baseline mom_long --noise-seeds 0 --no-report
python -m predict_stock backtest oos --final    # ONCE, at the very end; refused afterwards
```


## Phase 5 — SWING strategy  ·  implemented and evaluated (2026-09-20), uncommitted · **verdict: does not beat the baselines after costs**

### Pre-registration (written BEFORE any SWING model was trained)

Fixed now so that the result cannot be argued into shape afterwards. The same content is stored in `experiments` (`swing:preregistration`, with its timestamp) by `swing run` before it trains anything.

* **Model**: LightGBM (`lightgbm` 4.x), five boosters on the `swing:1` features: rank regression on `fwd_rank_5`; binary classifier for "target touched before the stop within 10 sessions" (`tb_label == 1`, target 2 ATR / stop 1 ATR); q10/q50/q90 of `fwd_ret_5`. Calibration isotonic on the validation rows (Platt kept for comparison). Deep learning is **not** attempted: nothing here justifies it before a tree model beats the baselines.
* **Primary strategy (the one the verdict is about)**: top-10 by the rank score at the close, equal weight, full rebalance on the first session of each week, orders at the next open through the Phase 4 engine. Like-for-like with the weekly baselines. No probability filter.
* **Secondary strategy**: barrier trades (entry at the next open, stop 1 ATR, target 2 ATR, time-stop 10 sessions, at most 10 positions, calibrated probability ≥ the training base rate). Reported, not decisive.
* **Walk-forward**: expanding, first fold trains on ≥ 500 sessions, test windows of 125 sessions, embargo 10 sessions (= the barrier horizon), samples purged by the end date of their labels, validation = the last 125 sessions of each training window. Nothing at or after 2025-09-19 is used.
* **Tuning**: one Optuna study (≤ 25 trials, seed 42) on the first fold's train/validation rows only, objective = mean daily rank IC of the ranking booster on the validation rows; the winner is frozen for every fold. Every trial, failed ones included, is an `experiments` row.
* **Sensitivities** (K = 5/10/20, entry-probability threshold, ATR target/stop, cost multiples) are reported for information and **never** used to choose the configuration.
* **Decision rule** — the held-out period is opened (once, together with all baselines) **only if** the primary strategy, over the same window as the baselines, has: (1) net Sharpe above **every** portfolio baseline; (2) above the 95th percentile of 20 random weekly portfolios; (3) mean daily rank IC > 0 with an overlap-adjusted t-statistic ≥ 2.0; (4) positive net Sharpe in ≥ 60% of the walk-forward test windows; (5) net Sharpe still > 0 at 2× costs. If any fails, the conclusion is written as "does not beat the baselines after costs", the held-out period stays untouched and the configuration is not tuned further.

### What was built

| piece | where |
|---|---|
| model: 5 LightGBM boosters (rank, event, q10/q50/q90), isotonic + Platt calibration, holding-time table, TreeSHAP, byte-stable `model.json.gz` | `src/predict_stock/swing/model.py`, `calibration.py`, `holding.py` |
| walk-forward with purging by label end + embargo, per-fold models, final model on all development data | `swing/folds.py`, `swing/walkforward.py` |
| Optuna (one study, first fold only, every trial → `experiments`) | `swing/tuning.py`, `swing/registry.py` |
| strategies: primary top-K weekly, secondary barrier trades (engine got `max_positions` and `only_if_flat`) | `swing/strategies.py`, `backtest/engine.py` |
| evaluation vs baselines over the same window, sensitivities, pre-registered decision, held-out guard | `swing/evaluate.py`, `swing/job.py`, `swing/metrics.py` |
| report + charts | `docs/SWING.md`, `docs/img/swing_*.png` |
| DB | migration `0004` (`predictions.details` JSON); `models` (12 names × 2 versions, status `candidate` / `superseded`), 136,958 `predictions`, 26 Optuna + 1 pre-registration + 1 amendments + 2 walk-forward + 4 strategy `experiments`, 88 `model_metrics` |

### Result (development walk-forward 2020-03-13 → 2025-09-18, 11 test windows, net of costs, baselines re-run over the same window)

| strategy | CAGR | Sharpe | max DD | turnover ×/yr | gross CAGR | gross Sharpe | Sharpe at 2× costs |
|---|---|---|---|---|---|---|---|
| **SWING top-10, weekly (primary)** | 31.9% | **1.07** | -48% | 32.3 | 64.7% | 1.82 | 0.41 |
| Equal-weight universe | 31.7% | **1.29** | -48% | 0.2 | 31.8% | 1.29 | 1.27 |
| Momentum 6-12 m | 32.2% | 1.15 | -57% | 3.0 | 35.2% | 1.23 | 1.08 |
| Momentum 10 d, weekly (like-for-like) | 25.8% | 0.97 | -44% | 25.5 | 49.8% | 1.61 | 0.41 |
| Random top-10 weekly (20 seeds) | median -0.07 Sharpe, 95th pct 0.08 | | | | | | |
| SWING barrier trades (secondary) | -1.1% | 0.04 | -49% | 39.7 | 29.5% | 1.42 | -1.11 |

Pre-registered decision: **1 of 5 criteria fails** — net Sharpe 1.07 is below equal-weight's 1.29 (it beats the weekly baselines — 10-day momentum 0.97, mean reversion, random — and the random-portfolio 95th percentile, but not equal-weight 1.29, 6-12-month momentum 1.15 or its inverse-vol variant 1.13; rank-IC t-stat 4.4; positive net Sharpe in 7 of 11 windows; still positive at 2× costs). The held-out period (2025-09-19 → 2026-09-18) was **not opened**, and no configuration was tuned after seeing the result.

What the numbers say (and do not):
* **There is a real ranking signal**: mean daily rank IC 0.055 (t 4.4 with the overlap adjustment; positive in 9 of 11 windows) against 0.015 for 10-day momentum and about 0 for mean reversion.
* **The signal does not survive costs well enough**: the weekly top-10 earns 65% a year before costs and 32% after (turnover 32×/year). Equal-weight earns the same CAGR with lower volatility, hence the higher Sharpe. It beats equal-weight in 4 of 11 windows. Costs are the whole story: with zero slippage the same portfolio has Sharpe 1.38, with zero fee and tax 1.52, with the configured costs 1.07 — but those are not achievable, and this was not used to choose anything.
* **The probabilities are not usable.** Out of sample the "target before stop" probability has no measurable skill: mean per-fold AUC 0.51 (0.44–0.55), Brier 0.224 (raw) / 0.226 (isotonic) against 0.223 for the training base rate. Calibration did not help (ECE raw 0.070, isotonic 0.073, Platt 0.070): the event rate moves between windows (22%–44%) far more than the model can follow. Filtering entries by probability made the weekly portfolio *far worse* (net Sharpe 0.05 at ≥ 1.0× the calibration-window base rate, -0.45 at 1.25×, and at 1.5× no name passes so there are no trades) — the probability is not a useful entry filter.
* **Return quantiles are reasonable in width** (q10/q50/q90 covered 10.7% / 48.8% / 88.6%; the q10–q90 interval 77.9% against 80% nominal; tail pinball loss slightly better than a constant, median slightly worse). **Expected holding time has no skill**: mean absolute error 2.70 sessions against 2.67 for always guessing the overall median.
* Barrier trades lose money after costs whatever the ATR pair or probability threshold that was tried (net Sharpe -0.58 … 0.46 across the ATR pairs and probability thresholds tried, none positive at 2× costs): gross Sharpe is 1.2–1.6 but turnover is 26–44×/year.
* K (5 / 10 / 20 → net Sharpe 1.03 / 1.07 / 0.93) is not what matters.
* Reading it at face value: Sharpe differences of a few tenths over ~5.5 years are within noise (standard error ≈ 0.5); "does not beat equal-weight" is the fair reading, not "is worse than".

Deep learning was **not** attempted: the boosted trees do not beat the baselines, so nothing justifies a heavier model.

### Bugs and design mistakes found while building it (kept on purpose)

1. **Isotonic calibration output a probability of 1.0** in 3 of 11 folds (a few top-scored validation rows were all events). Fixed by fitting isotonic on equal-count bins of ≥ 150 rows (`swing.isotonic_min_bin`; test with 8 planted "lucky" rows). Found by looking at the highest predicted probability per fold.
2. **The barrier strategy was flat for whole test windows** because its entry bar was the *training* base rate while the calibrated probability is centred on the *validation* base rate (lower in 4 of 11 folds). Threshold now relative to the calibration window's rate. Found by looking at the equity curve.
   Both were found after the first run had been looked at. They touch only the probability and the secondary strategy; the primary strategy (rank score) is byte-identical in both runs (Sharpe 1.0746 both times). The first run's figures are kept in `docs/SWING.md`, the change is recorded in `experiments` (`swing:amendments`), and the barrier strategy went from Sharpe 0.32 (partly idle) to 0.04 — i.e. the correction made it look *worse*, which is a sign it was not chosen for looks.
3. The report first said "3 rows purged" per fold; the purge removes **nothing** (embargo = barrier horizon). The 3 rows are unlabelled instrument-days of a suspended stock (Oct 2020). Accounting is now separate (`purged` vs `unlabelled`).
4. My first top-K helper renormalised weights when a probability filter left fewer than K names (each got 1/n instead of 1/K, so "cash" never appeared). Found by a unit test; weights are now 1/K.
5. Two chart/report slips (categorical x ticks, a stale "five test windows" sentence) fixed.
6. **A small leak into the reported figures, found during Phase 6** (fixed): the IC, quantile, calibration and holding-time figures used the 5 / 10-session forward labels of the last development days, whose windows end after 2025-09-19 — up to 10 sessions of *held-out* prices entered those figures through the labels. The portfolio backtests stop before the held-out period and were never affected; the models were not affected (training labels are purged); the verdict does not use those figures. Those figures now use only rows whose label ends before the held-out period (IC 0.0552 → 0.0554, ECE raw 0.0694 → 0.0701; recorded as amendment 3 in `experiments`). The held-out period is therefore untouched in the sense that no strategy or model was ever evaluated on it, and — after this fix — no figure depends on its prices.

### Limits and assumptions
* Everything under Phase 4's limits applies (universe look-ahead/survivorship, adjusted prices, inferred price band, T+2 all period, brief-default costs not re-verified, no market impact); LARGE50 hindsight inflates every strategy here, the baselines included.
* The comparison window starts at 2020-03-13 (the first test session), not 2019; baselines were re-run over it. Eleven six-month windows, one path of history.
* One Optuna study on the first fold's validation rows (25 trials, best validation IC 0.059 — optimistic by construction, being a max of 25). Its parameters are frozen; the test-window results do not use them for selection.
* Barrier strategy trades from the next open with stop/target as a percentage of the signal-day close, so realised barriers differ slightly from the label's.
* `swing oos --final` exists and is guarded (refuses unless the stored development run passed the pre-registered criteria under an unchanged pre-registration; exit code 4). It has not been and, under the pre-registered rule, must not be run for this model.

### How to run
```bash
make swing                                   # about 5 minutes; idempotent: unchanged data + config gives identical model files and rows
python -m predict_stock swing run --trials 25 --noise-seeds 20
python -m predict_stock swing report         # rewrite docs/SWING.md from the latest stored run
python -m predict_stock swing oos --final    # only after a PASS; refused otherwise
```

## Phase 6 — INVEST strategy (B1 1-3 months, B2 6-12+ months)  ·  implemented and evaluated (2026-09-20), uncommitted · **verdict: neither preset beats the baselines after costs**

### Pre-registration (written BEFORE any INVEST model was trained)

The same content is stored in `experiments` (`invest:preregistration`, timestamped) by `invest run` before anything is fitted.

* **Presets.** B1: label horizon 63 sessions, monthly rebalance. B2: horizon 126 sessions, quarterly rebalance. Embargo between training and test = the preset's horizon; every training/validation sample is purged by the end date of its label (labels overlap heavily). Folds: expanding, ≥ 500 training sessions, test windows of 125 sessions, validation = last 125 sessions of the training window. Nothing at or after 2025-09-19 is used.
* **Candidates (small sample ⇒ simple models, strong regularisation; no deep learning).**
  1. `factor` — rule-based composite, **no fitted parameter**: mean of the cross-sectional ranks of 6-month momentum, 12-month momentum (both skipping the last month), *low* 126-session volatility, price above its 200-day average, shallow 252-session drawdown. This is the pre-declared primary reference.
  2. `ridge`, 3. `elasticnet` — on the 16 `invest:2` features (median-imputed, standardised per fold), target = cross-sectional rank of the forward return at the horizon. One hyper-parameter grid on the first fold's training/validation rows, then frozen; every grid point is an `experiments` row.
  4. `lgbm` — LightGBM, 4 leaves, depth 2, ≥ 400 rows per leaf, L2 50, fixed (not tuned).
  The q10/q50/q90 "bear/base/bull" quantiles of the forward return at the horizon come from three quantile LightGBM boosters of the same shallow shape (one set per fold, shared by all candidates).
* **Primary portfolio.** Top 10 by score, equal weight, cap 15%, scheduled rebalance with the engine's minimum-deviation band, each rebalance carried out in **2 tranches** (5 / 10 sessions apart), orders at the next open through the Phase 4 engine. No regime filter, no DCA in the primary.
* **Reported variants (information only, never used to choose anything):** K = 8 / 10 / 15; weighting inverse-vol / risk parity / min-variance (Ledoit-Wolf) / HRP; 1 or 3 tranches; regime filter (VN30 — VNINDEX before VN30 has a 200-day average — below its SMA200 ⇒ keep 50% of the stock weights, the rest in cash); DCA of 10 M VND a month.
* **Decision rule.** For each preset, the candidate with the best net Sharpe over the walk-forward window is judged. The held-out period is opened (once, all baselines together) only if a preset meets ALL of: (1) net Sharpe above every portfolio baseline over the same window (equal-weight, both 6-12-month momentum variants, 10-day momentum, mean reversion); (2) **White's reality check over the four candidates** (stationary block bootstrap, mean block 21 sessions, 2000 resamples): p ≤ 0.10 that the best Sharpe difference against equal-weight is luck; (3) mean daily rank IC > 0 with an overlap-adjusted t-statistic ≥ 2 (effective sample = days / horizon); (4) the net return beats equal-weight in ≥ 60% of rolling 1-year windows; (5) net Sharpe still > 0 at 2× costs. Otherwise: "does not beat the baselines after costs", held-out untouched, no further tuning.
* **Thesis-break conditions** (an output, not a trading rule here): close below the 200-day average; relative strength (rank of 6-month momentum in the universe) below 0.40; drawdown from the 252-session high beyond 15% (B1) / 25% (B2). Their information content is reported, not tuned.
* **Fundamentals.** None exist in the data; everything runs on prices. If columns starting with `fund_` ever appear in the dataset, an ablation (with vs without them) is reported separately.

### What was built

| piece | where |
|---|---|
| candidates: rule-based factor score (nothing fitted), Ridge, ElasticNet, shallow LightGBM; q10/q50/q90 scenario boosters; JSON artifacts, byte-stable | `src/predict_stock/invest/models.py` |
| walk-forward (embargo = the preset's horizon, purge by that label's end), one grid on the first fold, labels ending inside the held-out period excluded from every figure | `invest/walkforward.py` |
| weights: equal / inverse-vol / risk parity / min-variance (Ledoit-Wolf) / HRP with a cap; tranches; regime filter; quarterly schedule | `invest/weights.py`, `invest/strategy.py`, `backtest/baselines.py` |
| stationary block bootstrap, White's reality check, rolling windows, DCA, thesis-break flags | `invest/stats.py` |
| decision rule, per-signal details, fundamentals ablation, held-out guard | `invest/evaluate.py`, `invest/job.py` |
| reports (one per preset) + charts | `docs/INVEST_B1.md`, `docs/INVEST_B2.md`, `docs/img/invest_b*_*.png` |
| DB | 72 `models` (2 presets × 4 candidates × (8 folds + final)), 13,148 `predictions` (rebalance dates only, with scenario quantiles / horizon / thesis flags / contributions in `details`), 34 hyper-parameter grid rows + 2 study rows, pre-registration, 2 walk-forward + 8 strategy `experiments`, 128 `model_metrics` |

### Result (walk-forward, net of costs, baselines re-run over the same window; 90% block-bootstrap intervals)

| | B1: 2021-05 → 2025-05 (horizon 63, monthly) | B2: 2021-08 → 2025-08 (horizon 126, quarterly) |
|---|---|---|
| Factor score | 5.0% · Sharpe 0.32 | 3.2% · 0.25 |
| Ridge / ElasticNet | 1.8% · 0.20 / 1.5% · 0.19 | 9.5% · 0.47 / 8.0% · 0.42 |
| Shallow LightGBM | -0.4% · 0.11 | 10.8% · 0.51 |
| **Equal-weight universe** | **9.2% · 0.49** | **13.3% · 0.66** |
| Momentum 6-12 m / VN30 buy & hold | 6.0% · 0.35 / 0.0% · 0.10 | 10.3% · 0.50 / 4.1% · 0.30 |
| Best candidate's Sharpe − equal-weight | factor: -0.17 [-0.63, +0.29] | LightGBM: -0.15 [-0.52, +0.20] |
| White's reality check p (4 candidates) | 0.963 | 0.967 |
| Best candidate's rank IC (t) | factor -0.019 (-0.25); ridge 0.062 (1.23) | LightGBM 0.013 (0.15); ridge 0.084 (1.11) |
| Rolling 1-year windows ahead of equal-weight | 32% | 36% |

Pre-registered decision: **B1 fails 4 of 5 criteria, B2 fails 4 of 5** (both pass only "still positive at 2× costs"). The held-out period was **not opened**; nothing was tuned afterwards.

What the numbers say (and do not):
* **No candidate is distinguishable from equal-weight, and none is better in point estimate.** Every Sharpe interval spans roughly -0.4 … +1.4 (about ±0.9 around the estimate) — four years, one path, overlapping 63/126-session labels. The reality check, which corrects for having looked at four candidates, says the best difference against equal-weight is what luck would produce at least 96% of the time.
* **The simple rule-based score is not worse than the learned models in B1 and is the worst in B2** — but IC by fold swings between about -0.4 and +0.3 for it (momentum regimes), so this is noise too. The only positive-IC learned models (ridge / elastic net, IC 0.06-0.09, t ≈ 1.1-1.2) do not convert it into a better portfolio after costs.
* **Costs are not the issue here** (1-4 points a year of cost drag at 2-5× turnover, against 14-20 points for the 10-day strategies): the low-turnover INVEST portfolios simply do not beat holding the whole universe.
* **The scenario quantiles (bear / base / bull) do not beat a constant** (pinball loss above the pooled out-of-sample quantile at q10, q50 and q90, both presets): they are a rough spread, not a forecast.
* **Variants (information only, differences the size of the intervals):** the regime filter (index < SMA200 ⇒ half in cash) improved drawdown (about -49% → -38%) at equal Sharpe in B2 and lifted B1's Sharpe from 0.32 to 0.49; a single tranche did better than two in both presets (B2: Sharpe 0.64 vs 0.51 vs 0.42 for 1/2/3 tranches); HRP / inverse-vol / risk-parity weights change little, min-variance hurts in B1. None of this was used to choose anything.
* **Thesis-break flags** carry little information in these holdings (e.g. B2: flagged names returned +7.8% over 126 sessions against +4.6% for clear ones — the wrong way round): reported, not a rule.
* **Fundamentals**: none exist in the data; the models run on prices only. If `fund_` columns ever appear, the same job reports the with/without difference (tested on synthetic data with a planted fundamental signal).
* **DCA** (10 M VND a month): B1 best candidate 5.3% money-weighted vs 7.3% for equal-weight; B2 16.0% vs 21.1% (approximation on the return series).

### Bugs and design mistakes found while building it (kept on purpose)
1. **Evaluation window vs the last complete test window.** With 63/126-session embargoes the last complete window ends 2025-05 (B1) / 2025-08 (B2), months before the last development session; leaving the portfolio unrebalanced after it would have distorted the tail. Every evaluation (candidates, baselines, noise, variants) now ends with the last complete test window. Found by the smoke run on real data before the real run.
2. **Labels reaching into the held-out period** (63/126 sessions forward): excluded from every IC / quantile figure by construction here. This is what led to finding and fixing the same small leak in Phase 5 (see there).
3. **Degenerate covariance** (windows shorter than 60 sessions, names that never traded): produced NaN weights and warnings in a test; every covariance scheme now falls back to equal weights and a test runs them with `-W error`.
4. HRP is only *nearly* invariant to the order of the names (it bisects a dendrogram order); the test now states the true property.
5. Chart labels of lines ending together overlapped (also in the SWING chart): spacing is now relative to the axis.
6. **The grid found nothing**: every ridge / elastic-net grid point had a *negative* validation IC on the first fold (about -0.17 in B1, -0.28 in B2), so the "best" point is the weakest penalty at the edge of the grid — not the strong regularisation intended. Reported in both docs; the grid was pre-registered and is not widened after the fact.
7. The shallow LightGBM often keeps ≤ 5 trees (B1: 5 of 8 folds), i.e. its scores are nearly constant there and ties are broken by instrument id. Shown per fold in the reports; the candidate that looks best by Sharpe in B2 is therefore the least trustworthy to read anything into.

### Limits and assumptions
* Phase 4's limits (LARGE50 hindsight, adjusted prices, inferred bands, T+2, brief-default costs, no market impact) apply; equal-weight LARGE50 is inflated by hindsight, which makes it a harder baseline than anything achievable in real time.
* Four years of out-of-sample returns and overlapping labels: the number of independent observations is a handful. The bootstrap resamples days in blocks of ~21 sessions; it does not repair a sample this short.
* The comparison window (from 2021) differs from Phase 4's (from 2019) and from SWING's; baselines were re-run per preset.
* The candidate with the best net Sharpe is judged (a selection); the reality check accounts for it, the other criteria do not.
* Predictions are stored for rebalance dates only (decisions are made there); per-day predictions exist only in the run's memory.
* `invest oos --final` exists and is guarded (refuses unless a preset passed under an unchanged pre-registration; exit code 4). Under the pre-registered rule it must not be run for these models.

### How to run
```bash
make invest                                  # about 3 minutes; idempotent (unchanged data + config: identical model files and rows)
python -m predict_stock invest run --preset b2 --noise-seeds 20
python -m predict_stock invest report        # rewrite the two reports from the latest stored runs
python -m predict_stock invest oos --final   # only after a PASS; refused otherwise
```

## Phase 7 — Recommendation cards & combined portfolio  ·  implemented and evaluated (2026-09-20), uncommitted · **the cards work as specified; the strategy behind them does not beat the baselines**

### Rules fixed BEFORE the recommendations were backtested (`config/default.yaml`, section `reco`)

None of these numbers may be changed after seeing the backtest of the cards; sensitivity to them would be reported, not used to choose.

* **Cards.** BUY / WATCH / NO_TRADE. A card is issued only with all four mandatory parts (entry, target — or scenarios for INVEST —, rationale, holding time) and valid prices; otherwise it is NO_TRADE with the reason. Every card carries the pre-registered validation verdict of the model behind it and a "statistical estimate, paper trading only" line.
* **SWING entry.** `auto`: breakout (buy-stop at the 20-session high + 0.1 ATR, zone up to +0.3 ATR, cancelled — not chased — if the session opens above the zone) when the close is within 2% of the 20-session high, else pullback (limit zone = close − 0.60 … 0.15 ATR). ATO is available by configuration. All prices are rounded to the HOSE tick and clamped into the next session's ±7% band; a zone that cannot fit is not issued. Validity 3 sessions.
* **SWING exits.** Reference entry = the middle of the zone. Stop = entry − 1.0 ATR. Target 1 = entry + 1.0 ATR (half of the position, then the stop moves to the average entry from the next session), target 2 = entry + 2.0 ATR (the barrier of the model's probability). Both targets are capped just under the highest high of the last 60 sessions when it lies above the entry. **R:R = (target 2 − entry) / (entry − stop); below 1.5 the card is not issued.** Time-stop 10 sessions. Earliest sale = 2 sessions after the fill (T+2).
* **SWING size.** Loss if the stop is hit = 0.75% of total capital, capped at 5% of capital per name, whole lots of 100, at most 6 positions; open risk ≤ 5% of capital.
* **INVEST cards.** Top 10 of the pre-declared candidate (the rule-based factor score, nothing fitted) at each scheduled rebalance; limit zone [−2%, +0.5%] around the close, order = limit at the zone's high, valid 5 sessions, 2 tranches (5 / 10 sessions apart); bear / base / bull = q10 / q50 / q90 of the horizon return turned into price zones; thesis-break conditions (below SMA200, relative-strength rank below 0.4, drawdown beyond 15% / 25%, out of the top 10) instead of a stop; equal weight within the sleeve.
* **Portfolio.** 30% SWING / 70% INVEST (35% B1 + 35% B2), ≤ 15% of capital in one name over all sleeves, **kill-switch at 20% drawdown from the peak** (all sold at the next open, no buys for 21 sessions, then a fresh peak). Each sleeve is a separate book of positions (a stock held in two sleeves is two positions).
* **Probability shown on a card.** The calibrated model probability is the headline number only if the model's PAST out-of-sample predictions (labels already ended) had AUC ≥ 0.55 and ≥ 3000 rows of evidence; otherwise the headline is the historical win rate of the similar-signal group (same calibrated-probability bucket of the fold's validation rows) with its n, flagged when n < 30. The grade (KHÁ / TRUNG BÌNH / THẤP / CHƯA ĐỦ BẰNG CHỨNG) is on the card.
* **Backtest.** The cards, exactly as issued (prices, order type, cancel condition, validity, stop, two targets, time-stop, size), are turned into engine orders; window = where all sleeves have out-of-sample predictions; costs = Phase 4's; reported before / after costs, with and without the kill-switch, per sleeve, against equal-weight, VN30 and VNINDEX, with block-bootstrap intervals; stated probability vs realised; stated R:R vs realised; expected vs actual holding.

### What was built

| piece | where |
|---|---|
| **Engine** (needed to backtest the cards as stated; every addition has tests, defaults leave Phase 4 results unchanged): explicit limit price, breakout buy-stop, ATO, "cancel if the session opens above the zone" (no chasing), absolute stop / target 1 / target 2 prices, partial sale at target 1 with the stop moved to the average entry from the next session, per-group position caps, kill-switch on drawdown (sell at the next open, no buys for a cooldown, fresh peak afterwards), order tags carried to fills and round trips | `backtest/engine.py` |
| **Card** (dataclass → JSON, Vietnamese text, `recommendations` row): identity, entry, exits, holding, confidence, rationale, sizing, validation status; consistency checks (all four mandatory parts, tick grid, next-session band, ordering stop < entry < target 1 ≤ target 2) | `reco/cards.py` |
| **Builders** with the configured rules: entry style (pullback / breakout / ATO), zone clamped into the ±7% band, stop / targets in ATR capped under the nearest resistance, R:R gate, sizing by risk-per-trade with per-name cap and lots of 100, thesis-break conditions and scenarios for INVEST, grade of the evidence, template rationale from real numbers (always risks and invalidation) | `reco/builders.py` |
| **Aggregator**: sleeve budgets (30 / 35 / 35%), per-name cap across sleeves (scale down when at least half fits, else reject with the reason), open-risk cap, kill-switch state | `reco/aggregate.py` |
| **Sources**: market context from the engine's own prices, stored walk-forward models, similar-signal statistics from each fold's validation rows (point-in-time), evidence (past out-of-sample AUC with labels already ended) | `reco/sources.py` |
| **Backtest of the cards** (one shared book, each sleeve its own set of positions via one price column per sleeve and stock), outcomes, calibration of the shown probability | `reco/backtest.py`, `reco/analyze.py`, `reco/job.py` |
| Report, charts, CLI (`reco generate / backtest / report`), `make reco`, migration `0005` (`recommendations.valid_until`, `card`, `card_text`) | `docs/RECOMMENDATIONS.md`, `docs/img/reco_*.png` |
| Live cards of the latest session (2026-09-18): 24 BUY, 13 WATCH, 3 NO_TRADE (VIC does not fit one lot of 100 at these weights on 1 bn; DGW's R:R is 1.26) — BUY / WATCH in `recommendations` (linked to the stored predictions of the final SWING model), texts and JSON in `artifacts/reco/live/` | `python -m predict_stock reco generate` |

### Result (walk-forward window 2021-08-11 → 2025-05-16, net of costs; the cards exactly as issued)

| | CAGR | Sharpe | max DD | gross CAGR |
|---|---|---|---|---|
| **Combined portfolio (cards, kill-switch)** | 2.7% | 0.24 | -42% | 4.6% |
| Combined, no kill-switch | 0.2% | 0.12 | -49% | |
| Equal-weight universe | 5.9% | 0.36 | -48% | 6.0% |
| Momentum 6-12 m | 1.5% | 0.19 | -58% | 3.6% |
| VN30 / VNINDEX buy & hold (price) | -2.0% / -1.2% | 0.00 / 0.03 | -42% / -40% | |

* Sharpe minus equal-weight **-0.13 [-0.62, +0.34]** (90% block bootstrap), minus VN30 +0.24 [-0.35, +0.82], minus VNINDEX +0.21 [-0.32, +0.69]; at 2× costs the combined Sharpe is -0.12. The combined portfolio is ahead of the two indices in most rolling windows (the indices fell in this window) and behind equal-weight in 39% of 1-year and 0% of 3-year windows.
* **Kill-switch** (20% drawdown) fired twice (2022-05-09, 2022-10-11); drawdown -42% vs -49% without it. One path: an illustration only.
* **Sleeves inside the shared book** (with / without kill-switch, % of capital): SWING +7.3% / +2.9%, INVEST B1 +3.4% / +3.3%, INVEST B2 -0.2% / -5.3%. They are not additive with the "run alone" figures (weights are shares of current equity).
* **SWING cards**: 4,965 BUY cards → 1,208 orders (the rest were for stocks already held, sleeve full, or kill-switch) → 71% filled → 856 trades. Ended: 34% reached target 2, 22% target 1 then the break-even stop, **36% the original stop**, 6% time-stop, 1% kill-switch. Holding time 4.0 sessions against 4.1 stated.
* **Stated vs realised R:R**: stated 1.93 R to target 2, realised expectancy **0.00 R** per trade (win rate 60%, average win 0.89 R, average loss 1.33 R, profit factor 1.08). Trades that ended at the original stop lost **1.43 R** on average, not 1 R: 34% of them were gap-throughs at the open (a stop cannot fire during T+2, so a fall in the first two sessions is executed at the first sellable session), fills averaged 1.33% below the stated stop.
* **Stated probability vs realised**: 31.0% stated vs 34.3% realised (ECE 0.078, AUC 0.499). The policy graded **all** 4,965 cards THẤP (past out-of-sample AUC 0.51-0.53 < 0.55) and showed the historical win rate of similar signals with n instead of the model probability; quoting the trailing realised rate would have ECE 0.048 (better calibrated, but the same for every card).
* **Sizing**: the loss-if-stop budget is 0.75% of capital but the 5% per-name cap bound on **every** card (mean weight 4.9%, mean loss if the stop is hit 0.20%): with 1-ATR stops of 2-4% the cap, not the risk budget, sets the size.
* **INVEST cards**: 250 (B1) and 160 (B2) BUY cards incl. second tranches, fill rate 98% / 99%; the realised horizon return fell below the cards' bear / base / bull in 15 / 58 / 91% (B1) and 25 / 56 / 92% (B2) of cases (nominal 10 / 50 / 90).
* **Cross-sleeve overlap**: the same stock was a target of SWING and INVEST in 434 cases (107 days); the largest combined target weight was 12.0% against the 15% cap; positions are tracked per sleeve.

### Acceptance
* Every issued card has entry, target (or scenarios + thesis-break conditions), rationale and holding time; a card that cannot is NO_TRADE with the reason (1,098 in the backtest, 1,066 of them R:R below 1.5). Unit tests: tick rounding on every price band, zone clamped inside ±7%, breakout trigger above the ceiling rejected, T+2 earliest sale and validity across holidays, R:R gate and resistance cap, limit fills / no chasing exactly as the card states, tranches, sizing in lots and caps, aggregator limits, kill-switch state, display grade, evidence point-in-time, DB round trip. 704 tests pass.

### Bugs and design mistakes found while building it (kept on purpose)
1. **The factor score's explanation was mislabelled** (Phase 6, fixed): the contributions of the rule-based score are computed on cross-sectional RANKS but were named after the raw features ("volatility = 96%" for a rank of 0.96). Now named after the rank columns; the stored `predictions.details` of the factor candidate were refreshed by re-running `invest run` (scores and model files identical).
2. **A diagnostic that contradicted itself**: the first "stop fill vs stop" figure mixed original stops with the break-even stops that follow target 1 (fills looked *above* the stop). It now uses original stops only; the per-outcome table shows what each ending earned.
3. **Sleeve attribution**: running each sleeve alone does not add up to the combined book (sizing follows current equity). The report now shows both, and says so.
4. **The risk budget never binds** (see above) — a consequence of the pre-registered caps, reported, not changed.
5. Reasons must not invent: the text builder only prints facts present in the data (a test removes RSI / volume / SMA / relative strength and checks they vanish from the text).

### Limits and assumptions
* All limits of Phases 4-6 apply (LARGE50 hindsight, adjusted prices, inferred bands, T+2 for the whole period, brief-default costs, no market impact). The models behind the cards **failed** their pre-registered criteria; every card shows that. `reco.gate_on_verdict: true` would turn every BUY of such a sleeve into WATCH.
* 3.75 years, one path: all intervals include zero. The rules were fixed before the backtest and were not tuned to it; sensitivity to the entry style, stop multiple or R:R gate was not explored.
* Live cards use the final models trained before the held-out period; they are forward-looking and nothing after their date is evaluated. Live prices are the vendor's; the next session's band is taken from the inferred band.
* INVEST EXIT / REDUCE orders at a rebalance are part of the backtest but are not cards (`recommendations.action` here holds BUY / WATCH; NO_TRADE cards are kept in the report and JSON, not in the table).
* Paper trading only: no order is placed anywhere.

### How to run
```bash
make reco                                                  # cards for the latest session + the backtest of the cards, about 3 minutes
python -m predict_stock reco generate --as-of 2026-09-18   # cards only (BUY / WATCH into `recommendations`)
python -m predict_stock reco backtest                      # backtest + docs/RECOMMENDATIONS.md
python -m predict_stock reco report                        # rewrite the report from the stored run
```
