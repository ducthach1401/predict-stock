# Operations: paper trading bot, alerts, backups

Two run modes exist: **`backtest`** (the research jobs: `backtest run`, `swing run`, `invest run`, `reco backtest`) and **`paper`** (the daily bot below). There is **no live mode**:
no code in this project places a real order, holds a broker credential or calls an order endpoint (a test scans the sources for it). `run.mode` in `config/default.yaml` and
`paper run --mode X` accept only those two words; anything else, including `live`, is refused with exit code 2.

## The daily job (`python -m predict_stock paper run`, cron 16:30 Asia/Ho_Chi_Minh, Mon-Fri)

Every step is its own row in `job_runs` (`paper_universe`, `paper_ingest`, `paper_calendar`, `paper_features`, `paper_flags`, `paper_cards`, `paper_state`, `paper_report`) with
status and duration; a failing step writes an alert (`job_failed`) and the day goes on with what can still be done. Every step is idempotent: run the day again and nothing changes.

| step | what it does |
|---|---|
| universe | if `paper.universe_snapshot` points to a CSV (`symbol[,exchange,weight,...]`) it is applied with the effective date of the day (a no-op when unchanged). A new member is backfilled and scored only once it has `ingest.min_sessions` sessions (`insufficient_history` alert until then). A removed member is **not sold**: see below |
| ingest | new OHLC bars of every member and the benchmarks (incremental), data-quality checks. Vendor errors → `api_error` alert; unexplained quality errors → `data_quality` alert and, with `paper.stop_on_quality_error`, no new cards that day |
| calendar | trading calendar from the stocks' bars |
| checks | members without a bar on the session (`data_missing`), newest bar older than `paper.stale_days` (`stale_data`); below `paper.min_share_with_bar` no cards are issued (`cards_skipped`) |
| features | datasets rebuilt with the new day (the point-in-time feature code of Phase 3) |
| flags | open recommendations of a stock that left the universe get the flag `ra khỏi rổ` (`recommendations.flags`) and a `left_universe` alert |
| cards | the final models score the members; SWING cards every day, INVEST cards **only on rebalance dates** (first session of the month for B1, of the quarter for B2) and on their tranche dates (rebalance + 5 / 10 sessions). Stored in `recommendations` (BUY / WATCH; a stored card is never rewritten), predictions of the final SWING model in `predictions` |
| state | the paper portfolio is **replayed** from the stored cards and the INVEST target book (`sleeve_targets`) by the same engine as the backtest, over the real prices, from the start of the paper record to the session; the result is written to `paper_orders`, `paper_positions` (one book per sleeve: `paper:swing`, `paper:invest_b1`, `paper:invest_b2`), `portfolio_snapshots`, `recommendation_outcomes` and the `status` of every recommendation |
| report | `reports/paper/<date>/report.md`, `report.html` and CSV files; optional Telegram / e-mail |

Recommendation status: `pending` (waiting to fill) → `holding` → `target` / `stopped` / `time_exit` / `kill_switch` / `closed` (a rebalance sale); never filled: `expired` (validity ran out),
`cancelled` (opened above the zone: not chased; kill-switch), `skipped` (no order: the stock was already held in the sleeve, the sleeve was full or the kill-switch was on); `watch` for WATCH cards.

Why a replay: the paper portfolio needs no separate simulation code, follows exactly the rules validated before (T+2, lots, tick, band, costs, no chasing, stops, targets, kill-switch),
and is reproducible: the tables always equal the replay, so a crash or a repeated run cannot leave a half-written state. The record starts at `paper init` (`experiments` row `paper:start`,
or `paper.start_date`) and never moves.

## First use

```bash
python -m predict_stock paper init --start 2026-09-21    # the first session whose cards belong to the paper record
python -m predict_stock paper run --as-of 2026-09-21     # (or let cron do it) after the close, once the vendor has published the day's bars
python -m predict_stock paper status
python -m predict_stock paper crontab                     # prints the cron entries; install them yourself:  python -m predict_stock paper crontab | crontab -
```

Catch-up after downtime: `paper run --from 2026-09-21 --as-of 2026-09-25` runs every session in between, in order (cards of a missed day are generated from the data of THAT day; nothing after it is used).
Do not paper-trade over the held-out period of the research (2025-09-19 → 2026-09-18): it would evaluate the models on it; the record therefore starts after the last date of the data.

## The universe changes

Edit the snapshot file (`data/universe/...csv`, the ticker list of the day) or run `python -m predict_stock universe apply --file ... --effective-date ...`; the next daily run applies it.
* **New stock**: created, backfilled (`backfill`), scored from the day it has enough history. An `universe_change` (info) alert says so.
* **Removed stock**: a `universe_change` (warn) alert; its open recommendations are flagged `ra khỏi rổ`. Nothing is sold for that reason. SWING: the position is held until its target, stop or
  time-stop; INVEST: it stays until the next rebalance, where it is no longer in the target and is sold like any dropped name; a tranche in progress does not add to it. `paper.on_removal`
  can be set to `close_now` (`invest`) to sell it at the next open instead (an `exit` row in the sleeve's target book).
* Demonstrated end-to-end on a simulated market by `tests/test_paper_e2e.py` (five consecutive daily runs; two stocks swapped on day 3 by editing the file; no code changed).

## Alerts (`alerts` table)

`python -m predict_stock paper alerts` lists the open ones, `--ack ID` acknowledges. An identical open alert is not written twice. Categories: `job_failed`, `api_error`, `data_missing`, `stale_data`,
`data_quality`, `cards_skipped`, `insufficient_history`, `universe_change`, `left_universe`, `kill_switch`, `paper_not_initialised`, `notify_failed`, `backup_failed`.
Notifications (off by default, `paper.notify`): Telegram needs `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, e-mail needs `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_TO` (and optionally `SMTP_PORT`) in `.env`. A failed notification is an alert, never a failed job.

## Backups

* `python -m predict_stock paper backup --prune` (cron 17:30 daily): one consistent `mysqldump` (`--single-transaction`, routines, triggers) gzip-compressed into `backups/<database>_<UTC timestamp>.sql.gz` with a `.sha256` next to it, written to a `.part` file and renamed when complete.
  Retention (`backup.daily_keep: 14`, `backup.weekly_keep: 8`): the latest dump of each of the newest 14 days plus the oldest dump of each of the 8 newest older weeks; everything else is deleted with its checksum.
* `python -m predict_stock paper restore-test` (cron Sunday 03:00): takes a fresh dump, verifies its checksum, restores it into a scratch database (`backup.restore_database`, created and dropped with the admin
  account `MYSQL_ROOT_PASSWORD`; it can never be the real or the test database) and compares every table (row count and `CHECKSUM TABLE`) with the live one. A difference or failure is a `backup_failed` alert and exit code 1.
  Last run on the real database: 33 tables, 264,767 rows, 0 differences, 61 s.
* The model files (`artifacts/models`), Parquet datasets (`artifacts/datasets`), reports and config are files, not in the dump: `git` holds the code and config, and datasets / models are reproducible from the database and the seed
  (`paper prune-datasets --keep 5` deletes the Parquet files of old dataset versions no model was trained on; the manifests stay).

### Restoring for real (disaster recovery)

```bash
# 1. stop cron / the daily job; keep the damaged database aside if you can (docker exec predict-stock-mysql mysqldump ...)
# 2. pick the dump and check it:      sha256sum -c backups/predict_stock_YYYYMMDD_HHMMSS.sql.gz.sha256
# 3. recreate the database (admin):   mysql -h127.0.0.1 -P3307 -uroot -p -e "DROP DATABASE predict_stock; CREATE DATABASE predict_stock CHARACTER SET utf8mb4"
# 4. load it (the dump contains the schema, the data and the alembic_version row):
gunzip -c backups/predict_stock_YYYYMMDD_HHMMSS.sql.gz | mysql -h127.0.0.1 -P3307 -uroot -p predict_stock
# 5. grants of predict_app / predict_migrator live in the MySQL instance, not in the dump: if the instance is new, run docker/mysql/init again
# 6. python -m alembic current   # must show the head revision;   python -m predict_stock paper status;  then re-enable cron
```
Repeat the daily job for the missed sessions with `paper run --from ... --as-of ...`: the replay rebuilds the paper state from the recommendations and prices.

## Scheduling

Cron with `flock` (single flight; an overlapping or repeated call is harmless because every job is idempotent). `paper crontab` prints the three entries (daily job, daily backup, weekly restore test);
nothing is installed automatically. Logs: `logs/paper_*.log`. One daily run takes about one minute on the real data (ingest 20 s, datasets 20 s, scoring 20 s).
