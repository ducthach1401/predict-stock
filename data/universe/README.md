# Universe snapshots

DNSE's public endpoint has **no constituent list** (verified in the API probe), so membership is
loaded from CSV. Nothing in the code knows an index name or size: a universe is a row in
`universes` plus dated `universe_membership` rows. Adding VN100, a sector basket, anything, is
just another file:

```bash
python -m predict_stock universe apply --universe VN100 --file vn100.csv \
    --effective-date 2026-09-18 --create --name "VN100" [--dry-run]
```

A file is a **snapshot**: the members you want on `--effective-date`. The job diffs it against the
current state, closes rows that left (`valid_to = effective date`), opens rows that joined, and writes
every change to `universe_change_log`. Applying the same file again changes nothing.

## Columns (only `symbol` is required; `#` lines and blank lines are ignored)

| column | meaning |
|---|---|
| `symbol` | ticker on the effective date |
| `previous_symbol` | rename (đổi mã): the ticker the same instrument had before |
| `exchange` | `HOSE` / `HNX` / `UPCOM`. A value different from the recorded one = exchange move (chuyển sàn) |
| `weight` | optional weight in [0, 1]; a change opens a new row so weights are point-in-time |
| `status` | `active` (default) / `suspended` (tạm dừng: stays a member, not tradable) / `delisted` (hủy niêm yết: closes membership and ticker, terminal) |
| `valid_from` | backfill for a NEW member, only allowed when seeding an empty universe |
| `note` | free text |

Rules: intervals are half-open `[valid_from, valid_to)`; the effective date may not precede the latest
recorded change of the universe (history is not rewritten); a member missing from the file is removed;
a removal cannot happen on the day the member joined.

Lifecycle events outside a file: `instrument rename|exchange|status` (same code path).
`universe check` verifies what MySQL cannot enforce (overlapping intervals, ticker used by two
instruments, membership not covered by a symbol row).

## Training vs trading universe

`config/default.yaml → universe.training_code / trading_code` name the two roles (both are just codes).
Training is normally wider; only the trading universe (minus suspended/delisted) may receive
recommendations, and `universe check` reports tradable instruments missing from training.

## Files here

* `candidates_hose_large.txt` — candidate pool for `universe build` (from memory, **unverified**).
* `large50_membership.csv` — **LARGE50**, the seed universe: top 50 candidates by median daily traded value
  (60 sessions to 2026-09-18). Liquidity proxy, not market cap; fixed membership, so backtests on it are
  optimistic. `valid_from` = first DNSE bar. Exchange history is unknown and left empty.
* `demo_membership.csv` — DEMO: 5 stocks, pipeline fixture only.
