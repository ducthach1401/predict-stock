# Universe membership files

DNSE's public endpoint has **no constituent list**, so membership must be supplied here.
Nothing in the code hardcodes an index name or size: a universe is just a `universe_code`
with dated rows.

## CSV format

```
universe_code,symbol,effective_from,effective_to
```

* one row per continuous membership interval of a symbol
* a symbol is a member on day `d` iff `effective_from <= d < effective_to`
  (`effective_to` **empty** = still a member)
* dates are ISO `YYYY-MM-DD`; `effective_from` should be the first day the symbol counts
  as a member (the day the index review takes effect), not the announcement date
* lines starting with `#` and blank lines are ignored
* loading is idempotent; overlapping intervals for the same symbol are rejected

Load with `python -m predict_stock load-membership data/universe/<file>.csv`.

## Files

* `demo_membership.csv` — **DEMO** universe: 5 large caps for exercising the pipeline. It is
  *not* an index and makes no claim about any index composition. Fixed membership implies
  survivorship bias, so never use it for reported performance.
* For a real index (e.g. VN30) provide the historical review-by-review CSV, sourced from the
  exchange's announcements, and record its provenance via `--source`.
