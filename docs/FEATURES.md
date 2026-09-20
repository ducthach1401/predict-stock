# Features, labels and datasets

## How it fits together

```
price_bar ──► Panel (calendar × instrument, repaired) ─┐
universe_membership ──► membership mask (per date) ────┼─► features (ts → cs) ─┐
VN30 (benchmark) ──► aligned, point-in-time ───────────┘                       ├─► one row per (date, instrument) ─► Parquet + `datasets` row
                                                     labels (look forward) ────┘
```

* Code = **plugins** in `src/predict_stock/features/` (`ts.py`, `cs.py`, `labels.py`, `fundamentals.py`), registered with a decorator under `(name, version)`.
* Data = **feature sets / label specs** in `config/feature_sets.yaml` (keys `name:version`), copied to the `feature_sets` / `label_specs` tables by `features sync`. A stored version is immutable: a different content needs a new version.
* A **dataset** is a Parquet file + a manifest row in `datasets`. `make datasets` builds SWING and INVEST; running it again reports `REUSED`.

## Adding a feature (no schema change)

```python
@register_feature
class Amplitude(TimeSeriesFeature):
    name, version, group = "amplitude", 1, "swing"
    DEFAULT_PARAMS = {"window": 10}
    def columns(self): return [f"amp_{self.params['window']}"]
    def warmup(self): return self.params["window"]
    def compute(self, bars, bench):              # bars: ONE instrument, rows <= t only
        n = self.params["window"]
        return pd.DataFrame({f"amp_{n}": (bars.high / bars.low - 1).rolling(n).mean()})
```
Then add `- {name: amplitude, version: 1, params: {window: 10}}` to a NEW set version in the YAML. Changing behaviour = a new class version
(`version = 2`); the old one keeps working and old sets keep pointing at it. A feature cannot expose an identifier (`symbol`, `ticker`,
`instrument`, `exchange`, `*_id`, `id`): the registry rejects the column name.

## Feature catalogue

| set | plugin (v1 unless noted) | columns | definition |
|---|---|---|---|
| SWING | `ret` | `ret_1,2,3,5,10` | `close/close[-n] - 1` |
| | `rsi` | `rsi_14` | Wilder RSI (0–100; 100 with no losses, 50 if flat) |
| | `macd` | `macd_pct`, `macd_hist_pct` | (EMA12−EMA26)/close and (MACD−signal9)/close |
| | `atr` | `atr_pct_14` | Wilder ATR / close |
| | `bollinger_pctb` | `bb_pctb_20` | (close − lower)/(upper − lower), k = 2 |
| | `donchian` | `don_hi_dist_20`, `don_lo_dist_20` | close / prior-20 high − 1 (row t excluded from the channel), same for the low |
| | `zscore` | `zscore_20` | (close − SMA20)/std20 |
| | `gap` | `gap_1` | open / previous close − 1 (NaN on repaired bars) |
| | `volume_spike` | `vol_spike_20` | volume / median of the prior 20 sessions |
| | `rel_strength` | `rs_5`, `rs_10` | stock n-session return − VN30 n-session return |
| | `cs_rank` | `<col>_csrank` | percentile rank among that day's universe members |
| INVEST | `momentum_skip` | `mom_3m,6m,12m` | `close[t-21] / close[t-21-63·m/3] - 1` (skips the last month) |
| | `volatility` | `vol_63`, `vol_126` | std of daily returns × √252 |
| | `downside_beta` | `dbeta_126` | beta to VN30 on VN30's down days only (NaN with < 20 such days) |
| | `drawdown` | `dd_252` | close / rolling 252 max − 1 |
| | `sma_ratio` | `sma_ratio_200` | close / SMA200 − 1 |
| | `liquidity` **v2** | `liq_trend_60_252`, `liq_zero_share_60` | log(median traded value 60d / 252d); share of zero-volume days |
| | `fundamentals` | `fund_<field>` | pass-through of a `FundamentalProvider`; **off** |

`liquidity` v1 (`liq_logvalue_60`) is deprecated and only in the deprecated set `invest:1`: it is a price *level*, and the vendor's prices are
back-adjusted for events that happen after the date, so its value at t carries future information. The look-ahead audit cannot see this (the
adjusted series is one download); on the real data it had by far the largest rank IC of the INVEST set (−0.11 vs < 0.05). The cross-sectional
*level* of liquidity therefore is not offered: it needs unadjusted prices or shares outstanding.

## Labels

| plugin | columns | definition |
|---|---|---|
| `fwd_rank_return` | `fwd_ret_h`, `fwd_rank_h`, `fwd_end_h` | forward return over h sessions (calendar sessions; NaN if either close is missing), its percentile rank among the universe members **on t**, and the date the outcome is known |
| `triple_barrier` | `tb_label` (−1/0/+1), `tb_time`, `tb_ret`, `tb_end` | target = close + m_t·ATR(t), stop = close − m_s·ATR(t), time barrier after H sessions; time to touch; realised return; end date |

Barrier rules (daily bars cannot show the intraday order): both touched in one session → **stop wins**; a session that opens beyond a barrier exits at
the **open** (gap-through); sessions without a bar are skipped; a window that does not fit in the data gives NaN, never a partial label. The label
entry is the **close of the decision date**: costs, slippage, T+2 and price limits belong to the backtester (Phase 4). `*_end` columns are there for
purging / embargo in walk-forward validation.

Shipped: `swing:1` = ranks over 3 and 5 sessions + triple barrier (2 ATR target, 1 ATR stop, 10 sessions); `invest:1` = ranks over 21, 63, 126 sessions.

## Rows and policies

* A row exists iff the instrument is a **member of the universe on that date** (`valid_from <= d < valid_to`), has a bar that day and has the feature set's warm-up of its own history. Suspended sessions produce no row.
* Time-series features of a new member use its history from **before** it joined (that data existed then); ranks on a date use only that date's members.
  A removed member's last rows are still labelled with prices after its removal (no survivorship in the labels or the ranks).
* Repair rule for the 65 known defective bars: `high = max(high, low, close)`, `low = min(low, close, high)`, `open = NaN` if still outside the range; a non-positive price empties the bar.
  Bars on days outside the trading calendar are dropped. Both counts are in the manifest.
* The benchmark is VN30 (from 2020-05-11 only): `rs_*` and `dbeta_*` are NaN before then unless `features.benchmark_fallback_symbol` (e.g. `VNINDEX`) is set; it is rescaled to agree with VN30 on VN30's first date and never used afterwards. Forward fill is at most 5 sessions and only carries the last *known* value.
* All history is always loaded, so recursive features (RSI, ATR, MACD) do not depend on the `--start` you pass. `--start/--end` are clamped to the days that have data, so asking for "until today" on a Sunday describes the same dataset as Friday.
* Instruments blocked for insufficient history at the end date (`ingest.min_sessions`) are left out and listed, with the reason, in the manifest.

## Look-ahead audit

`features/audit.py`, run on **every** build (`datasets.audit_lookahead`): all features are recomputed on a panel with everything after a cut date removed;
every value at or before the cut must be identical (checked at 3 dates spread over the range). Labels are checked the other way: unchanged when data
beyond `t + horizon` is removed. The test suite proves the audit can fail: 7 deliberate leaks (one row ahead, two ahead, full-sample statistics, a
centred window, the series' last value, the benchmark's future, reverse expanding max) and 2 cross-sectional leaks are all caught.
What it cannot see: leaks that come with the data itself (back-adjusted price levels, see above) and a fundamentals provider that ignores `as_of`.

## Dataset manifest (`datasets.manifest`)

`config_hash` (recipe: universe, dates, cut-off, feature/label specs, instruments, options), `content_hash` (of the values), `inputs_hash` (of the bars used),
`file_sha256`, key / feature / label / `label_end` columns, rows, dates, instruments, symbols, NaN share per column, excluded instruments, policy counts,
benchmark coverage, audit result, library versions, git commit, run id. `datasets.sha256` is the file's sha256; `verify` / `load_dataset` refuse a file that no longer matches.

* Same recipe + same data → byte-identical file → the existing row is **reused** (nothing written).
* Same recipe + changed data (e.g. the vendor re-adjusted a bar) → **version + 1**; the old version and file are kept.
* `data_cutoff` (default: latest trading day) limits how much future data labels may see; no rows exist after it.

Fundamentals: `features.fundamentals.enabled` (default false) is the master switch. A `FundamentalProvider.as_of(when, ids)` must return only what was
published on or before `when`. A set that includes `fundamentals` fails with `FundamentalsDisabled` while the switch is off, so a price-only run can never
silently contain empty fundamental columns; the shipped sets contain none.
