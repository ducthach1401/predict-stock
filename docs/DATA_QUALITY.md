# Data quality report — as of 2026-09-20

Universes: LARGE50 · instruments checked: 52 · daily bars: 111,037 · range: 2018-01-02 → 2026-09-18

## Verdict: **PASS**

* Unexplained errors: **0** — every error-level finding is either absent or covered by a known issue.
* Open warnings: 262 · explained findings: 65 · instruments blocked for insufficient history: 0

## Findings by check

| check | severity | open | explained |
| --- | --- | --- | --- |
| ohlc_inconsistent | error | 0 | 65 |
| big_move | warn | 109 | 0 |
| extra_day | warn | 10 | 0 |
| missing_trading_day | warn | 120 | 0 |
| repeated_bar | warn | 1 | 0 |
| return_outlier | warn | 6 | 0 |
| volume_spike | warn | 14 | 0 |
| zero_volume | warn | 2 | 0 |

## Unexplained findings

No unexplained errors.

**Warnings (not explained; they do not block):**

| check | findings | symbols | examples |
| --- | --- | --- | --- |
| big_move | 109 | 13 | ACB 2018-04-26; ACB 2018-05-28; ACB 2018-05-29; ACB 2018-07-06; ACB 2020-02-12; ACB 2020-03-23 … |
| extra_day | 10 | 5 | ACB 2018-01-23; ACB 2018-01-24; LPB 2018-01-23; LPB 2018-01-24; SHB 2018-01-23; SHB 2018-01-24 … |
| missing_trading_day | 120 | 11 | ACB 2020-12-02; ACB 2020-12-03; ACB 2020-12-04; ACB 2020-12-07; ACB 2020-12-08; GEX 2018-01-15 … |
| repeated_bar | 1 | 1 | VHM 2018-05-21 |
| return_outlier | 6 | 3 | ACB 2018-05-28; ACB 2020-03-23; LPB 2020-03-09; LPB 2020-03-23; VIB 2020-03-09; VIB 2020-03-23 |
| volume_spike | 14 | 6 | EIB 2018-12-04; EIB 2018-12-06; EIB 2018-12-14; EIB 2019-06-10; EIB 2022-11-17; NVL 2022-11-22 … |
| zero_volume | 2 | 1 | VHM 2018-05-18; VHM 2018-05-21 |

## Explained findings (known issues)

| key | check | findings | treatment | explanation |
| --- | --- | --- | --- | --- |
| ohlc-2019-vendor-defect-days | ohlc_inconsistent | 57 | keep_flagged_clip_envelope_downstream | Vendor OHLC defect on specific days of Jul-Sep 2019: the bar's OPEN lies outside the bar's [low, high] (the close never does). Root cause not confirmed; since only the open is affected, a possible mechanism is the opening-auction (ATO) price not being included in the day's high/low. Raw values are kept; consumers must not trust `open` on these bars and must clip high/low to the envelope. *Evidence:* 57 stock bars in 34 instruments; 53 fall on five dates (2019-07-15 x16, 07-19 x4, 08-07 x15, 08-30 x4, 09-04 x14; the other 4 are 07-18 x2 and 08-09 x2), each hitting many unrelated symbols at once, so it is a feed/day-level defect and not a per-stock error. Open outside [low, high] in all 57 bars, close outside in none; violation median 0.84%, p90 2.9%, max 6.5% of close. |
| ohlc-2019-vnindex-open | ohlc_inconsistent | 3 | exclude_open | VNINDEX bars of three consecutive days carry the same open (949.48), which is impossible; the index open is unreliable on these days. Root cause not identified. *Evidence:* open = 949.48 on 2019-06-24, 06-25 and 06-26 while high/low/close move; open lies outside [low, high] on all three. |
| ohlc-isolated-lpb-2020-02-18 | ohlc_inconsistent | 1 | keep_flagged_clip_envelope_downstream | Isolated bar: close 2690 below low 2730 (1.5% of close). Root cause not identified. *Evidence:* same day as the VIB bar below but different symbols; no wider cluster. |
| ohlc-isolated-pvd-2022-09-12 | ohlc_inconsistent | 1 | keep_flagged_clip_envelope_downstream | Isolated bar: open 12190 below low 12360 (1.4% of close). Root cause not identified. *Evidence:* single bar; the only stock OHLC defect after 2020-02-18. |
| ohlc-isolated-vci-2018-01-16 | ohlc_inconsistent | 1 | keep_flagged_clip_envelope_downstream | Isolated bar: open 10260 above high 10200 (0.6% of close). Root cause not identified. *Evidence:* single bar; no other symbol affected that day. |
| ohlc-isolated-vib-2019-12-31 | ohlc_inconsistent | 1 | keep_flagged_clip_envelope_downstream | Isolated bar: close 3490 below low 3510 (0.57% of close). Root cause not identified. *Evidence:* single bar; no other symbol affected that day. |
| ohlc-isolated-vib-2020-02-18 | ohlc_inconsistent | 1 | keep_flagged_clip_envelope_downstream | Isolated bar: close 3620 below low 3690 (1.9% of close). Root cause not identified. *Evidence:* same day as the LPB bar above; no wider cluster. |

## History coverage

Sessions per instrument: min 1428, median 2171, max 2171 (`ingest.min_sessions` = 500).

Shortest histories:

| symbol | sessions | first bar | last bar |
| --- | --- | --- | --- |
| MSB | 1428 | 2020-12-23 | 2026-09-18 |
| VN30 | 1556 | 2020-05-11 | 2026-09-18 |
| TCB | 2072 | 2018-06-04 | 2026-09-18 |
| VHM | 2084 | 2018-05-17 | 2026-09-18 |
| FRT | 2097 | 2018-04-26 | 2026-09-18 |
| TPB | 2101 | 2018-04-19 | 2026-09-18 |
| GVR | 2116 | 2018-03-21 | 2026-09-18 |
| POW | 2124 | 2018-03-06 | 2026-09-18 |

Blocked (insufficient history):

_none_

## Trading calendar

2,171 observed trading days, 2018-01-02 → 2026-09-18 (stock consensus; DNSE's official `get_working_dates` needs an API key, so it is not used). Future holidays are unknown.

| year | trading days | weekday gaps |
| --- | --- | --- |
| 2018 | 248 | 10 |
| 2019 | 250 | 11 |
| 2020 | 252 | 10 |
| 2021 | 250 | 11 |
| 2022 | 249 | 11 |
| 2023 | 249 | 11 |
| 2024 | 250 | 12 |
| 2025 | 249 | 12 |
| 2026 | 174 | 13 |

Weekdays outside the calendar on which some stocks do have bars (below the consensus): 2018-01-23 (5), 2018-01-24 (5)

## Price adjustment

Price basis of stored bars: vendor_adjusted = 111,037. DNSE serves back-adjusted prices (inferred in Phase 0: no gap beyond the price band that an unadjusted stock dividend or split would create), so no factors are applied. Adjustment candidates are only ever *reported*: none is applied without a confirmed file.

Corporate-action rows: stale = 1 · superseded bar revisions kept: 0.

| symbol | ex/gap date | type | status | details |
| --- | --- | --- | --- | --- |
| GVR | 2020-03-17 | unknown_gap | stale | band=0.15, gap=-0.160584, implied_factor=0.839416, open=9200.0, prev_close=10960.0 |
