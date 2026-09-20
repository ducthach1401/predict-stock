# DNSE market-data API — what was verified, and what is only assumed

Probed on 2026-09-20. All assumptions live in `src/predict_stock/data/dnse_client.py`.
Nothing below comes from DNSE documentation: the endpoint used is **not documented**
in any page I could read (see "Official sources" at the bottom).

## Endpoint used (public, no key)

```
GET https://api.dnse.com.vn/chart-api/v2/ohlcs/{stock|index}
    ?symbol=VCB&resolution=1D&from=<epoch s>&to=<epoch s>
```
`https://services.entrade.com.vn/chart-api/v2/ohlcs/...` returned identical data.

| Item | Observation | Status |
|---|---|---|
| Auth | none; plain GET works | verified |
| Response | parallel arrays `t,o,h,l,c,v` + `nextTime` (always `0` so far) | verified |
| `t` | epoch seconds. Stock daily bars: 09:00 ICT (02:00 UTC). Index daily bars: 09:15 ICT | verified |
| Stock prices | thousand VND (`60.33` = 60,330 VND) — converted in the client | verified |
| Index values | points, not scaled | verified |
| `v` | share count (index: traded value-like large number, not used) | verified for stocks; index meaning **unknown** |
| Resolutions | `1D` and `1H` return data; others untested. **1H depth: from 2023-09-21 only** (3,721 bars for VCB from any earlier `from`); a 1H bar's `t` is its start: 09:00, 10:00, 11:00, 13:00, 14:00 ICT (no 12:00 bar, lunch break) | verified |
| History depth | stocks from 2018-01-02 (VHM 2018-05-17); VNINDEX 2018-01-02; **VN30 index from 2020-05-11** | verified |
| Invalid symbol | HTTP 400 `{"code":"BAD_REQUEST","message":"invalid symbol"}` | verified |
| Rate limit | 60 sequential requests, no delay (~6 req/s) all HTTP 200; no rate-limit headers. **True limit unknown** — client throttles to ≥0.2 s between requests and retries 429/5xx | unknown |
| Price adjustment | history looks **back-adjusted**: no gap > 7.5 % in 8 years for 8 large caps incl. HPG/SSI which had stock dividends. Inferred from data, not stated by DNSE | inferred |
| Volume adjustment | **unknown** whether volume is also adjusted | unknown |
| Index constituents | **no endpoint found**; DNSE docs mention only index *data* (`plaintext/quotes/index/MI/{marketID}`) | not available |

## Quirks found in real data

1. **Split-session rows (2022-12-27).** Every stock returns two rows for this date: one stamped
   00:00 UTC with small volume (VCB: 136,100) and one at 02:00 UTC whose open equals the first
   row's close. The client merges them (open = first, high = max, low = min, close = last,
   volume = sum) and reports each merge in `client_warnings`. **This is an inference** — the
   merged HPG bar opens at 11,570 = previous close and its 32M volume is in the normal range —
   but it was not confirmed with DNSE or HOSE. Seen on this one date only, across all 45 stocks that had it in the 50-stock load.
2. **VNINDEX and VN30 series miss 33 real trading days** (2020: 16, 2021: 15, plus 2 later),
   days on which all five stocks have bars. Hence the trading calendar is derived from stock
   consensus (`ingest.calendar_min_stock_fraction`), never from an index series.
3. **OHLC-inconsistent bars from the source.** 62 stock bars across 36 of the 50 stocks (57 of them
   in Jul–Sep 2019; in 59 of 62 the `open` is outside `[low, high]`). VNINDEX 2019-06-24/25/26 all have
   `open = 949.48`, which is clearly wrong. These are **reported, not repaired**
   (`python -m predict_stock quality`).

4. **Exchange-dependent behaviour.** Seven symbols show many 9–10 % (and some ~15 %) daily moves and
   five have bars on 2018-01-23/24 when HOSE stocks do not. This fits an earlier listing on HNX/UPCoM
   (bands ±10 % / ±15 %), not unadjusted corporate actions; an inference — no exchange history is provided.

5. **Calendar.** DNSE's official `get_working_dates` needs an API key, so `trading_calendar` is observed
   (stock consensus, `ingest.calendar_min_stock_fraction`, with `calendar_grace_days` so that one stock running
   ahead of the others cannot create a trading day — a bug the tests caught). 8.7 years give 248-252 trading
   days and 10-13 weekday gaps per year, plausible for Vietnamese holidays but **not checked against the official
   holiday list**. Future holidays are unknown.
6. **Gap detection needs the number of sessions between bars.** The first version of the corporate-action detector
   flagged GVR 2020-03-17 (-16% open vs previous close). GVR simply has no bars for the five sessions before
   (each may move ±15%), so it was a data gap, not an event; the detector now compounds the price band over the
   sessions in between. Result on the 50 stocks: no candidate.

## Consequences for later phases

* **Adjusted prices are not point-in-time.** Back-adjustment bakes later corporate actions
  into earlier prices. Returns and ratios are fine; anything that needs the *actual* price on
  the day — tick size, ±7 % band, lot value, price-level features — must not use these
  values without an unadjusted source. `price_bar.price_basis` records this ('vendor_adjusted' today).
  The ingest job detects re-adjustment (stored close vs. fresh close beyond `adjust_rel_tolerance`) and
  rewrites the whole instrument; every overwritten row is first copied to `price_bar_revisions`, so earlier
  versions of the series are kept. `corporate_actions` / `adjustment_factors` exist but stay empty: DNSE offers
  no corporate-action data.
* The endpoint is unofficial and may change or throttle without notice.

## Official sources consulted

* SDK `dnse-sdk-openapi` (PyPI): host `openapi.dnse.com.vn`, **requires api_key/api_secret** for
  every market-data method (`get_ohlc`, `get_instruments`, ...). Not used.
* hdsd.dnse.com.vn LightSpeed pages: registration overview and WebSocket topics; no REST
  history endpoint or constituent list. `developers.dnse.com.vn` (linked from there) could not
  be read from this environment.
