# Backtest engine

Code: `src/predict_stock/backtest/` — `market.py` (rules), `engine.py` (simulator), `portfolio.py`, `metrics.py`, `walkforward.py`,
`baselines.py`, `runner.py` / `job.py` / `report.py`. Results of the baselines: [BASELINES.md](BASELINES.md).

## One session, in order

```
0. time-stops that fall due become sell orders
1. orders decided at the close of the previous session execute — SELLS first (they free cash), then BUYS
     market: fill at this session's OPEN plus adverse slippage (rounded to the tick against you)
     limit : fills only if the session's low (buy) / high (sell) reaches it, at the better of the limit and the open; no slippage
2. stop / target on positions whose shares are sellable (bought >= settlement_days sessions ago)
     open beyond a level  -> exit at the OPEN (gap-through)
     both levels in one bar -> the STOP first (config backtest.tie)
     stop = market on trigger (slippage, never printed below the day's low); target = limit (no slippage)
3. mark to market at the close (a suspended instrument keeps its last close)
4. the strategy's signal for THIS close (data up to this session only) becomes orders for the next session
```
A signal is never filled in the session it is computed in (tested). Nothing is short (a negative weight raises).

## Vietnam rules (all from `config.market`, principle 8 — the defaults are the brief's, **not re-verified** against the rulebook / a fee schedule)

| Rule | Implementation |
|---|---|
| Lot 100 | buys are whole lots; a target smaller than one lot is not traded (`lot_too_small`); the exit sells everything |
| Tick size by price | 10 (<10,000), 50 (10,000–49,950), 100 (≥50,000); buys round **up**, sells **down**, ceiling **down**, floor **up** |
| Price band | ceiling/floor from the last traded close; buys at the ceiling and sells at the floor cannot fill (`limit_up_locked` / `limit_down_locked`); a locked session cannot be exited by a stop. The band per date is the known exchange's, else **inferred** (7/10/15%) from the previous 252 sessions of returns |
| Fee / tax / slippage | fee on both sides, tax on sells only, slippage against you; a run with all three at 0 is the "before costs" run; `scaled_costs(k)` gives the sensitivity runs |
| T+2 | shares are sellable from the session `settlement_days` after the buy; a sell for more than the sellable quantity sells what it can and waits (`t_plus_settlement`), also for full rebalances and time-stops; stops do not fire before then |
| Suspension / no fill | no bar or no valid open ⇒ nothing fills (`suspended_or_no_open`); buys are then dropped, sells persist (up to 20 sessions) |
| Cash | buys are limited by cash after fees (`not_enough_cash`); sale proceeds are reusable at once |

## Writing a strategy (what a model plugs into in Phase 5)

A strategy is a function from data known at a close to a `Signal`: `Signal(items=[SignalItem(instrument_id, weight, stop_pct, target_pct,
max_hold, order="open"|"limit", limit_offset, valid_sessions)], full_rebalance=True|False)`, stored in `{session_index: Signal}`.
`full_rebalance=True` sells every held instrument not listed (portfolio strategies); `False` leaves other positions to their exit rules (swing entries).
Stops/targets are fractions of the entry fill price; `max_hold` counts sessions. The scores must come from PIT features (Phase 3) — the engine cannot check that.

## Walk-forward and the held-out period

* `make_folds(sessions, scheme="expanding"|"rolling", train_sessions, test_sessions, step_sessions, embargo_sessions, end)`: an embargo gap of exactly that many
  sessions between train and test; nothing at or after `end` is used.
* `purge_train(frame, fold, label_end_col)`: drops a training sample whose label ends **on or after** the test start (a label ending on the first test session
  already used its data). The dataset's `*_end` columns are the input; the backtest report uses the latest end date of all a row's labels.
* The last `backtest.oos_months` (12) months are **held out**: development runs stop before them (`assert_development_only`), no fold reaches them.
  `python -m predict_stock backtest oos --final` evaluates them **once** for every baseline and records it in `experiments` (`oos:<id>`); a second call is refused
  (exit code 3) whatever the candidate. Do it once, for the final model and all baselines together. It has **not** been run.

## Metrics conventions

252 sessions a year; simple daily returns; CAGR over calendar days / 365.25; risk-free 0 by default; Sortino = mean / root-mean-square of negative returns (over all days);
MDD = deepest peak-to-trough; Calmar = CAGR / |MDD|; turnover = (buys + sells) / 2 / average equity / years; a trade = a closed position (first buy → full exit), win rate /
profit factor / expectancy over closed trades only (reported from 30 trades on). Sub-periods: consecutive 126-session windows labelled up / sideways / down by VNINDEX
(±10%) — ex-post, for reporting only.
