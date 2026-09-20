# INVEST B1 (1-3 months) — walk-forward 2021-05-14 → 2025-05-16

Universe **LARGE50** · run `d21025ab84346e04` · dataset `2bbbd6815f11` · horizon 63 sessions · rebalance monthly in 2 tranches · top 10, equal weight, cap 15% · held-out period 2025-09-19 → 2026-09-18 **untouched**. Prices only: no fundamental data is used.

## Verdict

**INVEST B1 (1-3 months) does not beat the baselines after costs.** The best candidate by net Sharpe is Factor score (rule-based): net Sharpe 0.32 (90% interval -0.49 … 1.22), CAGR 5.0%, max drawdown -48.5%; equal_weight over the same window: 0.49. 4 of 5 pre-registered criteria fail. The held-out period stays closed and nothing is tuned further.

| pre-registered criterion (for the best candidate) | value | threshold | detail | met |
| --- | --- | --- | --- | --- |
| net Sharpe above every portfolio baseline | 0.324 | 0.492 | best baseline: equal_weight | **no** |
| reality check p <= 0.1 (Sharpe vs equal-weight, 4 candidates) | 0.963 | 0.100 | best by the check: factor | **no** |
| rank IC t-statistic >= 2.0 | -0.249 | 2.000 | mean IC -0.0190 | **no** |
| share of rolling 1y windows ahead of equal-weight >= 0.6 | 0.322 | 0.600 | 748 overlapping windows | **no** |
| net Sharpe > 0 at 2x costs | 0.242 | 0.000 |  | yes |

## Results (net of costs unless stated; baselines re-run over the same window, engine and costs)

| strategy | CAGR | Sharpe | Sortino | max DD | Calmar | turnover ×/yr | CAGR gross | cost drag (points/yr) | Sharpe at 2× costs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Factor score (rule-based)** (best) | 5.0% | 0.32 | 0.43 | -49% | 0.10 | 3.1 | 7.5% | 2.5 | 0.24 |
| Ridge | 1.8% | 0.20 | 0.27 | -54% | 0.03 | 5.4 | 5.3% | 3.5 | 0.07 |
| ElasticNet | 1.5% | 0.19 | 0.26 | -54% | 0.03 | 5.4 | 5.3% | 3.8 | 0.08 |
| LightGBM (shallow) | -0.4% | 0.11 | 0.14 | -46% | -0.01 | 4.5 | 2.5% | 2.9 | 0.01 |
| Equal-weight universe | 9.2% | 0.49 | 0.65 | -49% | 0.19 | 0.2 | 9.6% | 0.4 | 0.49 |
| Momentum 6-12m | 6.0% | 0.35 | 0.46 | -57% | 0.10 | 3.1 | 7.8% | 1.8 | 0.27 |
| Momentum 6-12m inv-vol | 3.6% | 0.27 | 0.35 | -57% | 0.06 | 3.3 | 5.9% | 2.3 | 0.20 |
| Momentum 10d | 7.8% | 0.41 | 0.56 | -45% | 0.17 | 25.3 | 27.8% | 20.0 | -0.13 |
| Mean reversion | -26.8% | -1.12 | -1.43 | -78% | -0.35 | 27.8 | -12.6% | 14.2 | -1.75 |
| VNINDEX buy&hold | 0.6% | 0.13 | 0.17 | -40% | 0.01 | 0.0 | 0.7% | 0.1 | 0.12 |
| VN30 buy&hold | -0.0% | 0.10 | 0.14 | -42% | -0.00 | 0.0 | 0.1% | 0.1 | 0.10 |

Random top-10 portfolios on the same monthly schedule (20 seeds): net Sharpe median 0.24, 5th–95th percentile -0.07 … 0.42.

![equity](img/invest_b1_equity.png)

## Confidence intervals (stationary block bootstrap)

2000 resamples, mean block 21 sessions, 90% two-sided intervals, on 999 daily returns. Differences are computed on the same resampled days as the benchmark (paired).

| candidate | Sharpe | CAGR | max drawdown | Sharpe − equal-weight | resamples with a positive difference |
| --- | --- | --- | --- | --- | --- |
| Factor score (rule-based) | 0.32 [-0.49, 1.22] | 5.0% [-14.5%, 28.8%] | -48.5% [-65.2%, -24.7%] | -0.17 [-0.63, 0.29] | 29% |
| Ridge | 0.20 [-0.56, 1.04] | 1.8% [-18.9%, 26.3%] | -53.9% [-69.9%, -26.1%] | -0.29 [-0.60, -0.00] | 5% |
| ElasticNet | 0.19 [-0.58, 1.03] | 1.5% [-19.2%, 25.9%] | -54.3% [-70.2%, -25.9%] | -0.30 [-0.61, -0.00] | 5% |
| LightGBM (shallow) | 0.11 [-0.63, 0.93] | -0.4% [-18.3%, 20.8%] | -46.5% [-66.9%, -26.7%] | -0.38 [-0.82, 0.09] | 8% |

**White's reality check** over the 4 candidates against equal-weight (Sharpe difference): largest observed difference -0.17 (Factor score (rule-based)), **p = 0.963** — the probability of seeing a best-of-4 difference this large if none of them had any edge. Pre-registered threshold 0.1.

![forest](img/invest_b1_forest.png)

## Rolling windows

| candidate | window | windows | share ahead of equal-weight | median excess | worst excess | median return | share with a loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Factor score (rule-based) | 1y | 748 | 32% | -8.4% | -41.1% | 6.8% | 45% |
| Factor score (rule-based) | 3y | 244 | 0% | -20.4% | -34.2% | -7.0% | 76% |
| Ridge | 1y | 748 | 15% | -6.8% | -23.6% | 1.1% | 48% |
| Ridge | 3y | 244 | 0% | -20.3% | -29.8% | -8.8% | 69% |
| ElasticNet | 1y | 748 | 15% | -6.3% | -26.6% | 1.3% | 48% |
| ElasticNet | 3y | 244 | 0% | -21.5% | -31.3% | -8.6% | 75% |
| LightGBM (shallow) | 1y | 748 | 31% | -2.3% | -25.0% | 3.3% | 45% |
| LightGBM (shallow) | 3y | 244 | 9% | -11.6% | -32.4% | 2.7% | 44% |

* Factor score (rule-based) against Momentum 6-12m, 1-year windows: ahead in 55% of 748 (overlapping) windows, median excess 0.6%.
* Factor score (rule-based) against Momentum 6-12m, 3-year windows: ahead in 48% of 244 (overlapping) windows, median excess -0.5%.
* Factor score (rule-based) against VN30 buy&hold, 1-year windows: ahead in 59% of 748 (overlapping) windows, median excess 3.7%.
* Factor score (rule-based) against VN30 buy&hold, 3-year windows: ahead in 59% of 244 (overlapping) windows, median excess 3.9%.

Rolling windows overlap almost completely, so their number overstates the evidence; a 3-year window needs 3 years of out-of-sample data and there are only a few.

![rolling](img/invest_b1_rolling.png)

## Ranking quality

Daily cross-sectional Spearman IC between the score and the realised forward return over 63 sessions (only labels that end before the held-out period). t-statistic with an effective sample of days / 63.

| candidate | days | mean IC | std | ICIR | t-stat | days IC>0 |
| --- | --- | --- | --- | --- | --- | --- |
| Factor score (rule-based) | 1000 | -0.0190 | 0.303 | -0.06 | -0.25 | 50% |
| Ridge | 1000 | 0.0620 | 0.201 | 0.31 | 1.23 | 64% |
| ElasticNet | 1000 | 0.0608 | 0.200 | 0.30 | 1.21 | 64% |
| LightGBM (shallow) | 994 | -0.0542 | 0.218 | -0.25 | -0.99 | 39% |

Mean IC by fold:

| fold | test window | fit rows | val rows | LightGBM trees kept | Factor score (rule-based) | Ridge | ElasticNet | LightGBM (shallow) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2021-05-14 → 2021-11-08 | 14844 | 6084 | 1 | -0.126 | -0.032 | -0.033 | -0.097 |
| 1 | 2021-11-09 → 2022-05-13 | 20928 | 6116 | 50 | -0.032 | -0.049 | -0.049 | -0.049 |
| 2 | 2022-05-16 → 2022-11-08 | 27034 | 6126 | 1 | 0.011 | 0.120 | 0.117 | 0.031 |
| 3 | 2022-11-09 → 2023-05-15 | 33147 | 6250 | 5 | -0.393 | 0.179 | 0.177 | -0.287 |
| 4 | 2023-05-16 → 2023-11-08 | 39341 | 6250 | 1 | -0.069 | -0.017 | -0.018 | -0.097 |
| 5 | 2023-11-09 → 2024-05-15 | 45591 | 6250 | 14 | 0.254 | 0.163 | 0.162 | 0.165 |
| 6 | 2024-05-16 → 2024-11-08 | 51841 | 6250 | 80 | 0.235 | -0.130 | -0.131 | 0.005 |
| 7 | 2024-11-11 → 2025-05-16 | 58091 | 6250 | 3 | -0.031 | 0.262 | 0.262 | -0.106 |

In fold(s) 0, 2, 3, 4, 7 early stopping kept 5 trees or fewer: the shallow LightGBM found almost nothing on the validation window there, so its scores are close to constant (ties are broken by instrument id) and its portfolio in those windows is close to arbitrary.

Embargo = 63 sessions (the label horizon); samples purged by the end date of the 63-session label. The four candidates are compared on the same folds; the ridge / elastic-net penalty was chosen once on the first fold: ridge {"alpha": 1.0, "validation_ic": -0.17093}; elasticnet {"alpha": 0.0005, "l1_ratio": 0.2, "validation_ic": -0.17276}.

Grid caveat: the chosen penalty of ridge and elasticnet sits at the edge of the pre-registered grid; the best validation IC of ridge and elasticnet was not even positive, so the selection among grid points is mostly noise. The grid was fixed in advance and is not widened after the fact.

## Scenario quantiles (bear / base / bull) at the horizon

q10 / q50 / q90 of the 63-session forward return from three shallow quantile boosters (sorted so they never cross): observed share below each = 12.7% / 55.2% / 91.3% (nominal 10 / 50 / 90%); the q10–q90 band covers 78.6% (nominal 80%), mean width 52.3%. Pinball loss model / constant: q10 0.0372 / 0.0338, q50 0.0790 / 0.0742, q90 0.0498 / 0.0431. Overlapping labels make these intervals less reliable than the sample size suggests.

**The scenario quantiles do not beat a constant** at q10, q50, q90 (pinball loss above that of the pooled out-of-sample quantile, itself a yardstick that sees the whole sample): read the bear / base / bull figures as a rough spread of outcomes, not as a forecast.

## Variants of the best candidate (information only; nothing here chose the configuration)

| variant | CAGR | Sharpe | Sortino | max DD | Calmar | turnover | Sharpe gross | Sharpe at 2× costs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary: K = 10, equal, 2 tranches | 5.0% | 0.32 | 0.43 | -49% | 0.10 | 3.1 | 0.42 | 0.24 |
| K = 8 | 5.9% | 0.35 | 0.47 | -51% | 0.12 | 3.3 | 0.43 | 0.26 |
| K = 10 | 5.0% | 0.32 | 0.43 | -49% | 0.10 | 3.1 | 0.42 | 0.24 |
| K = 15 | 6.5% | 0.38 | 0.51 | -44% | 0.15 | 2.5 | 0.46 | 0.32 |
| weights: inverse_vol | 5.2% | 0.34 | 0.45 | -46% | 0.11 | 3.1 | 0.43 | 0.26 |
| weights: risk_parity | 4.4% | 0.31 | 0.41 | -48% | 0.09 | 3.2 | 0.41 | 0.24 |
| weights: min_variance | 1.7% | 0.19 | 0.25 | -50% | 0.03 | 3.7 | 0.29 | 0.09 |
| weights: hrp | 4.8% | 0.32 | 0.43 | -47% | 0.10 | 3.6 | 0.43 | 0.24 |
| 1 tranche(s) | 6.1% | 0.37 | 0.49 | -48% | 0.13 | 3.1 | 0.44 | 0.30 |
| 3 tranche(s) | 4.9% | 0.32 | 0.43 | -48% | 0.10 | 3.1 | 0.40 | 0.25 |
| regime filter (index < SMA200 ⇒ 50% of the stock weights, rest cash) | 8.3% | 0.49 | 0.66 | -38% | 0.22 | 2.8 | 0.60 | 0.42 |

These variants differ by amounts of the same size as the bootstrap intervals above; a difference between two rows is not evidence that one setting is better.

The index was below its 200-day average on 37% of the sessions in the window (VN30, or VNINDEX where VN30 has no 200-day average yet).

**Periodic contribution (DCA)** of 10 M VND on the first session of each month, invested at each strategy's own daily return (an approximation: no lot rounding on the marginal contribution; costs are inside the returns):

| portfolio | paid in | final value | gain | money-weighted return / yr |
| --- | --- | --- | --- | --- |
| best candidate | 490 M | 544 M | 54 M | 5.3% |
| equal-weight | 490 M | 566 M | 76 M | 7.3% |
| VN30 buy&hold | 490 M | 535 M | 45 M | 4.4% |

## Signal outputs and thesis-break conditions

Each stored signal carries its score, the bear / base / bull quantiles, the horizon, the factor contributions and three thesis-break flags: close below the 200-day average; relative strength (rank of 6-month momentum in the universe) below 0.4; drawdown from the 252-session high deeper than 15%. Among the 490 holdings the best candidate would have taken at rebalance dates, the forward 63-session return was:

| flag | flagged: n | flagged: mean return | flagged: share negative | clear: n | clear: mean return | clear: share negative |
| --- | --- | --- | --- | --- | --- | --- |
| below SMA200 | 43 | -0.5% | 47% | 447 | 2.3% | 45% |
| weak relative strength | 5 | 2.2% | 40% | 485 | 2.1% | 45% |
| deep drawdown | 73 | -0.9% | 45% | 417 | 2.6% | 45% |
| any flag | 83 | -1.1% | 47% | 407 | 2.7% | 45% |

Holdings overlap from one rebalance to the next, so these are descriptive, not independent tests. The flags are an output for the reader; they are not a trading rule in the backtest.

Stored signals (`predictions.details`, latest rebalance date of the last fold model, best candidate):

* 2025-05-05 · SHB · rank 1 · score 0.888 · bear/base/bull -18% / 3% / 31% over 63 sessions · thesis-break triggers: none · contributions: mom_6m_csrank=0.98 → +0.096; mom_12m_csrank=0.98 → +0.096; sma_ratio_200_csrank=0.92 → +0.084; dd_252_csrank=0.88 → +0.076
* 2025-05-05 · TCB · rank 2 · score 0.860 · bear/base/bull -17% / 3% / 24% over 63 sessions · thesis-break triggers: none · contributions: mom_12m_csrank=0.94 → +0.088; mom_6m_csrank=0.88 → +0.076; sma_ratio_200_csrank=0.86 → +0.072; dd_252_csrank=0.84 → +0.068

## Fundamentals

no fundamental columns in the dataset: the INVEST models ran on prices only. The pipeline is ready for them: if the dataset gains columns starting with `fund_`, `invest run` re-fits the learned candidates with and without them and this section reports the difference in out-of-sample rank IC.

## Reproducibility and records

Models (`invest_b1_<candidate>_fNN` per fold, `_final` on all development data) are registered in `models` with status `candidate`; 9,760 predictions (rebalance dates only) are in `predictions` with scenario quantiles, horizon, thesis flags and contributions in `details`. The hyper-parameter grid (one row per point) is in `experiments` (`invest:grid:55e8225afef75723:NNN`), the pre-registration in `invest:preregistration`. Re-running with unchanged data and configuration reproduces identical model files and reuses the rows.

## Assumptions and limits

* Everything in [BASELINES.md](BASELINES.md) applies: universe look-ahead/survivorship (LARGE50 = today's 50 most traded names), dividend-adjusted vendor prices, inferred price bands, T+2 for the whole period, brief-default costs (not re-verified), no market impact.
* **Small sample.** About 4 years of out-of-sample returns, up to 50 names, one path of history, and labels of 63 / 126 sessions that overlap almost completely: the number of *independent* observations is a handful. The intervals above are the honest way to read the point estimates.
* The comparison window starts at the first test session (≥ 500 training sessions + an embargo equal to the horizon), later than the Phase 4 baselines' window; the baselines were re-run over it.
* The regime filter uses VN30 only from when it has a 200-day average (VN30 exists from 2020-05); before that VNINDEX.
* DCA is a return-series approximation (marginal contributions are assumed to earn the strategy's return, without lot rounding).
* Forward-return labels that would end inside the held-out period are excluded from every IC / quantile figure; the portfolio backtests stop at the last development session. The held-out period was not used for any choice.
