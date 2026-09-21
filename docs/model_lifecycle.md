# Model lifecycle: monitoring, retraining, champion / challenger

The rule of this phase in one sentence: **a model is replaced only when a challenger, trained with exactly the current protocol and watched in shadow on data that appeared after both were trained, beats the champion by more than luck, and a person confirms it.** Nothing here places an order and nothing here decides by itself to switch a model.

## 1. What exists

| piece | where | commands |
|---|---|---|
| Registry: status `candidate → shadow → champion → retired` per strategy (`swing`, `invest_b1`, `invest_b2`), one champion per strategy, every change logged with actor and reason | `models`, `model_status_log`, `lifecycle/registry.py` | `lifecycle status`, `lifecycle history`, `reject`, `rollback` |
| Provenance of every recommendation: recommendation → prediction → model (file + sha256 + seed + training cut-off) → training dataset → feature set / label spec → git commit → the price data **as it was known on the day** (a fingerprint that survives later vendor re-adjustments through `price_bar_revisions`) | `recommendations.provenance`, `lifecycle/provenance.py` | `lifecycle trace --recommendation-id N` |
| Monitoring: PSI / KS per feature, rolling rank IC, hit rate, calibration error, paper fill rate, deviation between expected and actual holding time; stored in `monitoring_metrics`, alerts in `alerts` | `lifecycle/monitor.py`, `lifecycle/reference.py` | `lifecycle monitor`; runs inside `paper run` (steps `shadow`, `monitor`) |
| Retrain triggers and the retrain itself | `lifecycle/retrain.py` | `lifecycle due`, `retrain --strategy swing|invest|invest_b1|invest_b2 --trigger schedule|drift|manual` |
| Shadow: the challenger scores the universe every day, **predictions only** (no card, no order, no position) | `lifecycle/shadow.py` | `lifecycle shadow` (also part of `paper run`) |
| Comparison with the pre-registered rule, promotion, rollback | `lifecycle/compare.py` | `lifecycle compare --model-id X`, `promote --model-id X`, `rollback --strategy S --reason "..."` |
| What the closed recommendations say; recalibration proposal; meta-labeling trial | `lifecycle/outcomes.py` | `lifecycle assess`, `lifecycle recalibrate`, `lifecycle meta-label` |
| Monthly evaluation: retrains that are due and why, every shadow comparison, outcomes | `lifecycle/job.py` | `lifecycle evaluate`; runs on the first session of each month inside `paper run` |

`make lifecycle-status lifecycle-monitor lifecycle-due lifecycle-evaluate`.

Existing models were migrated by `0007`: the latest `swing_lgbm_final`, `invest_b1_factor_final` and `invest_b2_factor_final` are the champions (they are what the cards used); the superseded versions are `retired`. Cards issued before this phase have no stored provenance: `trace` rebuilds the chain from the model row and says so (no data fingerprint).

## 2. Monitoring

| metric (`monitoring_metrics.metric_name`) | what | alert when |
|---|---|---|
| `psi:<feature>`, `ks:<feature>`, `psi_max`, `drift_features_alert` | the last 60 sessions against the training rows of the model (quantile reference stored in `models.params.reference`) | at least `psi_features_trigger` (3) features beyond their **own yardstick** — see below |
| `rolling_ic`, `hit_rate` | rank IC of the model score against the realised 5- / 63- / 126-session return, on paper-record sessions only | rolling IC below the model's out-of-sample IC by more than `ic_z` standard errors (or below 0 when it had none), with at least `ic_min_days` days |
| `calibration_gap`, `ece` | stated SWING probability against the realised event rate | `|gap| > calibration_gap` with at least `calibration_min_events` events |
| `fill_rate` | share of BUY cards that were filled (backtest of the cards: 71%) | below `fill_rate_min` with at least `fill_min_orders` orders |
| `hold_dev` | actual − stated holding time, sessions, mean | beyond `hold_dev_sessions` with at least `hold_min_trades` trades |
| `universe_changes_90d` | names that joined or left in 90 days | at least `universe_changes_trigger` |

All thresholds are in `config/default.yaml` (`lifecycle.thresholds`), none in code.

**The drift yardstick is calibrated on the training period itself.** A fixed PSI limit (0.25) was tried first and was useless: the 50 stocks share one market state, so a 60-session slice holds far fewer independent observations than its 3,000 rows, and level features (12-month momentum, distance to the 252-day high, SMA ratios) move as a block with the market. Measured on the real data with a reference ending 2023-12: for INVEST 8–9 of 15 features were beyond 0.25 in almost every 60-session window of 2024–2026, i.e. the alert would have fired every day. Now each feature carries the 95th percentile of the PSI / KS of the training period's own 60-session slices (`psi_ref`, `ks_ref`), and a feature is flagged only beyond `max(config floor, that yardstick)`. With that yardstick and the same 2023-12 reference, INVEST flagged ≥ 3 features in 0 of 31 windows and SWING in 8 of 31 (April 2024, Nov 2024 – Mar 2025, Aug–Sep 2025; I did not investigate what was different in those stretches). With a reference two years old the SWING alert fires in about a quarter of the windows, which is what a stale model should look like. A drift alert means "unlike anything in the training period", not "the market moved".

**The held-out year is never used to judge a champion.** The research held out 2025-09-19 → 2026-09-18 (`lifecycle.protected_period`) and never looked at it. Feature drift needs no returns, so it is computed on any dates. IC, hit rate and calibration need realised outcomes, so they are computed **only from the paper record** (`paper.start_date`, 2026-09-21 onwards). `compare` refuses a window that touches the protected period unless `holdout:consumed` was recorded (section 4).

## 3. When a retrain is due (`lifecycle due`, `lifecycle evaluate`)

A trigger only says that a retrain is **due**; it starts nothing (`lifecycle.auto_retrain: false`). Reasons: `schedule` (the champion's training data is at least `retrain_months` old; default 6, allowed 3–12), `drift` (section 2), `ic` (rolling IC decay), `calibration`, `universe` (many changes), `new_feature_set` (the config names a feature set the champion was not trained on: adding a feature makes the champion stale by definition), `manual`. Fill rate and holding-time deviation alert but are not retrain reasons (they say the cards' rules, not the model, need a look).

On the real database today: all three champions are due by `schedule` (training data ends 2025-09-04, 2025-06-19 and 2025-03-18), no drift, and no retrain can be run without consuming the held-out year (section 4).

## 4. Retrain → shadow → compare → promote

```
python -m predict_stock retrain --strategy swing --trigger schedule
python -m predict_stock lifecycle shadow          # also done every day by `paper run`
python -m predict_stock lifecycle compare --model-id N
python -m predict_stock promote --model-id N --note "reviewed the comparison"
python -m predict_stock rollback --strategy swing --reason "..."
```

**Retrain** trains a challenger with the protocol in force: same folds, purging and embargo (`fit_final`), the same hyper-parameter search on the first fold (an earlier study with an identical first fold is reused, not repeated), same candidates and label spec. It stores `protocol` (a hash of that protocol) in the model's params, registers the model as `candidate`, moves it to `shadow`, and writes one `experiments` row `lifecycle:retrain:<strategy>` per challenger. INVEST retrains the fitted candidates (ridge, elastic net, LightGBM); the `factor` candidate has no fitted parameter, so retraining it changes nothing and is skipped. Only one challenger per strategy at a time.

**Guard against reusing the held-out year.** Training data that reaches into `protected_period` uses that year up as a test set for good (it can no longer be an honest test for anything trained before). `retrain` therefore refuses, unless `--consume-holdout` is given; the act is then recorded once as the experiment `holdout:consumed`, and only then may monitoring and comparison touch that period. Nothing does this automatically. Because the champions stop in 2025, **the first retrain on the real system necessarily consumes the held-out year**: do it deliberately, once, after deciding that fresh training data is worth more than keeping that year as a reserve (the paper record from 2026-09-21 is the new, untouched out-of-sample stretch).

**Compare** (pre-registered in `config/default.yaml` → `lifecycle.promotion`; each comparison is written to `experiments` as `lifecycle:compare:<id>` with the rule that was applied):

| criterion | rule |
|---|---|
| same protocol | the challenger's stored protocol hash equals the current one |
| enough shadow time and data | `shadow_min_weeks` (swing 8, invest_b1 21, invest_b2 33: at least the label horizon plus 8 weeks) and `min_days` sessions with a realised label |
| window | only sessions **after the later of the two training cut-offs**; both models are read from their **stored** predictions (the challenger's were written on the day) |
| better | **rank IC OR net-of-cost Sharpe**: the lower bound of a paired block-bootstrap of the difference must be above 0 at one-sided level `alpha / k` |
| not worse on drawdown | challenger max drawdown ≥ champion's − `mdd_tolerance` (2 points) |
| not worse on calibration | challenger ECE ≤ champion's + `ece_tolerance` (SWING probability) |
| person | `promote --model-id X` re-runs the comparison at that moment and refuses unless it passes; there is no override |

The Sharpe comes from the research's own portfolio construction (SWING top-K weekly; INVEST top-K on the preset's schedule) built from each model's stored scores through the same engine, costs included.

**Multiple testing.** `k` = the number of challengers already tried against the same champion (`experiments` rows `lifecycle:retrain:<strategy>` since it became champion). The more challengers are tried, the smaller the level `alpha / k` and the more the challenger has to win by. A challenger that is only a different random draw of the same skill will show a nonzero point difference; the bootstrap keeps it from replacing the champion (test `..._merely_a_different_random_draw_...`).

**Rollback** puts the champion that the current one replaced back in charge (the current one is retired), logged.

## 5. What the closed recommendations are used for

* `lifecycle assess`: win rate, share reaching target 2, net return, calibration (with at least `recalibration_min_events` = 300 events; otherwise it says it computes none), expected vs actual holding time. The event is "target 2 reached after the fill" (a paper trade), not the model's own event measured from the signal close — the note says so.
* `lifecycle recalibrate`: a **proposal**. An isotonic map is fitted on the earlier 70% of the closed trades and judged (Brier, ECE) on the later 30%; together with a holding-time scale (median actual ÷ stated). It is never applied by itself; a proposal that improves the later trades is meant to be built into the **next** model version, which then passes the challenger process.
* `lifecycle meta-label`: a regularised logistic model as a secondary filter ("will this SWING trade end with a net gain?") from what the card knew when issued (probability, ATR %, R:R, expected hold, RSI, volume spike, median return forecast, regime, style). Trained on the earliest 70% of trades, threshold fixed from the training scores (keep the top 60%, not tuned), judged **once per attempt** on the later 30%; the verdict is the bootstrap lower bound of (mean net return of kept trades − mean of all trades) > 0. Needs at least `meta_min_trades` = 300 trades. Every attempt is an `experiments` row `lifecycle:meta_label`, so the number of tries is visible. A filter that passes is only a candidate feature of a future model.
* First result on the Phase-7 backtest trades (856 SWING trades, 600 to train, 256 later ones from 2024-03-07): the filter kept 152 of them; mean net return of kept trades −0.02% against +0.16% for all trades; gain −0.18 points with a bootstrap lower bound of −0.47 points → **discarded**. The paper record has too few closed trades yet to try again (experiment #117 is the only attempt so far).

## 6. How to add a feature or a model

**A feature**: register it in `features/` (a new version of the feature set — never edit a version in place), `features sync`, rebuild datasets. The config now names a feature set the champion was not trained on, so `lifecycle due` reports `new_feature_set`. Then `retrain` trains a challenger on it (same protocol), it shadows for the required weeks, and it becomes champion only through `compare` and `promote`. The old champion stays available for `rollback`.

**A model family** (a new candidate): implement it beside the existing candidates (`invest/models.py`, `swing/model.py`), add it to `invest.candidates` / register it under a new name (`<strategy>_<name>_final`); it enters through `retrain` as a challenger. The number of tries against the champion goes up by one (multiple testing).

**A label or protocol change** (folds, horizon, label spec) changes `protocol_hash`: models trained before are no longer comparable and the new protocol has to be run through the research phase (`swing run` / `invest run`) again, not smuggled in through `retrain`.

**Never**: choose a model on the protected period, pick the best of several challengers by their shadow numbers (they all count as attempts), lower `alpha` or the shadow time after seeing a comparison.

## 7. Limits

* No challenger has been trained on the real system yet: doing so needs `--consume-holdout` (section 4). The retrain, shadow, compare and promote paths are exercised on simulated data (`tests/test_lifecycle.py`, `tests/test_lifecycle_e2e.py`), where a worse challenger is rejected and a better one passes; the simulated "skilled" model is built from future returns, so those runs show the mechanics and the guards, not any expected real-world gain.
* Rolling IC, hit rate, calibration, fill rate and holding-time deviation are empty until closed recommendations exist (the paper record starts 2026-09-21; a SWING trade lasts 3–15 sessions, an INVEST horizon is 63 / 126 sessions). Until then the monitoring that works is feature drift, universe turnover and the model age.
* The drift yardstick depends on the training period containing a variety of regimes; a training period with one regime makes the yardstick narrow and the alert sensitive.
* The retrain reproduces the protocol's tuning-and-final-fit stage. The walk-forward IC table of the research is not recomputed by `retrain`; the challenger's out-of-sample evidence is its shadow period, which is the only evidence that counts here.
* The shadow start is the day the challenger is registered; earlier days are not back-filled unless `lifecycle shadow --replay MODEL_ID --start .. --end ..` is run, and such a replay is only a convenience for catching up: it uses the model file and the dataset rows of that date, but the weeks it adds count as shadow weeks only if the days are after both training cut-offs.
