"""SWING strategy (Phase 5): a LightGBM model that ranks the universe, estimates the probability that the target is hit before
the stop (calibrated), the q10/q50/q90 of the forward return and the expected holding time, evaluated by walk-forward against the
Phase 4 baselines through the same backtest engine. See docs/SWING.md."""
