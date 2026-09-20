"""Paper trading and operations (Phase 8): the daily job, the paper portfolio (replayed from the recommendations by the same engine as the backtest), reports,
alerts, backups. Modes are `backtest` and `paper`; nothing in this package places a real order."""
MODES = ("backtest", "paper")


def check_mode(mode: str) -> str:
    """The only accepted modes. Asking for anything else (e.g. 'live') is an error, not a fallback."""
    if mode not in MODES:
        raise ValueError(f"unknown run mode {mode!r}: choose one of {MODES}. There is no live-trading mode: this system never places a real order.")
    return mode
