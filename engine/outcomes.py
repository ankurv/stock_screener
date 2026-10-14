"""
Scores past recommendations for every enabled strategy once they're due for
evaluation - what "due" and "resolve" mean is strategy-specific (option
expiry passed and compared to strike for cash-secured puts, a configurable
evaluation_horizon_days elapsed and compared to entry price for dip-buy -
see engine/strategies/) but the loop here is generic.

Run once at server startup (see app.py) rather than as a scheduled job,
since "periodically" here just means "each time the server comes back up",
and a past recommendation's outcome never changes once computed.
"""

from __future__ import annotations

from common import get_logger
from engine.strategies import STRATEGIES

log = get_logger("outcomes")


def evaluate_past_recommendations(db, cfg: dict) -> dict:
    summary = {}
    for name, strategy in STRATEGIES.items():
        due = strategy.due_for_evaluation(db, cfg)
        strat_summary = {"evaluated": 0, "errors": []}
        for rec in due:
            try:
                result = strategy.resolve_outcome(db, rec, cfg)
            except Exception as exc:  # noqa: BLE001
                log.warning("Outcome resolution failed for %s %s: %s", name, rec.get("ticker"), exc)
                strat_summary["errors"].append(f"{rec.get('ticker')}: {exc}")
                continue
            if result is None:
                continue  # not enough data yet - try again next startup
            outcome, resolved_price = result
            strategy.record_outcome(db, rec, outcome, resolved_price)
            strat_summary["evaluated"] += 1
            strat_summary[outcome] = strat_summary.get(outcome, 0) + 1

        if strat_summary["evaluated"]:
            log.info("Evaluated %d past %s recommendation(s): %s", strat_summary["evaluated"], name, strat_summary)
        summary[name] = strat_summary
    return summary
