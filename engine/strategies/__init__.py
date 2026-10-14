"""
Strategy registry. run_pipeline.py, app.py, and engine/outcomes.py only ever
reach into a strategy through STRATEGIES and the handful of functions each
strategy module exposes - generate, save, get_latest, get_history,
due_for_evaluation, resolve_outcome, record_outcome - never a strategy's
internals directly. Adding a new strategy means dropping a module here with
those functions, registering it below, and adding its key to config.yaml's
strategies_enabled/strategies - nothing in the shared pipeline/outcomes/app
code needs to change.
"""

from __future__ import annotations

from engine.strategies import cash_secured_put, dip_buy, three_day_rule

STRATEGIES = {
    cash_secured_put.KEY: cash_secured_put,
    dip_buy.KEY: dip_buy,
    three_day_rule.KEY: three_day_rule,
}
