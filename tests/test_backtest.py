"""
tests/test_backtest.py
----------------------
Unit tests for LiveBacktester.compute_metrics().
These tests do NOT require the model or internet — they work on
synthetic backtest result DataFrames.

Run with:
    pytest tests/test_backtest.py -v

Location: F:/commodity_trading_project/tests/test_backtest.py
"""

import sys
import os
import types
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub heavy dependencies so tests run without TensorFlow installed
# (CI installs TF; locally you already have it — this is just a safety net)
for _mod in ["tensorflow", "tensorflow.keras"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Provide minimal tf.keras.models stub
_tf = sys.modules["tensorflow"]
if not hasattr(_tf, "keras"):
    _keras = types.ModuleType("tensorflow.keras")
    _models = types.ModuleType("tensorflow.keras.models")
    _models.load_model = lambda *a, **kw: None
    _keras.models = _models
    _tf.keras = _keras

from src.backtest.engine import LiveBacktester


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def profitable_results() -> pd.DataFrame:
    """
    Synthetic backtest results where the strategy makes money.
    Strategy returns are a small positive constant — easy to reason about.
    """
    n = 200
    strategy_ret = np.full(n, 0.001)   # +0.1% per candle
    market_ret   = np.full(n, 0.0005)  # +0.05% per candle

    cum_strategy = np.cumprod(1 + strategy_ret)
    cum_market   = np.cumprod(1 + market_ret)

    return pd.DataFrame({
        "Timestamp":         pd.date_range("2026-01-01", periods=n, freq="5min"),
        "Price":             np.linspace(75, 80, n),
        "Prediction":        np.full(n, 0.65),
        "Signal":            np.ones(n),
        "Next_Ret":          market_ret,
        "Strategy_Ret":      strategy_ret,
        "Cum_Market_Ret":    cum_market,
        "Cum_Strategy_Ret":  cum_strategy,
        "Equity":            10_000 * cum_strategy,
    })


@pytest.fixture
def losing_results() -> pd.DataFrame:
    """Synthetic backtest where strategy loses money consistently."""
    n = 200
    strategy_ret = np.full(n, -0.001)
    market_ret   = np.full(n,  0.0003)

    cum_strategy = np.cumprod(1 + strategy_ret)
    cum_market   = np.cumprod(1 + market_ret)

    return pd.DataFrame({
        "Timestamp":         pd.date_range("2026-01-01", periods=n, freq="5min"),
        "Price":             np.linspace(75, 80, n),
        "Prediction":        np.full(n, 0.35),
        "Signal":            np.full(n, -1),
        "Next_Ret":          market_ret,
        "Strategy_Ret":      strategy_ret,
        "Cum_Market_Ret":    cum_market,
        "Cum_Strategy_Ret":  cum_strategy,
        "Equity":            10_000 * cum_strategy,
    })


# ---------------------------------------------------------------------------
# compute_metrics — output structure
# ---------------------------------------------------------------------------

def test_metrics_returns_dict(profitable_results):
    """compute_metrics must return a dictionary."""
    m = LiveBacktester.compute_metrics(profitable_results)
    assert isinstance(m, dict)


def test_metrics_has_all_required_keys(profitable_results):
    """All 7 metric keys must be present in the output."""
    required = {
        "total_return", "market_return", "win_rate",
        "sharpe_ratio", "max_drawdown", "profit_factor", "calmar_ratio",
    }
    m = LiveBacktester.compute_metrics(profitable_results)
    assert required.issubset(m.keys())


def test_metrics_all_values_are_floats(profitable_results):
    """Every metric value must be a float (or int, which is a subclass)."""
    m = LiveBacktester.compute_metrics(profitable_results)
    for key, val in m.items():
        assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"


# ---------------------------------------------------------------------------
# compute_metrics — profitable strategy
# ---------------------------------------------------------------------------

def test_profitable_total_return_positive(profitable_results):
    m = LiveBacktester.compute_metrics(profitable_results)
    assert m["total_return"] > 0, "Expected positive total return for profitable strategy"


def test_profitable_win_rate_high(profitable_results):
    """All strategy returns are +0.1%, so win rate must be 100%."""
    m = LiveBacktester.compute_metrics(profitable_results)
    assert m["win_rate"] == pytest.approx(100.0, abs=0.1)


def test_profitable_sharpe_positive(profitable_results):
    """Consistently positive returns should give a positive Sharpe ratio."""
    m = LiveBacktester.compute_metrics(profitable_results)
    assert m["sharpe_ratio"] > 0


def test_profitable_outperforms_market(profitable_results):
    m = LiveBacktester.compute_metrics(profitable_results)
    assert m["total_return"] > m["market_return"]


# ---------------------------------------------------------------------------
# compute_metrics — losing strategy
# ---------------------------------------------------------------------------

def test_losing_total_return_negative(losing_results):
    m = LiveBacktester.compute_metrics(losing_results)
    assert m["total_return"] < 0


def test_losing_win_rate_zero(losing_results):
    """All strategy returns are -0.1%, so win rate must be 0%."""
    m = LiveBacktester.compute_metrics(losing_results)
    assert m["win_rate"] == pytest.approx(0.0, abs=0.1)


def test_losing_max_drawdown_negative(losing_results):
    """Max drawdown must be negative (it measures decline)."""
    m = LiveBacktester.compute_metrics(losing_results)
    assert m["max_drawdown"] < 0


# ---------------------------------------------------------------------------
# compute_metrics — boundary / edge cases
# ---------------------------------------------------------------------------

def test_metrics_win_rate_between_0_and_100(profitable_results):
    m = LiveBacktester.compute_metrics(profitable_results)
    assert 0 <= m["win_rate"] <= 100


def test_metrics_max_drawdown_at_most_zero(profitable_results):
    """
    Max drawdown is always <= 0.
    For a strictly upward equity curve it should be exactly 0.
    """
    m = LiveBacktester.compute_metrics(profitable_results)
    assert m["max_drawdown"] <= 0
