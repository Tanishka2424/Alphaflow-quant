"""
tests/test_metrics.py
---------------------
Unit tests for src/backtest/metrics.py.
Tests use synthetic data — no internet, no model needed.

Run with:
    pytest tests/test_metrics.py -v

Location: F:/commodity_trading_project/tests/test_metrics.py
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest.metrics import (
    total_return,
    market_return,
    win_rate,
    sharpe_ratio,
    max_drawdown,
    profit_factor,
    calmar_ratio,
    compute_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def all_winning() -> pd.Series:
    """Strategy that wins every single trade."""
    return pd.Series(np.full(200, 0.002))   # +0.2% per candle


@pytest.fixture
def all_losing() -> pd.Series:
    """Strategy that loses every single trade."""
    return pd.Series(np.full(200, -0.002))


@pytest.fixture
def mixed_returns() -> pd.Series:
    """Realistic mix of wins and losses — 55% win rate."""
    np.random.seed(0)
    rets = np.where(
        np.random.rand(500) > 0.45,
        np.random.uniform(0.001, 0.003, 500),
        np.random.uniform(-0.002, -0.001, 500),
    )
    return pd.Series(rets)


@pytest.fixture
def equity_curve(mixed_returns) -> pd.Series:
    return pd.Series(10_000 * (1 + mixed_returns).cumprod().values)


@pytest.fixture
def full_results(mixed_returns) -> pd.DataFrame:
    """Full results DataFrame as returned by backtest engine."""
    n  = len(mixed_returns)
    mr = mixed_returns * 0.6   # market returns (weaker than strategy)
    return pd.DataFrame({
        "Strategy_Ret":     mixed_returns,
        "Next_Ret":         mr,
        "Cum_Strategy_Ret": (1 + mixed_returns).cumprod(),
        "Cum_Market_Ret":   (1 + mr).cumprod(),
        "Equity":           10_000 * (1 + mixed_returns).cumprod(),
    })


# ---------------------------------------------------------------------------
# total_return
# ---------------------------------------------------------------------------

def test_total_return_positive_for_winning(all_winning):
    cum = (1 + all_winning).cumprod()
    assert total_return(cum) > 0


def test_total_return_negative_for_losing(all_losing):
    cum = (1 + all_losing).cumprod()
    assert total_return(cum) < 0


def test_total_return_is_float(all_winning):
    cum = (1 + all_winning).cumprod()
    assert isinstance(total_return(cum), float)


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

def test_win_rate_100_for_all_winning(all_winning):
    assert win_rate(all_winning) == pytest.approx(100.0, abs=0.01)


def test_win_rate_0_for_all_losing(all_losing):
    assert win_rate(all_losing) == pytest.approx(0.0, abs=0.01)


def test_win_rate_between_0_and_100(mixed_returns):
    assert 0 <= win_rate(mixed_returns) <= 100


def test_win_rate_mixed_near_55(mixed_returns):
    """Mixed fixture has ~55% win rate by construction."""
    assert 50 <= win_rate(mixed_returns) <= 65


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

def test_sharpe_positive_for_winning(all_winning):
    assert sharpe_ratio(all_winning) > 0


def test_sharpe_negative_for_losing(all_losing):
    assert sharpe_ratio(all_losing) < 0


def test_sharpe_zero_std_returns_zero():
    """Constant returns have zero std — should return 0 not crash."""
    flat = pd.Series(np.zeros(100))
    assert sharpe_ratio(flat) == 0.0


def test_sharpe_is_float(mixed_returns):
    assert isinstance(sharpe_ratio(mixed_returns), float)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_negative_for_losses(all_losing):
    equity = pd.Series(10_000 * (1 + all_losing).cumprod().values)
    assert max_drawdown(equity) < 0


def test_max_drawdown_zero_for_monotone_growth(all_winning):
    """Equity that only goes up has zero drawdown."""
    equity = pd.Series(10_000 * (1 + all_winning).cumprod().values)
    assert max_drawdown(equity) == pytest.approx(0.0, abs=0.01)


def test_max_drawdown_at_most_zero(equity_curve):
    assert max_drawdown(equity_curve) <= 0


def test_max_drawdown_at_least_minus_100(equity_curve):
    """Drawdown can never be worse than -100%."""
    assert max_drawdown(equity_curve) >= -100


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------

def test_profit_factor_above_1_for_winning(mixed_returns):
    assert profit_factor(mixed_returns) > 1.0


def test_profit_factor_below_1_for_losing(all_losing):
    assert profit_factor(all_losing) == pytest.approx(0.0, abs=0.01)


def test_profit_factor_inf_when_no_losses(all_winning):
    assert profit_factor(all_winning) == float("inf")


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all_returns_dict(full_results):
    m = compute_all(full_results)
    assert isinstance(m, dict)


def test_compute_all_has_all_keys(full_results):
    required = {
        "total_return", "market_return", "win_rate",
        "sharpe_ratio", "max_drawdown", "profit_factor", "calmar_ratio",
    }
    assert required.issubset(compute_all(full_results).keys())


def test_compute_all_values_are_numeric(full_results):
    for k, v in compute_all(full_results).items():
        assert isinstance(v, (int, float)), f"{k} is not numeric"


def test_compute_all_strategy_beats_market(full_results):
    """Mixed fixture strategy return is stronger than market by construction."""
    m = compute_all(full_results)
    assert m["total_return"] > m["market_return"]
