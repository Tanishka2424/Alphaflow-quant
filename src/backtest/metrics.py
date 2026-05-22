"""
src/backtest/metrics.py
-----------------------
All trading performance metrics in one place.
These are computed from a backtest results DataFrame.

Separated from engine.py so they can be tested independently
and reused by the dashboard without importing the full engine.

Location: F:/commodity_trading_project/src/backtest/metrics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ANNUALISATION_FACTOR


def total_return(cum_strategy_ret: pd.Series) -> float:
    """
    Total percentage return of the strategy.

    Args:
        cum_strategy_ret: Cumulative product of (1 + strategy_ret).

    Returns:
        Total return as a percentage (e.g. 12.4 means +12.4%).
    """
    return round((cum_strategy_ret.iloc[-1] - 1) * 100, 2)


def market_return(cum_market_ret: pd.Series) -> float:
    """Buy-and-hold market return as a percentage."""
    return round((cum_market_ret.iloc[-1] - 1) * 100, 2)


def win_rate(strategy_ret: pd.Series) -> float:
    """
    Percentage of trades that were profitable.

    Args:
        strategy_ret: Per-candle strategy returns series.

    Returns:
        Win rate as a percentage (e.g. 54.2 means 54.2% of trades won).
    """
    return round(float((strategy_ret > 0).mean() * 100), 2)


def sharpe_ratio(strategy_ret: pd.Series) -> float:
    """
    Annualised Sharpe ratio.

    Measures risk-adjusted return. A Sharpe > 2.0 on 5-min data
    is considered strong. Computed using 5-min annualisation factor.

    Args:
        strategy_ret: Per-candle strategy returns series.

    Returns:
        Sharpe ratio (higher is better). Returns 0.0 if std is zero.
    """
    std = strategy_ret.std()
    if std == 0:
        return 0.0
    return round(
        float(np.sqrt(ANNUALISATION_FACTOR) * strategy_ret.mean() / std),
        2,
    )


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough decline of the equity curve.

    Answers: "What is the worst losing streak this strategy had?"
    A drawdown of -8.3 means the equity fell 8.3% from its peak
    before recovering.

    Args:
        equity: Equity curve series (dollar values).

    Returns:
        Max drawdown as a negative percentage (e.g. -8.3).
    """
    peak     = equity.cummax()
    drawdown = (equity / peak) - 1
    return round(float(drawdown.min() * 100), 2)


def profit_factor(strategy_ret: pd.Series) -> float:
    """
    Ratio of gross profit to gross loss.

    A profit factor > 1.0 means the strategy makes more than it loses.
    > 1.5 is considered good. > 2.0 is excellent.

    Args:
        strategy_ret: Per-candle strategy returns series.

    Returns:
        Profit factor. Returns inf if there are no losing trades.
    """
    gross_profit = strategy_ret[strategy_ret > 0].sum()
    gross_loss   = strategy_ret[strategy_ret < 0].sum()
    if gross_loss == 0:
        return float("inf")
    return round(float(gross_profit / abs(gross_loss)), 2)


def calmar_ratio(strategy_ret: pd.Series, equity: pd.Series) -> float:
    """
    Annual return divided by maximum drawdown.

    Measures how much return the strategy generates per unit of
    drawdown risk. Higher is better.

    Args:
        strategy_ret: Per-candle returns series.
        equity:       Equity curve series.

    Returns:
        Calmar ratio. Returns inf if max drawdown is zero.
    """
    ann_ret = total_return(
        (1 + strategy_ret).cumprod()
    ) * (ANNUALISATION_FACTOR / max(len(strategy_ret), 1))
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float("inf")
    return round(float(ann_ret / abs(mdd)), 2)


def compute_all(results: pd.DataFrame) -> dict:
    """
    Compute all metrics from a backtest results DataFrame in one call.

    Args:
        results: DataFrame with columns:
                 Strategy_Ret, Next_Ret, Cum_Strategy_Ret,
                 Cum_Market_Ret, Equity.

    Returns:
        Dictionary with all 7 metrics.
    """
    sr = results["Strategy_Ret"]
    eq = results["Equity"]

    return {
        "total_return":  total_return(results["Cum_Strategy_Ret"]),
        "market_return": market_return(results["Cum_Market_Ret"]),
        "win_rate":      win_rate(sr),
        "sharpe_ratio":  sharpe_ratio(sr),
        "max_drawdown":  max_drawdown(eq),
        "profit_factor": profit_factor(sr),
        "calmar_ratio":  calmar_ratio(sr, eq),
    }
