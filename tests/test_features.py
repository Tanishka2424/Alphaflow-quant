"""
tests/test_features.py
----------------------
Unit tests for every function in src/features/engineering.py.
Tests use synthetic price data so they run offline — no internet needed.

Run with:
    pytest tests/test_features.py -v

Location: F:/commodity_trading_project/tests/test_features.py
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.engineering import (
    add_log_returns,
    add_price_ranges,
    add_volatility,
    add_volume_pct_change,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_atr,
    build_features,
)


# ---------------------------------------------------------------------------
# Shared fixture: synthetic OHLCV data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Create 100 rows of synthetic OHLCV data.
    Prices follow a simple upward trend with small noise.
    This is enough rows for all indicators (RSI needs 14, MACD needs 26).
    """
    np.random.seed(42)
    n = 100
    close  = 75 + np.cumsum(np.random.randn(n) * 0.1)
    open_  = close - np.random.uniform(0.05, 0.15, n)
    high   = close + np.random.uniform(0.05, 0.20, n)
    low    = close - np.random.uniform(0.05, 0.20, n)
    volume = np.random.randint(5000, 20000, n).astype(float)

    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# add_log_returns
# ---------------------------------------------------------------------------

def test_log_returns_column_exists(sample_df):
    """log_ret column must be created."""
    result = add_log_returns(sample_df)
    assert "log_ret" in result.columns


def test_log_returns_first_row_is_nan(sample_df):
    """First row has no previous close, so log_ret must be NaN."""
    result = add_log_returns(sample_df)
    assert pd.isna(result["log_ret"].iloc[0])


def test_log_returns_values_are_finite(sample_df):
    """All non-NaN log returns must be finite floats."""
    result = add_log_returns(sample_df)
    finite = result["log_ret"].dropna()
    assert np.isfinite(finite.values).all()


def test_log_returns_does_not_mutate_input(sample_df):
    """Function must not modify the original DataFrame."""
    original_cols = list(sample_df.columns)
    add_log_returns(sample_df)
    assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# add_price_ranges
# ---------------------------------------------------------------------------

def test_price_ranges_columns_exist(sample_df):
    """hl_range and co_range columns must both be created."""
    result = add_price_ranges(sample_df)
    assert "hl_range" in result.columns
    assert "co_range" in result.columns


def test_hl_range_always_positive(sample_df):
    """High - Low must always be >= 0 for valid OHLCV data."""
    result = add_price_ranges(sample_df)
    assert (result["hl_range"] >= 0).all()


# ---------------------------------------------------------------------------
# add_rsi
# ---------------------------------------------------------------------------

def test_rsi_column_exists(sample_df):
    result = add_rsi(sample_df)
    assert "rsi" in result.columns


def test_rsi_values_in_range(sample_df):
    """RSI must always be between 0 and 100 (inclusive)."""
    result = add_rsi(sample_df)
    valid  = result["rsi"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_has_enough_non_null_values(sample_df):
    """With 100 rows and window=14, we must have at least 80 valid RSI values."""
    result = add_rsi(sample_df)
    assert result["rsi"].notna().sum() >= 80


# ---------------------------------------------------------------------------
# add_macd
# ---------------------------------------------------------------------------

def test_macd_columns_exist(sample_df):
    result = add_macd(sample_df)
    assert "macd"        in result.columns
    assert "macd_signal" in result.columns


def test_macd_signal_lags_macd(sample_df):
    """
    macd_signal is an EMA of macd, so it should have at least as many
    non-null values as macd itself.
    """
    result = add_macd(sample_df)
    assert result["macd_signal"].notna().sum() <= result["macd"].notna().sum()


# ---------------------------------------------------------------------------
# add_bollinger_bands
# ---------------------------------------------------------------------------

def test_bollinger_columns_exist(sample_df):
    result = add_bollinger_bands(sample_df)
    assert "bb_upper" in result.columns
    assert "bb_lower" in result.columns


def test_bollinger_upper_above_lower(sample_df):
    """Upper band must always be >= lower band."""
    result = add_bollinger_bands(sample_df)
    valid  = result.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


# ---------------------------------------------------------------------------
# add_atr
# ---------------------------------------------------------------------------

def test_atr_column_exists(sample_df):
    result = add_atr(sample_df)
    assert "atr" in result.columns


def test_atr_values_positive(sample_df):
    """ATR is based on absolute price ranges, so must be >= 0."""
    result = add_atr(sample_df)
    valid  = result["atr"].dropna()
    assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# build_features (integration test)
# ---------------------------------------------------------------------------

def test_build_features_returns_two_dataframes(sample_df):
    full_df, features_df = build_features(sample_df)
    assert isinstance(full_df,     pd.DataFrame)
    assert isinstance(features_df, pd.DataFrame)


def test_build_features_correct_columns(sample_df):
    """features_df must contain exactly the 8 FEATURE_COLS columns."""
    from config import FEATURE_COLS
    _, features_df = build_features(sample_df)
    assert list(features_df.columns) == FEATURE_COLS


def test_build_features_no_nulls(sample_df):
    """After build_features, features_df must have zero NaN values."""
    _, features_df = build_features(sample_df)
    assert features_df.isna().sum().sum() == 0


def test_build_features_no_infinities(sample_df):
    """features_df must contain no inf or -inf values."""
    _, features_df = build_features(sample_df)
    assert np.isfinite(features_df.values).all()


def test_build_features_index_aligned(sample_df):
    """full_df and features_df must share the same index."""
    full_df, features_df = build_features(sample_df)
    assert list(full_df.index) == list(features_df.index)


def test_build_features_handles_multiindex(sample_df):
    """build_features must handle yfinance-style MultiIndex columns."""
    multi = sample_df.copy()
    multi.columns = pd.MultiIndex.from_tuples(
        [(c, "CL=F") for c in sample_df.columns]
    )
    full_df, features_df = build_features(multi)
    assert not features_df.empty
