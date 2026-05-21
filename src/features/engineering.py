"""
src/features/engineering.py
---------------------------
All technical indicator calculations live here.
Extracted from LiveBacktester so they can be tested independently
and reused by any model or backtest engine.

Location: F:/commodity_trading_project/src/features/engineering.py
"""

import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log return between consecutive close prices.

    Log returns are preferred over simple returns in ML models because
    they are additive over time and more normally distributed.

    Args:
        df: DataFrame with a 'Close' column.

    Returns:
        DataFrame with added 'log_ret' column.
    """
    df = df.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_price_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute high-low range and close-open range per candle.

    These capture intrabar volatility and directional pressure,
    which are strong short-term momentum signals.

    Args:
        df: DataFrame with 'High', 'Low', 'Open', 'Close' columns.

    Returns:
        DataFrame with added 'hl_range' and 'co_range' columns.
    """
    df = df.copy()
    df["hl_range"] = df["High"] - df["Low"]
    df["co_range"] = df["Close"] - df["Open"]
    return df


def add_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute rolling standard deviation of log returns as a volatility proxy.

    A 14-period rolling std on 5-minute data captures ~70 minutes of
    recent volatility — useful for filtering high-noise periods.

    Args:
        df:     DataFrame with a 'log_ret' column (run add_log_returns first).
        window: Rolling window size. Default 14.

    Returns:
        DataFrame with added 'volatility' column.
    """
    df = df.copy()
    df["volatility"] = df["log_ret"].rolling(window=window).std()
    return df


def add_volume_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage change in volume between consecutive candles.

    Volume spikes often precede directional moves, making this a
    useful leading indicator for the CNN-LSTM model.

    Args:
        df: DataFrame with a 'Volume' column.

    Returns:
        DataFrame with added 'volume_pct' column.
    """
    df = df.copy()
    df["volume_pct"] = df["Volume"].pct_change()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes.
    Values above 70 indicate overbought; below 30 indicate oversold.
    Uses Wilder's smoothing (rolling mean) for consistency.

    Args:
        df:     DataFrame with a 'Close' column.
        window: Lookback period. Default 14 (standard).

    Returns:
        DataFrame with added 'rsi' column.
    """
    df = df.copy()
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs    = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute MACD line and signal line.

    MACD captures trend direction and momentum through the difference
    between two exponential moving averages. Standard parameters are
    (12, 26, 9) — widely used in commodity and crypto markets.

    Args:
        df:     DataFrame with a 'Close' column.
        fast:   Fast EMA period. Default 12.
        slow:   Slow EMA period. Default 26.
        signal: Signal line EMA period. Default 9.

    Returns:
        DataFrame with added 'macd' and 'macd_signal' columns.
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Compute Bollinger Bands (upper and lower).

    Bollinger Bands adapt to volatility — they widen in volatile markets
    and narrow in calm ones. Useful as additional context for the model.

    Args:
        df:      DataFrame with a 'Close' column.
        window:  Rolling window. Default 20.
        num_std: Number of standard deviations. Default 2.0.

    Returns:
        DataFrame with added 'bb_upper' and 'bb_lower' columns.
    """
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["bb_upper"] = sma + num_std * std
    df["bb_lower"] = sma - num_std * std
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of
    an asset for that period. Widely used for position sizing in HFT.

    Args:
        df:     DataFrame with 'High', 'Low', 'Close' columns.
        window: Lookback period. Default 14.

    Returns:
        DataFrame with added 'atr' column.
    """
    df   = df.copy()
    prev = df["Close"].shift(1)
    tr   = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"]  - prev).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=window).mean()
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full feature engineering pipeline on raw OHLCV data.

    Applies all indicator functions in the correct dependency order,
    cleans infinities and NaNs, then returns both the full DataFrame
    (with Close prices for backtest P&L) and the feature-only DataFrame
    (for model input).

    Args:
        df: Raw OHLCV DataFrame from yfinance. Must have columns:
            Open, High, Low, Close, Volume.

    Returns:
        full_df:     Full DataFrame aligned to valid feature rows.
        features_df: Feature-only DataFrame (FEATURE_COLS columns).

    Example:
        >>> full_df, features_df = build_features(raw_data)
        >>> scaled = scaler.transform(features_df)
    """
    from config import FEATURE_COLS

    df = df.copy()

    # Flatten MultiIndex columns (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    df = add_log_returns(df)
    df = add_price_ranges(df)
    df = add_volatility(df)
    df = add_volume_pct_change(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)

    # Replace inf values that can appear in pct_change or division
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    features_df = df[FEATURE_COLS].dropna()
    full_df     = df.loc[features_df.index].copy()

    return full_df, features_df
