"""
src/backtest/engine.py
----------------------
Clean, type-hinted version of LiveBacktester.
Feature engineering is now delegated to src/features/engineering.py.
All constants come from config.py.

Location: F:/commodity_trading_project/src/backtest/engine.py
"""

from __future__ import annotations

import os



import numpy as np
import pandas as pd
import yfinance as yf

import tensorflow as tf

from joblib import load

from config import (
    WINDOW_SIZE,
    FEATURE_COLS,
    MODEL_PATH,
    SCALER_PATH,
    LOG_PATH,
    DEFAULT_CAPITAL,
    DEFAULT_THRESHOLD,
    FETCH_INTERVAL,
    LIVE_FETCH_PERIOD,
    BACKTEST_PERIOD,
    ANNUALISATION_FACTOR,
)
from src.features.engineering import build_features


class LiveBacktester:
    """
    End-to-end prediction and backtesting engine for a single asset.

    Loads a trained CNN-LSTM model and RobustScaler, fetches live OHLCV
    data from yfinance, runs feature engineering, generates BUY/SELL
    signals, and simulates a long/short backtest.

    Attributes:
        symbol:      Yahoo Finance ticker symbol (e.g. 'CL=F').
        model:       Loaded Keras CNN-LSTM model.
        scaler:      Loaded RobustScaler fitted during training.
        window_size: Number of time steps per input sequence.
    """

    def __init__(
        self,
        symbol: str = "CL=F",
        model_path: str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
    ) -> None:
        self.symbol      = symbol
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.window_size = WINDOW_SIZE
        self.model       = None
        self.scaler      = None

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------

    def load_resources(self) -> None:
        """Load the Keras model and scaler from disk."""
        print(f"Loading model  → {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Loading scaler → {self.scaler_path}")
        self.scaler = load(self.scaler_path)
        print("Model and scaler loaded successfully.")
     

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        period: str   = BACKTEST_PERIOD,
        interval: str = FETCH_INTERVAL,
    ) -> pd.DataFrame:
        """
        Download OHLCV data from yfinance for the configured symbol.

        Args:
            period:   How far back to fetch (e.g. '1mo', '5d').
            interval: Candle interval (e.g. '5m', '1h').

        Returns:
            Cleaned DataFrame with Open, High, Low, Close, Volume columns.

        Raises:
            ValueError: If no data is returned for the symbol/period.
        """
        print(f"Fetching {self.symbol} | period={period} | interval={interval}")
        df = yf.download(
            self.symbol,
            period=period,
            interval=interval,
            progress=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        if df.empty:
            raise ValueError(
                f"No data returned for symbol '{self.symbol}' "
                f"with period='{period}'. Check the symbol or try a different period."
            )

        print(f"Fetched {len(df)} rows.")
        return df

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def get_latest_signal(
        self,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[dict | None, str | None]:
        """
        Predict the direction of the next 5-minute candle.

        Fetches the most recent data, extracts the last WINDOW_SIZE rows,
        scales them, and passes them through the CNN-LSTM model.

        Args:
            threshold: Probability cutoff. Predictions above this are BUY.

        Returns:
            A tuple of (signal_dict, error_message).
            On success: (signal_dict, None).
            On failure: (None, error_message string).

        Example return value:
            {
                "timestamp":   Timestamp("2026-05-20 14:30:00"),
                "price":       78.34,
                "probability": 0.724,
                "signal":      "BUY",
                "confidence":  "72.40%",
            }
        """
        try:
            raw   = self.fetch_data(period=LIVE_FETCH_PERIOD)
            full_df, features_df = build_features(raw)

            last_window = features_df.tail(self.window_size)
            if len(last_window) < self.window_size:
                return None, (
                    f"Insufficient data: got {len(last_window)} rows, "
                    f"need {self.window_size}. Try a longer fetch period."
                )

            scaled = self.scaler.transform(last_window)
            x_live = np.array([scaled])          # shape: (1, 60, 8)
            prob   = float(self.model.predict(x_live, verbose=0)[0][0])

            signal     = "BUY"  if prob > threshold else "SELL"
            confidence = prob   if prob > threshold else (1 - prob)

            result = {
                "timestamp":   features_df.index[-1],
                "price":       float(full_df["Close"].iloc[-1]),
                "probability": round(prob, 4),
                "signal":      signal,
                "confidence":  f"{confidence * 100:.2f}%",
            }

            self._log_prediction(result)
            return result, None

        except Exception as exc:
            return None, str(exc)

    # ------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        initial_capital: float = DEFAULT_CAPITAL,
        threshold: float       = DEFAULT_THRESHOLD,
        period: str            = BACKTEST_PERIOD,
    ) -> pd.DataFrame:
        """
        Simulate a long/short trading strategy over historical data.

        For every completed WINDOW_SIZE window, the model predicts
        the next candle's direction. A BUY signal goes long; a SELL
        signal goes short. Transaction costs are not modelled (educational).

        Args:
            initial_capital: Starting portfolio value in USD.
            threshold:       Probability cutoff for signal direction.
            period:          History to fetch (e.g. '1mo', '3mo').

        Returns:
            DataFrame with columns:
                Timestamp, Price, Next_Price, Prediction, Signal,
                Next_Ret, Strategy_Ret, Cum_Market_Ret,
                Cum_Strategy_Ret, Equity.
        """
        raw_data = self.fetch_data(period=period)
        full_df, features_df = build_features(raw_data)

        scaled = self.scaler.transform(features_df)

        windows:    list[np.ndarray] = []
        prices:     list[float]      = []
        timestamps: list             = []

        for i in range(len(scaled) - self.window_size):
            windows.append(scaled[i : i + self.window_size])
            prices.append(float(full_df["Close"].iloc[i + self.window_size - 1]))
            timestamps.append(full_df.index[i + self.window_size - 1])

        X = np.array(windows)
        print(f"Running model on {len(X)} windows...")
        predictions = self.model.predict(X, verbose=0).flatten()

        results = pd.DataFrame({
            "Timestamp":  timestamps,
            "Price":      prices,
            "Prediction": predictions,
        })

        # Next-candle prices for computing realised returns
        next_prices = full_df["Close"].iloc[self.window_size:].values
        n           = min(len(results), len(next_prices))
        results     = results.iloc[:n].copy()
        results["Next_Price"] = next_prices[:n]
        results["Next_Ret"]   = (results["Next_Price"] - results["Price"]) / results["Price"]

        results["Signal"]       = np.where(results["Prediction"] > threshold, 1, -1)
        results["Strategy_Ret"] = results["Signal"] * results["Next_Ret"]

        results["Cum_Market_Ret"]   = (1 + results["Next_Ret"]).cumprod()
        results["Cum_Strategy_Ret"] = (1 + results["Strategy_Ret"]).cumprod()
        results["Equity"]           = initial_capital * results["Cum_Strategy_Ret"]

        self._print_summary(results, initial_capital)
        return results

    # ------------------------------------------------------------------
    # Metrics (static helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(results: pd.DataFrame) -> dict:
        """
        Compute all performance metrics from a backtest results DataFrame.

        Args:
            results: DataFrame returned by run_backtest().

        Returns:
            Dictionary with keys:
                total_return, market_return, win_rate, sharpe_ratio,
                max_drawdown, profit_factor, calmar_ratio.
        """
        strategy_rets = results["Strategy_Ret"]
        equity        = results["Equity"]

        total_return  = (results["Cum_Strategy_Ret"].iloc[-1] - 1) * 100
        market_return = (results["Cum_Market_Ret"].iloc[-1]   - 1) * 100
        win_rate      = float((strategy_rets > 0).mean() * 100)

        sharpe = (
            float(np.sqrt(ANNUALISATION_FACTOR)
            * strategy_rets.mean()
            / strategy_rets.std())
            if strategy_rets.std() != 0 else 0.0
        )

        # Drawdown: peak-to-trough decline of equity curve
        peak         = equity.cummax()
        drawdown     = (equity / peak) - 1
        max_drawdown = float(drawdown.min() * 100)

        # Profit factor: gross profit / gross loss
        gross_profit = strategy_rets[strategy_rets > 0].sum()
        gross_loss   = strategy_rets[strategy_rets < 0].sum()
        profit_factor = (
            float(gross_profit / abs(gross_loss))
            if gross_loss != 0 else float("inf")
        )

        # Calmar ratio: annual return / max drawdown
        annual_return = total_return * (ANNUALISATION_FACTOR / max(len(results), 1))
        calmar_ratio  = (
            float(annual_return / abs(max_drawdown))
            if max_drawdown != 0 else float("inf")
        )

        return {
            "total_return":   round(total_return,  2),
            "market_return":  round(market_return, 2),
            "win_rate":       round(win_rate,       2),
            "sharpe_ratio":   round(sharpe,         2),
            "max_drawdown":   round(max_drawdown,   2),
            "profit_factor":  round(profit_factor,  2),
            "calmar_ratio":   round(calmar_ratio,   2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_prediction(self, signal: dict) -> None:
        """Append a prediction record to the CSV log for post-hoc analysis."""
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        row = pd.DataFrame([signal])
        row.to_csv(
            LOG_PATH,
            mode="a",
            header=not os.path.exists(LOG_PATH),
            index=False,
        )

    def _print_summary(self, results: pd.DataFrame, initial_capital: float) -> None:
        """Print a formatted backtest summary to stdout."""
        m = self.compute_metrics(results)
        print("\n" + "=" * 35)
        print("       BACKTEST RESULTS")
        print("=" * 35)
        print(f"Symbol:          {self.symbol}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Equity:    ${results['Equity'].iloc[-1]:,.2f}")
        print(f"Total Return:    {m['total_return']}%")
        print(f"Market Return:   {m['market_return']}%")
        print(f"Win Rate:        {m['win_rate']}%")
        print(f"Sharpe Ratio:    {m['sharpe_ratio']}")
        print(f"Max Drawdown:    {m['max_drawdown']}%")
        print(f"Profit Factor:   {m['profit_factor']}")
        print(f"Calmar Ratio:    {m['calmar_ratio']}")
        print("=" * 35)
