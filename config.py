"""
config.py
---------
Central configuration for the AI Trading Platform.
All constants live here. Change a value here and it updates everywhere.

Location: F:/commodity_trading_project/config.py  (project root)
"""

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
SUPPORTED_ASSETS: dict[str, dict] = {
    "CL=F":    {"name": "Crude Oil", "type": "commodity", "interval": "5m"},
    "GC=F":    {"name": "Gold",      "type": "commodity", "interval": "5m"},
    "BTC-USD": {"name": "Bitcoin",   "type": "crypto",    "interval": "5m"},
}
DEFAULT_SYMBOL: str = "CL=F"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
WINDOW_SIZE: int   = 60     # Lookback window — 60 x 5 min = 5 hours of history
N_FEATURES:  int   = 8      # Number of input features fed into the model
DEFAULT_THRESHOLD: float = 0.55  # Probability cutoff for BUY vs SELL signal

# Feature column names — order must match what the scaler was trained on
FEATURE_COLS: list[str] = [
    "log_ret",
    "hl_range",
    "co_range",
    "volatility",
    "volume_pct",
    "rsi",
    "macd",
    "macd_signal",
]

# ---------------------------------------------------------------------------
# File paths  (relative to project root — works on any machine)
# ---------------------------------------------------------------------------
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH:  str = os.path.join(_ROOT, "model", "crude_cnn_lstm.keras")
SCALER_PATH: str = os.path.join(_ROOT, "model", "scaler.joblib")
SHAP_PATH:   str = os.path.join(_ROOT, "model", "shap_values.json")
LOG_PATH:    str = os.path.join(_ROOT, "logs",  "predictions.csv")

# ---------------------------------------------------------------------------
# Backtest defaults
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL: int   = 10_000
DEFAULT_PERIOD:  str   = "1mo"

# Annualisation factor for Sharpe Ratio on 5-minute candles
# 252 trading days × 24 hours × 12 candles/hour
ANNUALISATION_FACTOR: float = 252 * 24 * 12

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
FETCH_INTERVAL:       str = "5m"
LIVE_FETCH_PERIOD:    str = "5d"   # Enough rows to fill one 60-step window
BACKTEST_PERIOD:      str = "1mo"
