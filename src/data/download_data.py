"""
src/data/download_data.py
-------------------------
Downloads OHLCV data for all supported assets and saves to data/.

Location: F:/commodity_trading_project/src/data/download_data.py

Usage:
    python src/data/download_data.py
"""

import os
import sys
import yfinance as yf

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import SUPPORTED_ASSETS, FETCH_INTERVAL, BACKTEST_PERIOD

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_asset(symbol: str, name: str) -> None:
    """
    Download OHLCV data for one asset and save as CSV.

    Args:
        symbol: Yahoo Finance ticker (e.g. 'CL=F').
        name:   Human-readable name used in the output filename.
    """
    print(f"Downloading {name} ({symbol})...")
    df = yf.download(
        tickers=symbol,
        interval=FETCH_INTERVAL,
        period=BACKTEST_PERIOD,
        progress=False,
    )

    if df.empty:
        print(f"  WARNING: No data returned for {symbol}. Skipping.")
        return

    # Flatten MultiIndex columns (recent yfinance versions)
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)

    filename = os.path.join(DATA_DIR, f"{name.lower().replace(' ', '_')}_5m.csv")
    df.to_csv(filename)
    print(f"  Saved {len(df)} rows → {filename}")
    print(f"  Columns: {list(df.columns)}\n")


if __name__ == "__main__":
    for sym, meta in SUPPORTED_ASSETS.items():
        download_asset(sym, meta["name"])
    print("All downloads complete.")
