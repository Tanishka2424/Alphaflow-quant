# Changelog

All notable changes to AlphaFlow Quant are documented here.

---

## [1.0.0] — May 2026

### Added
- Multi-page Streamlit dashboard (Live Signal, Backtest, Performance, Model Explainer)
- Asset selector: Crude Oil, Gold, Bitcoin
- Interactive threshold slider (0.50–0.75) with live signal update
- Confidence gauge meter with Plotly
- Candlestick chart with Bollinger Bands and RSI subplot
- Drawdown chart on Backtest page
- Rolling Sharpe Ratio chart on Performance page
- Model comparison table (CNN-LSTM vs XGBoost vs SMA)
- SHAP feature importance bar chart with feature descriptions
- Live feature values table showing current indicator values

### Changed
- Converted single-file app to modular multi-page architecture
- Extracted feature engineering into `src/features/engineering.py`
- Extracted backtest engine into `src/backtest/engine.py`
- Extracted metrics into `src/backtest/metrics.py`
- All constants moved to `config.py`

---

## [0.3.0] — April 2026

### Added
- XGBoost baseline model for comparison (`src/model/baseline.py`)
- Walk-forward validation with `TimeSeriesSplit` (`src/model/validator.py`)
- SHAP explainability wrapper (`src/model/predictor.py`)
- 75 pytest tests across 5 test files
- GitHub Actions CI pipeline (runs tests on every push)
- Bollinger Bands and ATR added to feature engineering
- Multi-asset data downloader (Crude Oil, Gold, Bitcoin)
- Prediction logging to CSV

### Changed
- Replaced random train/test split with TimeSeriesSplit
- Pinned all package versions in requirements.txt

---

## [0.2.0] — March 2026

### Added
- Historical backtest with equity curve
- Strategy vs buy-and-hold comparison
- Sharpe Ratio, Win Rate metrics
- Plotly interactive charts replacing matplotlib
- IST timezone handling
- Market open/closed status detection

### Changed
- Moved from script-based to class-based architecture (LiveBacktester)

---

## [0.1.0] — February 2026

### Added
- Initial CNN-LSTM model training on Crude Oil 5-minute data
- Basic live BUY/SELL signal prediction
- RSI, MACD, Volatility, Log Returns feature engineering
- RobustScaler for feature normalisation
- Basic Streamlit dashboard
- yfinance data integration
