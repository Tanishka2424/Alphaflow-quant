# 📈 AlphaFlow Quant — AI Trading Intelligence Platform

![CI](https://github.com/Tanishka2424/Alphaflow-quant/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)
![License](https://img.shields.io/badge/License-Educational-green)

A **production-style deep learning platform** for predicting short-term price direction
across commodity and crypto markets using a hybrid CNN-LSTM architecture.
Built as a research-grade internship project with full ML engineering best practices.

---

## 🎯 What This Project Does

AlphaFlow Quant ingests live 5-minute OHLCV data for **Crude Oil, Gold, and Bitcoin**,
engineers 8 technical indicators, feeds them into a trained CNN-LSTM model, and outputs
a **BUY or SELL signal with confidence score** — updated every 5 minutes.

The platform also includes a full backtesting engine, performance analytics,
and a SHAP-powered model explainability page.

---

## 🖥️ Live Dashboard

> **4-page Streamlit dashboard:**
> 🔥 Live Signal · 📈 Backtest · 📊 Performance · 🧠 Model Explainer

---

## 📊 Model Results

Walk-forward validated using `TimeSeriesSplit` — **no lookahead bias.**

| Model | Accuracy | Sharpe Ratio | Win Rate | Validated |
|-------|----------|-------------|----------|-----------|
| **CNN-LSTM (Ours)** | **~53%** | **~2.87** | **~54%** | ✅ Walk-forward |
| XGBoost Baseline | ~51% | ~1.9 | ~52% | ✅ Walk-forward |
| SMA Crossover | ~49% | ~0.8 | ~49% | ✅ Walk-forward |

> In HFT, a 3% edge over random (50%) is considered statistically significant.
> The CNN-LSTM outperforms both baselines on all metrics.

---

## 🏗️ Architecture

```
Live Market Data (yfinance / Binance)
            ↓
   Feature Engineering
   RSI · MACD · Bollinger · ATR
   Volatility · Log Returns
            ↓
   CNN-LSTM Model
   60-candle window → (60, 8) input
   CNN: spatial pattern detection
   LSTM: temporal sequence memory
            ↓
   Signal Engine
   Probability → BUY/SELL + Confidence
            ↓
   Streamlit Dashboard
   Live Signal · Backtest · Performance · SHAP
```

---

## 🧠 Model Details

**Architecture:** Hybrid CNN-LSTM
- **Input:** 60 × 5-minute candles × 8 features = `(60, 8)` tensor
- **CNN Layer:** Detects spatial patterns (breakouts, consolidation, spikes)
- **LSTM Layer:** Captures temporal momentum evolution over 5 hours
- **Output:** Sigmoid probability → BUY if > threshold, SELL otherwise

**Features (8 total):**
| Feature | Description |
|---------|-------------|
| `log_ret` | Logarithmic price return |
| `rsi` | Relative Strength Index (14-period) |
| `macd` | MACD line (12, 26 EMA difference) |
| `macd_signal` | MACD signal line (9-period EMA) |
| `volatility` | Rolling 14-period std of log returns |
| `hl_range` | High-Low intrabar range |
| `co_range` | Close-Open directional pressure |
| `volume_pct` | Volume percentage change |

**Validation:** `TimeSeriesSplit(n_splits=5)` — training always precedes test in time.

---

## 📦 Project Structure

```
alphaflow-quant/
│
├── app/                        # Streamlit multi-page dashboard
│   ├── main.py                 # Landing page + navigation
│   └── pages/
│       ├── 1_Live_Signal.py    # Real-time BUY/SELL signal
│       ├── 2_Backtest.py       # Historical strategy simulation
│       ├── 3_Performance.py    # Sharpe, drawdown, rolling metrics
│       └── 4_Model_Explainer.py# SHAP feature importance
│
├── src/                        # Core ML engine (modular)
│   ├── features/
│   │   └── engineering.py      # RSI, MACD, ATR, Bollinger, etc.
│   ├── backtest/
│   │   ├── engine.py           # LiveBacktester class
│   │   └── metrics.py          # Sharpe, drawdown, profit factor
│   ├── model/
│   │   ├── predictor.py        # CNN-LSTM wrapper + SHAP
│   │   ├── baseline.py         # XGBoost comparison model
│   │   └── validator.py        # Walk-forward validation
│   └── data/
│       └── download_data.py    # Multi-asset data downloader
│
├── tests/                      # 75 pytest tests
│   ├── test_config.py          # Config sanity checks (9 tests)
│   ├── test_features.py        # Feature engineering (21 tests)
│   ├── test_backtest.py        # Backtest engine (12 tests)
│   ├── test_metrics.py         # Performance metrics (19 tests)
│   └── test_baseline.py        # XGBoost baseline (14 tests)
│
├── experiments/
│   └── crude_cnn_lstm.ipynb    # Model training + walk-forward validation
│
├── model/                      # Saved weights (gitignored)
│   ├── crude_cnn_lstm.keras
│   └── scaler.joblib
│
├── config.py                   # All constants in one place
├── conftest.py                 # Pytest path configuration
├── requirements.txt            # Pinned dependencies
├── .github/workflows/test.yml  # CI — runs tests on every push
├── ARCHITECTURE.md             # Design decisions explained
├── MODEL_CARD.md               # Model facts and limitations
└── CHANGELOG.md                # Version history
```

---

## 🚀 Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/Tanishka2424/Alphaflow-quant.git
cd Alphaflow-quant
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run app/main.py
```

### 3. Run tests
```bash
pytest tests/ -v
# Expected: 75 passed
```

### 4. Download fresh data
```bash
python src/data/download_data.py
```

---

## 📈 Assets Supported

| Symbol | Asset | Market | Interval |
|--------|-------|--------|----------|
| `CL=F` | Crude Oil Futures | Commodity | 5-min |
| `GC=F` | Gold Futures | Commodity | 5-min |
| `BTC-USD` | Bitcoin | Crypto | 5-min |

---

## 🔬 Key Engineering Decisions

**Walk-forward validation instead of random split**
Random train/test split on time-series causes lookahead bias — the model trains on
future data. `TimeSeriesSplit` ensures training always uses only past data.

**XGBoost baseline for comparison**
Without a baseline, accuracy numbers are meaningless. XGBoost uses the same features
but no sequence modelling, proving the CNN-LSTM's temporal advantage.

**RobustScaler over StandardScaler**
Financial data has extreme outliers (flash crashes, news spikes).
RobustScaler uses median/IQR instead of mean/std — less sensitive to outliers.

**60-candle lookback window**
12 candles (1h) = too noisy. 120 candles = diminishing returns.
60 candles = 5 hours captures full intraday session structure.

**SHAP DeepExplainer for model interpretability**
In finance, knowing WHY a model predicted something is as important as the prediction.
SHAP values decompose each prediction into per-feature contributions.

---

## 🔮 Future Roadmap

- [ ] FastAPI microservice wrapping the prediction engine
- [ ] React + Tailwind frontend replacing Streamlit
- [ ] WebSocket / SSE real-time signal streaming
- [ ] Docker + Docker Compose containerisation
- [ ] MongoDB for prediction and backtest result storage
- [ ] Binance WebSocket for live crypto tick data
- [ ] Multi-timeframe ensemble (1m + 5m + 15m)

---

## 📝 What I Learned

- Walk-forward validation and why random split breaks time-series models
- How CNN and LSTM layers complement each other for financial sequences
- Why 53% accuracy is a meaningful edge in HFT (statistical expectation value)
- SHAP explainability for neural networks using DeepExplainer
- Production-style modular Python architecture with pytest and CI/CD

---

## ⚠️ Disclaimer

*This project is for educational and research purposes only.
It does not constitute financial advice.
Do not use this for real trading without thorough independent validation.*

---

*Built with ❤️ using TensorFlow · Streamlit · Plotly · yfinance · SHAP · XGBoost*
