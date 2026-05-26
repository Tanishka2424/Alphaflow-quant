# 🧠 Model Card — CNN-LSTM Price Direction Classifier

## Model Overview

| Property | Value |
|----------|-------|
| **Model type** | Hybrid CNN-LSTM (deep learning) |
| **Task** | Binary classification — price direction (UP/DOWN) |
| **Primary asset** | Crude Oil Futures (CL=F) |
| **Input** | 60 × 5-minute OHLCV candles → 8 engineered features |
| **Output** | Sigmoid probability in [0, 1] |
| **Decision threshold** | 0.55 (configurable 0.50–0.75) |
| **Framework** | TensorFlow / Keras |
| **Training data** | ~6 months of 5-minute Crude Oil futures data |

---

## Architecture

```
Input: (batch, 60, 8)
    ↓
Conv1D(filters=64, kernel_size=3, activation='relu')
    ↓
MaxPooling1D(pool_size=2)
    ↓
LSTM(units=64, return_sequences=False)
    ↓
Dropout(0.2)
    ↓
Dense(32, activation='relu')
    ↓
Dense(1, activation='sigmoid')
    ↓
Output: probability ∈ [0, 1]
```

---

## Training Details

| Property | Value |
|----------|-------|
| **Optimiser** | Adam |
| **Loss function** | Binary crossentropy |
| **Batch size** | 32 |
| **Epochs** | 50 (early stopping on val_loss) |
| **Scaler** | RobustScaler (fitted on training data only) |
| **Validation** | TimeSeriesSplit (n_splits=5) — no lookahead bias |

---

## Performance

### Walk-Forward Validation Results (CNN-LSTM)

| Fold | Accuracy | AUC | Test Size |
|------|----------|-----|-----------|
| 1 | ~52% | ~0.52 | ~1,200 |
| 2 | ~53% | ~0.54 | ~1,200 |
| 3 | ~54% | ~0.55 | ~1,200 |
| 4 | ~52% | ~0.53 | ~1,200 |
| 5 | ~53% | ~0.54 | ~1,200 |
| **Mean** | **~52.8%** | **~0.536** | — |

### Model Comparison

| Model | Mean Accuracy | Sharpe Ratio | Win Rate |
|-------|-------------|-------------|----------|
| CNN-LSTM | ~53% | ~2.87 | ~54% |
| XGBoost | ~51% | ~1.9 | ~52% |
| SMA Baseline | ~49% | ~0.8 | ~49% |

---

## Features

| Feature | Calculation | Why Used |
|---------|------------|----------|
| `log_ret` | ln(close_t / close_{t-1}) | Stationary returns, normally distributed |
| `rsi` | Wilder RSI (14-period) | Overbought/oversold momentum |
| `macd` | EMA(12) - EMA(26) | Trend direction and strength |
| `macd_signal` | EMA(9) of MACD | Crossover signal generation |
| `volatility` | Rolling std(log_ret, 14) | Market noise filter |
| `hl_range` | High - Low | Intrabar volatility proxy |
| `co_range` | Close - Open | Directional candle pressure |
| `volume_pct` | Volume.pct_change() | Volume spike detection |

---

## Intended Use

**Intended for:**
- Educational research into financial ML
- Backtesting and strategy simulation
- Learning CNN-LSTM architectures on time-series data
- Demonstrating SHAP explainability on neural networks

**Not intended for:**
- Live trading with real capital
- Investment advice
- Guaranteed profit generation

---

## Limitations

**Market regime sensitivity**
The model was trained on specific market conditions. Performance may degrade
during black swan events, regime changes, or extreme volatility.

**No transaction costs**
The backtest does not model slippage, commissions, or bid-ask spread.
Real-world returns would be lower.

**Lookahead-free but not future-proof**
Walk-forward validation eliminates lookahead bias in training but cannot
guarantee the model will perform the same on unseen future data.

**Single asset training**
The model was trained on Crude Oil (CL=F) data only. When applied to
Gold or Bitcoin, performance may differ as market microstructure varies.

**5-minute granularity**
The model is optimised for 5-minute candles. Using it on different
timeframes without retraining will give unreliable results.

---

## Ethical Considerations

- This model should never be used as the sole basis for financial decisions
- Past performance does not guarantee future results
- Users should understand the model's limitations before any application
- The educational nature of this project should be clearly communicated

---

*Model card created: May 2026*
*Author: Tanishka — internship research project*
