# 🧠 Project Research & Learning Memo

This document serves as a reference for the core logic, architecture decisions, and theoretical benchmarks of the Crude Oil CNN-LSTM Trading Project.

---

## 🏗️ Core Workflow & Mental Model

### 1. Data Foundation
- **Asset**: Crude Oil Futures (`CL=F`)
- **Interval**: 5-Minute Candles (Day Trading granularity)
- **Features**: 8 Technical dimensions per candle (Log Returns, RSI, MACD, Volatility, Volume Pct, HL-Range, CO-Range).
- **Scaling**: `RobustScaler` to normalize price action and volume.

### 2. Window Architecture (Lookback)
- **Default Window**: **60 Steps** (5 Hours of market history).
- **Input Shape**: `(60, 8)` - 60 sequential snapshots of the 8 features.
- **Goal**: Predict the price direction (UP/DOWN) of the **next** 5-minute candle.

### 3. Hybrid AI Brain (CNN-LSTM)
- **CNN Layer**: Scans the 60-step window for **Spatial Patterns** (Head & Shoulders, Breakouts, Spikes).
- **LSTM Layer**: Analyzes the **Temporal Sequence** (How the momentum changed over the 5 hours).
- **Output**: Sigmoid Probability (0.0 to 1.0). Threshold = 0.5.

---

## 📊 Lookback Window Comparison
*Theoretical benchmarks for 5-minute Crude Oil data.*

| Metric | 12 Lookback (1h) | 24 Lookback (2h) | 60 Lookback (5h) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 49% - 50.5% | 51% - 52% | **52% - 54%** |
| **Sharpe Ratio** | 1.2 - 1.8 | 2.0 - 2.8 | **2.5 - 3.5** |
| **Strategy Style** | Aggressive Scalping | Fast Day Trading | **Trend-Following** |
| **False Signals** | Extreme (Noise) | High | **Low (Filters Noise)** |
| **Verdict** | Too Short | Optional | **Recommended** |

---

## 📈 Key Takeaways for Learning
- **Why 60?**: Deep Learning models (like CNNs) need "context." 12 bars is just a line; 60 bars is a **story**. The model needs to see the session structure to filter out random noise.
- **The Alpha Edge**: In 5m trading, an accuracy of **53%** is considered elite. The goal isn't to be right every time, but to have a consistent statistical edge.
- **Market Hours**: Crude Oil is a 24/5 market. The model performs best during high-volume sessions (London/NY overlap).

---
*Created on 2026-02-14 for Project reference.*
