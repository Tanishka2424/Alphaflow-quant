# 🏗️ Architecture Decisions

This document explains every major design decision in AlphaFlow Quant
and the reasoning behind each choice.

---

## 1. Why CNN-LSTM instead of pure LSTM or pure CNN?

**Problem:** Financial time-series has two distinct structures:
- *Local patterns* — head and shoulders, breakouts, candlestick formations
- *Sequential dependencies* — how momentum evolves over hours

**Pure LSTM** captures sequences but treats each timestep independently,
missing local structural patterns within a window.

**Pure CNN** detects local patterns but has no concept of time ordering —
it treats the sequence like a static image.

**CNN-LSTM hybrid** uses CNN to extract local spatial features first,
then feeds the compressed representation into LSTM for temporal modelling.
This is why it outperforms either model alone on financial data.

---

## 2. Why 60-candle lookback window?

| Window | Hours | Verdict |
|--------|-------|---------|
| 12 | 1h | Too short — random noise dominates |
| 24 | 2h | Better — captures some session structure |
| **60** | **5h** | **Optimal — full intraday session context** |
| 120 | 10h | Diminishing returns, higher memory cost |

60 candles = 5 hours captures the London-NY overlap session,
which is the highest liquidity period for Crude Oil and Gold.

---

## 3. Why RobustScaler instead of StandardScaler?

Financial data contains extreme outliers — flash crashes, news-driven spikes,
circuit breakers. StandardScaler (mean/std) is heavily distorted by these.

RobustScaler uses **median and IQR** (interquartile range):
```
scaled = (x - median) / IQR
```
This makes feature scaling resistant to outliers — critical for commodity data
where a single news event can move prices 5% in one candle.

---

## 4. Why TimeSeriesSplit instead of random train/test split?

Random split on time-series causes **lookahead bias**:
```
Random split example (WRONG):
  Train: [Jan, Mar, May, Jul]  ← includes future data
  Test:  [Feb, Apr, Jun]       ← tests on past data

TimeSeriesSplit (CORRECT):
  Fold 1: Train [Jan-Feb]    Test [Mar]
  Fold 2: Train [Jan-Apr]    Test [May]
  Fold 3: Train [Jan-Jun]    Test [Jul]
```

With random split, the model "knows the future" during training,
producing artificially inflated accuracy that collapses in live trading.
TimeSeriesSplit guarantees training data always precedes test data in time.

---

## 5. Why XGBoost as baseline?

Without a baseline, "53% accuracy" is a meaningless number.
We need to answer: *53% compared to what?*

XGBoost was chosen because:
- It uses the same 8 features as CNN-LSTM (fair comparison)
- It's a strong model on its own (not a toy baseline)
- It has no sequence/temporal modelling → isolates the value of CNN-LSTM
- Its feature importances can be compared against CNN-LSTM SHAP values

If XGBoost achieved 53% too, the CNN-LSTM would add no value.
The fact that CNN-LSTM outperforms XGBoost proves temporal context matters.

---

## 6. Why Streamlit instead of React + FastAPI?

**Short answer:** Deployable working demo beats unfinished architecture.

Streamlit was chosen for the initial version because:
- Single Python file per page — no frontend/backend coordination
- Native Plotly integration — candlestick charts in 5 lines
- `st.cache_resource` — model loads once, cached across reruns
- Streamlit Cloud — free one-click deployment with live URL

The future roadmap includes migrating to FastAPI + React for production,
but for a research/internship project, a working Streamlit demo
deployed live is more valuable than a half-built React app.

---

## 7. Why modular src/ structure?

The original project had all logic in one file (`live_backtest.py`).
Splitting into modules provides:

```
src/features/engineering.py  → testable independently
src/backtest/engine.py        → imports from features, testable
src/backtest/metrics.py       → pure functions, 100% testable
src/model/predictor.py        → isolated model concerns
src/model/baseline.py         → isolated comparison model
```

Each module has a single responsibility and can be tested,
imported, and modified without touching unrelated code.
This is the **separation of concerns** principle.

---

## 8. Why SHAP for explainability?

In finance, model predictions must be **justifiable**.
A black-box "the model said SELL" is not acceptable in:
- Risk management reviews
- Regulatory compliance
- Strategy debugging

SHAP (SHapley Additive exPlanations) decomposes each prediction
into per-feature contributions with mathematical guarantees:
- **Consistency:** if a feature's impact increases, its SHAP value increases
- **Accuracy:** SHAP values sum to the actual model output
- **Missingness:** features not used get zero SHAP value

DeepExplainer was chosen specifically for Keras models as it
uses a background distribution to compute expected SHAP values
efficiently without retraining the model.

---

## 9. Why 75 pytest tests?

Tests serve three purposes in this project:

1. **Correctness verification** — RSI must be 0-100, MACD must have two columns,
   drawdown must be ≤ 0. These are mathematical facts that can be automated.

2. **Regression prevention** — when refactoring code, tests catch breaking changes
   before they reach the dashboard or model.

3. **CI/CD signal** — the green badge on GitHub proves the code is tested,
   not just written. Recruiters and reviewers can verify this instantly.

---

*Architecture document created: May 2026*
