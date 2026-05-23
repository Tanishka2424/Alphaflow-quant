"""
app/main.py
-----------
Entry point for the multi-page Streamlit dashboard.
Run with: streamlit run app/main.py

Location: F:/commodity_trading_project/app/main.py
"""

import sys
import os

# Make project root importable from app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="AlphaFlow Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/combo-chart.png",
    width=60,
)
st.sidebar.title("AlphaFlow Quant")
st.sidebar.caption("AI-powered trading signals")
st.sidebar.divider()

st.sidebar.markdown("""
**Navigation**
- 🔥 Live Signal
- 📈 Backtest
- 📊 Performance
- 🧠 Model Explainer
""")

st.sidebar.divider()
st.sidebar.caption("Built with CNN-LSTM · TensorFlow · Streamlit")

# ── Landing page ──────────────────────────────────────────────────────
st.title("📈 AlphaFlow Quant")
st.subheader("AI-Powered Trading Intelligence Platform")

st.markdown("""
Welcome to **AlphaFlow Quant** — a deep learning platform for predicting
short-term price direction across commodity and crypto markets.

---
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model",      "CNN-LSTM")
col2.metric("Accuracy",   "~53%")
col3.metric("Assets",     "3")
col4.metric("Interval",   "5-min")

st.markdown("""
---
### How to use
Use the **sidebar** to navigate between pages:

| Page | What it does |
|------|-------------|
| 🔥 **Live Signal** | Real-time BUY/SELL prediction for any asset |
| 📈 **Backtest** | Simulate strategy on 30 days of historical data |
| 📊 **Performance** | Sharpe ratio, drawdown, profit factor charts |
| 🧠 **Model Explainer** | SHAP feature importance — why did the model predict this? |

---
### Assets supported
| Symbol | Asset | Market |
|--------|-------|--------|
| CL=F | Crude Oil Futures | Commodity |
| GC=F | Gold Futures | Commodity |
| BTC-USD | Bitcoin | Crypto |
""")
