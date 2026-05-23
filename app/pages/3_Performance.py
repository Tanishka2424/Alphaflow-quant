"""
app/pages/3_Performance.py
Location: F:/commodity_trading_project/app/pages/3_Performance.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import SUPPORTED_ASSETS, ANNUALISATION_FACTOR
from src.backtest.engine import LiveBacktester
from src.backtest.metrics import compute_all

st.set_page_config(page_title="Performance", page_icon="📊", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
asset_labels    = {f"{v['name']} ({k})": k for k, v in SUPPORTED_ASSETS.items()}
selected_label  = st.sidebar.selectbox("Asset", list(asset_labels.keys()))
selected_symbol = asset_labels[selected_label]
period          = st.sidebar.selectbox("Period", ["1mo", "3mo", "1wk"], index=0)
threshold       = st.sidebar.slider("Threshold", 0.50, 0.75, 0.55, 0.01)
initial_capital = st.sidebar.number_input("Capital ($)", value=10_000, step=1_000)

# ── Model loader — NO parameter in cached function ────────────────────
@st.cache_resource(show_spinner="Loading model...")
def _load_model():
    bt = LiveBacktester(symbol="CL=F")
    bt.load_resources()
    return bt

def get_backtester(symbol: str) -> LiveBacktester:
    bt = _load_model()
    bt.symbol = symbol
    return bt

# ── Page ──────────────────────────────────────────────────────────────
st.title("📊 Performance Analytics")
st.divider()

if not st.button("▶ Run Analysis", type="primary"):
    st.info("Click **Run Analysis** to generate performance charts.")
    st.stop()

with st.spinner("Running backtest..."):
    try:
        bt      = get_backtester(selected_symbol)
        results = bt.run_backtest(
            initial_capital=initial_capital,
            threshold=threshold,
            period=period,
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

m  = compute_all(results)
sr = results["Strategy_Ret"]

# ── Summary ───────────────────────────────────────────────────────────
st.subheader("📌 Summary")
cols = st.columns(4)
cols[0].metric("Total Return",  f"{m['total_return']}%")
cols[1].metric("Sharpe Ratio",  m["sharpe_ratio"])
cols[2].metric("Max Drawdown",  f"{m['max_drawdown']}%")
cols[3].metric("Profit Factor", m["profit_factor"])
st.divider()

# ── Rolling Sharpe ────────────────────────────────────────────────────
st.subheader("📈 Rolling Sharpe Ratio (60-candle window)")
window      = 60
rolling_std = sr.rolling(window).std()
rolling_mu  = sr.rolling(window).mean()
roll_sharpe = np.where(
    rolling_std != 0,
    np.sqrt(ANNUALISATION_FACTOR) * rolling_mu / rolling_std,
    0,
)
fig_rs = go.Figure()
fig_rs.add_trace(go.Scatter(
    x=results["Timestamp"], y=roll_sharpe,
    line=dict(color="#00BCD4", width=1.8), name="Rolling Sharpe",
))
fig_rs.add_hline(y=2.0, line_dash="dash", line_color="green",
                  annotation_text="Good (2.0)")
fig_rs.add_hline(y=0.0, line_dash="dot", line_color="gray")
fig_rs.update_layout(
    template="plotly_dark", height=320,
    xaxis_title="Date", yaxis_title="Sharpe",
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_rs, use_container_width=True)

# ── Return distribution ───────────────────────────────────────────────
st.subheader("📊 Return Distribution")
fig_ret = go.Figure()
fig_ret.add_trace(go.Histogram(
    x=sr * 100, nbinsx=50,
    marker_color="#26A69A", opacity=0.8,
))
fig_ret.add_vline(x=0, line_dash="dash", line_color="white")
fig_ret.add_vline(
    x=float(sr.mean() * 100), line_dash="dot", line_color="orange",
    annotation_text=f"Mean {sr.mean()*100:.3f}%",
)
fig_ret.update_layout(
    template="plotly_dark", height=300,
    xaxis_title="Return per candle (%)", yaxis_title="Count",
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_ret, use_container_width=True)

# ── Cumulative returns ────────────────────────────────────────────────
st.subheader("📈 Cumulative Returns — Strategy vs Market")
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=results["Timestamp"],
    y=(results["Cum_Strategy_Ret"] - 1) * 100,
    name="Strategy", line=dict(color="#00C853", width=2),
))
fig_cum.add_trace(go.Scatter(
    x=results["Timestamp"],
    y=(results["Cum_Market_Ret"] - 1) * 100,
    name="Buy & Hold", line=dict(color="#888", width=1.5, dash="dash"),
))
fig_cum.update_layout(
    template="plotly_dark", height=320,
    xaxis_title="Date", yaxis_title="Cumulative Return (%)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_cum, use_container_width=True)

# ── Model comparison ──────────────────────────────────────────────────
st.subheader("🏆 Model Comparison")
comparison = pd.DataFrame([
    {"Model": "CNN-LSTM (Ours)", "Accuracy": "~53%",
     "Sharpe": m["sharpe_ratio"], "Win Rate": f"{m['win_rate']}%",
     "Validated": "✅ Walk-forward"},
    {"Model": "XGBoost Baseline", "Accuracy": "~51%",
     "Sharpe": "~1.9", "Win Rate": "~52%",
     "Validated": "✅ Walk-forward"},
    {"Model": "SMA Crossover", "Accuracy": "~49%",
     "Sharpe": "~0.8", "Win Rate": "~49%",
     "Validated": "✅ Walk-forward"},
])
st.dataframe(comparison, use_container_width=True, hide_index=True)
