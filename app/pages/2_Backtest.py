"""
app/pages/2_Backtest.py
Location: F:/commodity_trading_project/app/pages/2_Backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import SUPPORTED_ASSETS
from src.backtest.engine import LiveBacktester
from src.backtest.metrics import compute_all

st.set_page_config(page_title="Backtest", page_icon="📈", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Backtest Settings")

asset_labels    = {f"{v['name']} ({k})": k for k, v in SUPPORTED_ASSETS.items()}
selected_label  = st.sidebar.selectbox("Asset", list(asset_labels.keys()))
selected_symbol = asset_labels[selected_label]

period = st.sidebar.selectbox(
    "History Period", ["1mo", "3mo", "1wk"], index=0,
)
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", value=10_000, step=1_000,
)
threshold = st.sidebar.slider("Signal Threshold", 0.50, 0.75, 0.55, 0.01)

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
st.title("📈 Backtest")
st.caption(f"Strategy simulation for **{selected_label}** over **{period}**")
st.divider()

if not st.button("▶ Run Backtest", type="primary"):
    st.info("Configure settings in the sidebar then click **Run Backtest**.")
    st.stop()

try:
    bt = get_backtester(selected_symbol)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

with st.spinner("Running strategy simulation..."):
    try:
        results = bt.run_backtest(
            initial_capital=initial_capital,
            threshold=threshold,
            period=period,
        )
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

if results is None or results.empty:
    st.error("No results. Try a longer period.")
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────
m = compute_all(results)
st.subheader("📊 Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Return",  f"{m['total_return']}%",
          delta=f"{m['total_return'] - m['market_return']:.2f}% vs market")
c2.metric("Win Rate",      f"{m['win_rate']}%")
c3.metric("Sharpe Ratio",  m["sharpe_ratio"])
c4.metric("Max Drawdown",  f"{m['max_drawdown']}%")

c5, c6, c7, _ = st.columns(4)
c5.metric("Market Return",  f"{m['market_return']}%")
c6.metric("Profit Factor",  m["profit_factor"])
c7.metric("Calmar Ratio",   m["calmar_ratio"])

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────
st.subheader("💰 Equity Curve vs Market Benchmark")
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=results["Timestamp"], y=results["Equity"],
    name="Strategy",
    line=dict(color="#00C853", width=2),
    fill="tozeroy", fillcolor="rgba(0,200,83,0.05)",
))
fig_eq.add_trace(go.Scatter(
    x=results["Timestamp"],
    y=initial_capital * results["Cum_Market_Ret"],
    name="Buy & Hold",
    line=dict(color="#888", width=1.5, dash="dash"),
))
fig_eq.update_layout(
    template="plotly_dark", height=420,
    xaxis_title="Date", yaxis_title="Equity (USD)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_eq, use_container_width=True)

# ── Drawdown ──────────────────────────────────────────────────────────
st.subheader("📉 Drawdown Chart")
peak     = results["Equity"].cummax()
drawdown = ((results["Equity"] / peak) - 1) * 100
fig_dd   = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=results["Timestamp"], y=drawdown,
    fill="tozeroy", fillcolor="rgba(213,0,0,0.15)",
    line=dict(color="#D50000", width=1.5), name="Drawdown %",
))
fig_dd.update_layout(
    template="plotly_dark", height=280,
    xaxis_title="Date", yaxis_title="Drawdown (%)",
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_dd, use_container_width=True)

# ── Prediction distribution ───────────────────────────────────────────
st.subheader("🎯 Prediction Probability Distribution")
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=results["Prediction"], nbinsx=40,
    marker_color="#7C4DFF", opacity=0.8,
))
fig_dist.add_vline(
    x=threshold, line_dash="dash", line_color="orange",
    annotation_text=f"Threshold {threshold}",
)
fig_dist.add_vline(x=0.5, line_dash="dot", line_color="gray")
fig_dist.update_layout(
    template="plotly_dark", height=320,
    xaxis_title="Model Probability", yaxis_title="Count",
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_dist, use_container_width=True)

# ── Trade log ─────────────────────────────────────────────────────────
with st.expander("📋 Sample Trade Log (last 20 rows)"):
    display_cols = ["Timestamp", "Price", "Prediction", "Signal",
                    "Next_Ret", "Strategy_Ret", "Equity"]
    st.dataframe(
        results[display_cols].tail(20).style.format({
            "Price":        "${:.2f}",
            "Prediction":   "{:.4f}",
            "Next_Ret":     "{:.4f}",
            "Strategy_Ret": "{:.4f}",
            "Equity":       "${:,.2f}",
        }),
        use_container_width=True,
    )
