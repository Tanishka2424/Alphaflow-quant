"""
app/pages/4_Model_Explainer.py
Location: F:/commodity_trading_project/app/pages/4_Model_Explainer.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import SUPPORTED_ASSETS, SHAP_PATH, FEATURE_COLS
from src.backtest.engine import LiveBacktester
from src.features.engineering import build_features

st.set_page_config(page_title="Model Explainer", page_icon="🧠", layout="wide")

FEATURE_DESCRIPTIONS = {
    "rsi":         "Relative Strength Index — measures overbought/oversold momentum (0-100)",
    "macd":        "MACD Line — difference between 12 and 26 period EMAs, shows trend direction",
    "macd_signal": "MACD Signal — 9-period EMA of MACD, used for crossover signals",
    "volatility":  "Rolling 14-period std of log returns — captures market noise level",
    "log_ret":     "Log Return — logarithmic price change, more stable than simple returns",
    "hl_range":    "High-Low Range — intrabar price spread, proxy for intrabar volatility",
    "co_range":    "Close-Open Range — directional pressure within each candle",
    "volume_pct":  "Volume % Change — spike detection, often precedes directional moves",
}

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
asset_labels    = {f"{v['name']} ({k})": k for k, v in SUPPORTED_ASSETS.items()}
selected_label  = st.sidebar.selectbox("Asset", list(asset_labels.keys()))
selected_symbol = asset_labels[selected_label]

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
st.title("🧠 Model Explainer")
st.caption("Understand **why** the CNN-LSTM predicted BUY or SELL using SHAP values.")
st.divider()

# ── SHAP values ───────────────────────────────────────────────────────
if os.path.exists(SHAP_PATH):
    with open(SHAP_PATH) as f:
        shap_data = json.load(f)
    st.success("✅ SHAP values loaded from cache.")
else:
    st.warning("SHAP values not yet computed. Showing demonstration values.")
    shap_data = {
        "rsi":         0.182,
        "macd":        0.143,
        "volatility":  0.091,
        "macd_signal": 0.088,
        "log_ret":     0.071,
        "hl_range":    0.065,
        "co_range":    0.058,
        "volume_pct":  0.044,
    }

# ── SHAP bar chart ────────────────────────────────────────────────────
st.subheader("📊 Feature Importance (Mean |SHAP| Value)")
features = list(shap_data.keys())
values   = list(shap_data.values())
max_val  = max(values) if values else 1
colours  = [
    f"rgba(124,77,255,{0.4 + 0.6*(v/max_val):.2f})"
    for v in values
]
fig_shap = go.Figure(go.Bar(
    x=values, y=features, orientation="h",
    marker_color=colours,
    text=[f"{v:.4f}" for v in values],
    textposition="outside",
))
fig_shap.update_layout(
    template="plotly_dark", height=420,
    xaxis_title="Mean |SHAP Value|",
    yaxis=dict(autorange="reversed"),
    margin=dict(t=20, b=20, r=80),
)
st.plotly_chart(fig_shap, use_container_width=True)

# ── Feature table ─────────────────────────────────────────────────────
st.subheader("📋 Feature Breakdown")
rows = []
for feat, val in shap_data.items():
    rows.append({
        "Feature":     feat,
        "SHAP Value":  round(val, 4),
        "Importance":  f"{'█' * int(val/max_val*20):<20}",
        "Description": FEATURE_DESCRIPTIONS.get(feat, "—"),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.divider()

# ── Architecture explainer ────────────────────────────────────────────
st.subheader("🏗️ How the CNN-LSTM Works")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Input**
- 60 consecutive 5-minute candles
- 8 features per candle
- Shape: `(60, 8)`

**CNN Layer**
- Scans window for spatial patterns
- Detects breakouts, spikes, consolidation

**LSTM Layer**
- Captures how momentum evolved over 5 hours
- Remembers which patterns led to price moves

**Output**
- Sigmoid probability 0.0 → 1.0
- > threshold → BUY
- ≤ threshold → SELL
    """)
with col2:
    st.markdown("""
**Why CNN + LSTM?**
CNN sees patterns but ignores order.
LSTM sees sequences but misses local patterns.
Together they capture both.

**Why 60 candles?**
- 12 = 1 hour — too short, just noise
- 60 = 5 hours — full session structure ✅
- 120 = diminishing returns

**Why 53% is strong:**
Random = 50%. A 3% edge across thousands
of trades compounds into real returns.

**Validation:**
Walk-forward (TimeSeriesSplit).
Model never trains on future data.
    """)
st.divider()

# ── Live feature values ───────────────────────────────────────────────
st.subheader("📡 Current Feature Values")
st.caption("The actual indicator values feeding into the model right now.")
try:
    bt       = get_backtester(selected_symbol)
    raw      = bt.fetch_data(period="5d")
    _, features_df = build_features(raw)
    last_row = features_df.tail(1).T.reset_index()
    last_row.columns = ["Feature", "Current Value"]
    last_row["Description"] = last_row["Feature"].map(
        lambda f: FEATURE_DESCRIPTIONS.get(f, "—")
    )
    last_row["Current Value"] = last_row["Current Value"].round(4)
    st.dataframe(last_row, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Could not fetch live feature values: {e}")
