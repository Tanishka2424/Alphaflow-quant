"""
app/pages/1_Live_Signal.py
Location: F:/commodity_trading_project/app/pages/1_Live_Signal.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh

from config import SUPPORTED_ASSETS
from src.backtest.engine import LiveBacktester
from src.features.engineering import build_features

st.set_page_config(page_title="Live Signal", page_icon="🔥", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Signal Settings")

asset_labels    = {f"{v['name']} ({k})": k for k, v in SUPPORTED_ASSETS.items()}
selected_label  = st.sidebar.selectbox("Asset", list(asset_labels.keys()))
selected_symbol = asset_labels[selected_label]

threshold = st.sidebar.slider(
    "Signal Threshold", 0.50, 0.75, 0.55, 0.01,
    help="Probability cutoff for BUY signal. Higher = more selective.",
)

st.sidebar.divider()
st.sidebar.markdown("""
**Threshold guide**
- `0.50` → Trade everything
- `0.55` → Balanced ✅
- `0.65` → High confidence only
- `0.75` → Very selective
""")

# ── IST time ──────────────────────────────────────────────────────────
IST     = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)

# ── Model loader — NO parameter in cached function ────────────────────
@st.cache_resource(show_spinner="Loading CNN-LSTM model...")
def _load_model():
    bt = LiveBacktester(symbol="CL=F")
    bt.load_resources()
    return bt

def get_backtester(symbol: str) -> LiveBacktester:
    bt = _load_model()
    bt.symbol = symbol
    return bt

# ── Auto refresh every 5 minutes ─────────────────────────────────────
st_autorefresh(interval=300_000, key="live_refresh")

# ── Header ────────────────────────────────────────────────────────────
st.title("🔥 Live Signal")
col_t, col_r = st.columns([3, 1])
with col_r:
    st.caption(f"🕐 IST: {now_ist.strftime('%d %b %Y  %H:%M:%S')}")
    if st.button("🔄 Refresh Now"):
        st.cache_resource.clear()
        st.rerun()

# ── Market status ─────────────────────────────────────────────────────
day        = now_ist.weekday()
hour_float = now_ist.hour + now_ist.minute / 60.0
market_open = not (
    (day == 5 and hour_float >= 3.5) or
    (day == 6) or
    (day == 0 and hour_float < 4.5)
)

s_col, _ = st.columns([1, 3])
with s_col:
    if market_open:
        st.success("🟢 Market Open")
    else:
        st.warning("🔴 Market Closed")

st.divider()

# ── Fetch signal ──────────────────────────────────────────────────────
try:
    backtester = get_backtester(selected_symbol)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

with st.spinner(f"Fetching {selected_label} data..."):
    signal, error = backtester.get_latest_signal(threshold=threshold)

if error:
    st.error(f"Signal error: {error}")
    st.stop()

# ── Signal card ───────────────────────────────────────────────────────
is_buy    = signal["signal"] == "BUY"
sig_color = "#00C853" if is_buy else "#D50000"
sig_bg    = "rgba(0,200,83,0.08)" if is_buy else "rgba(213,0,0,0.08)"
sig_emoji = "🟢" if is_buy else "🔴"

st.markdown(f"""
<div style="background:{sig_bg};border:1.5px solid {sig_color};
border-radius:16px;padding:28px 32px;text-align:center;margin-bottom:24px;">
    <div style="font-size:3rem;font-weight:700;color:{sig_color}">
        {sig_emoji} {signal['signal']}
    </div>
    <div style="font-size:1.1rem;color:#aaa;margin-top:8px">
        {selected_label} &nbsp;·&nbsp;
        Confidence: <b style="color:{sig_color}">{signal['confidence']}</b>
        &nbsp;·&nbsp; Price: <b>${signal['price']:.2f}</b>
    </div>
    <div style="font-size:0.85rem;color:#666;margin-top:6px">
        Signal time (IST): {signal['timestamp'].strftime('%d %b %Y  %H:%M')}
        &nbsp;·&nbsp; Threshold: {threshold}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Raw Probability",  f"{signal['probability']:.4f}")
m2.metric("Signal",           signal["signal"])
m3.metric("Threshold Active", f"{threshold:.2f}")

# ── Confidence gauge ──────────────────────────────────────────────────
conf_float = float(signal["confidence"].replace("%", "")) / 100
fig_gauge  = go.Figure(go.Indicator(
    mode  = "gauge+number",
    value = conf_float * 100,
    title = {"text": "Model Confidence %", "font": {"size": 16}},
    gauge = {
        "axis": {"range": [50, 100]},
        "bar":  {"color": sig_color},
        "threshold": {
            "line":      {"color": "orange", "width": 3},
            "thickness": 0.75,
            "value":     threshold * 100,
        },
    },
    number = {"suffix": "%", "font": {"size": 40}},
))
fig_gauge.update_layout(
    template="plotly_dark", height=260,
    margin=dict(t=40, b=0, l=30, r=30),
)
st.plotly_chart(fig_gauge, use_container_width=True)

# ── Candlestick chart ─────────────────────────────────────────────────
st.subheader("📊 Price Chart with Indicators")
with st.spinner("Building chart..."):
    try:
        raw     = backtester.fetch_data(period="5d")
        full_df, features_df = build_features(raw)
        plot_df = full_df.tail(100).copy()

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], vertical_spacing=0.05,
            subplot_titles=("OHLCV — Last 100 candles", "RSI (14)"),
        )
        fig.add_trace(go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"],   close=plot_df["Close"],
            name="Price",
            increasing_line_color="#00C853",
            decreasing_line_color="#D50000",
        ), row=1, col=1)
        if "bb_upper" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df["bb_upper"],
                line=dict(color="rgba(100,100,255,0.4)", width=1),
                name="BB Upper",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df["bb_lower"],
                line=dict(color="rgba(100,100,255,0.4)", width=1),
                fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
                name="BB Lower",
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["rsi"],
            line=dict(color="#FFA726", width=1.5), name="RSI",
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(
            template="plotly_dark", height=520,
            showlegend=False, xaxis_rangeslider_visible=False,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart unavailable: {e}")

if not market_open:
    st.info(
        "Market is currently closed. Signal shown is based on the "
        "last available candle. Updates resume when market reopens."
    )
