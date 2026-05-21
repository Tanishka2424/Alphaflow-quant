import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from live_backtest import LiveBacktester

st.set_page_config(page_title="Crude Oil Live Signal & Backtest", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #30363d;
    }
    .buy-signal {
        background-color: rgba(0, 255, 0, 0.1);
        border-color: #00ff00;
    }
    .sell-signal {
        background-color: rgba(255, 0, 0, 0.1);
        border-color: #ff0000;
    }
    </style>
    """, unsafe_allow_html=True)



# Sidebar settings
st.sidebar.header("Strategy Settings")
symbol = st.sidebar.text_input("Symbol", "CL=F")
threshold = 0.5 # Default decision boundary
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000)
period = st.sidebar.selectbox("Backtest History", ["1mo", "1wk", "3mo"], index=0)

# Dynamically determine the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

model_path = os.path.join(base_dir, "model", "crude_cnn_lstm.keras")
scaler_path = os.path.join(base_dir, "model", "scaler.joblib")

@st.cache_resource
def get_tester(model_p, scaler_p, sym):
    t = LiveBacktester(model_p, scaler_p, symbol=sym)
    t.load_resources()
    return t

tester = get_tester(model_path, scaler_path, symbol)

# Market Status Logic (Converted to IST)
# IST = UTC + 5:30
ist_tz = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(ist_tz)

# Crude Oil (CL=F) Market Hours in IST:
# Closes: Friday 5:00 PM ET  -> Saturday 3:30 AM IST
# Opens:  Sunday 6:00 PM ET  -> Monday 4:30 AM IST

day = now_ist.weekday() # 0=Mon, 5=Sat, 6=Sun
hour_float = now_ist.hour + now_ist.minute/60.0

market_open = True
if day == 5: # Saturday
    if hour_float >= 3.5: # After 3:30 AM IST
        market_open = False
elif day == 6: # Sunday
    market_open = False
elif day == 0: # Monday
    if hour_float < 4.5: # Before 4:30 AM IST
        market_open = False

# UI - Header & Refresh
c_head, c_status = st.columns([3, 1])
with c_head:
    st.title("🛢️ Crude Oil CNN-LSTM Live Signal")
with c_status:
    if market_open:
        st.success("🟢 Market Open")
    else:
        st.error("🔴 Market Closed (Weekend)")
    if st.button("🔄 Force Refresh"):
        st.rerun()

# Auto-refresh every 5 minutes (300 seconds)
# We show the refresh even if market is closed so the user knows the app is alive
st_autorefresh(interval=300 * 1000, key="data_refresh")

# Live Signal Section
st.subheader("🔥 Current Market Signal")
with st.spinner("Fetching latest market data..."):
    live_data, error = tester.get_latest_signal(threshold=threshold)
    
    if live_data:
        is_buy = "BUY" in live_data['signal']
        sig_class = "buy-signal" if is_buy else "sell-signal"
        
        # Convert data timestamp to IST for display
        display_ts = live_data['timestamp'].astimezone(ist_tz) if live_data['timestamp'].tzinfo else live_data['timestamp']
        
        st.markdown(f"""
            <div class="signal-box {sig_class}">
                <h1 style='margin:0; color:{"#00ff00" if is_buy else "#ff0000"}'>{live_data['signal']}</h1>
                <p style='font-size:1.2em; color:#8b949e'>Confidence: {live_data['confidence']} | Price: ${live_data['price']:.2f}</p>
                <p style='font-size:0.9em; color:#484f58'>Point Time (IST): {display_ts.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Probability", f"{live_data['probability']:.4f}")
        c2.metric("Market Sentiment", "Bullish" if is_buy else "Bearish")
        c3.metric("Current Window", "60 candles (5m)")

        if not market_open:
            st.warning(f"Note: The market is currently **CLOSED**. The signal shown above is based on the final trading candle of Friday ({live_data['timestamp'].strftime('%b %d, %H:%M')}). Real-time updates will resume when the market opens on Sunday evening.")
    else:
        st.error(error)

st.divider()

# Backtest Section
st.header("📈 Historical Backtest Performance")
if st.button("Run Full Backtest Analysis"):
    with st.spinner("Analyzing historical performance..."):
        try:
            results = tester.run_backtest(initial_capital=initial_capital, threshold=threshold)
            
            if results is None or len(results) == 0:
                st.error("No backtest results available for the selected parameters. Try a longer history or different symbol.")
                st.stop()

            # Display Metrics
            m1, m2, m3, m4 = st.columns(4)
            
            total_ret = (results['Cum_Strategy_Ret'].iloc[-1] - 1) * 100
            market_ret = (results['Cum_Market_Ret'].iloc[-1] - 1) * 100
            win_rate = (results['Strategy_Ret'] > 0).mean() * 100
            sharpe = np.sqrt(252 * 24 * 12) * results['Strategy_Ret'].mean() / results['Strategy_Ret'].std()
            
            m1.metric("Total Return", f"{total_ret:.2f}%")
            m2.metric("Market Return", f"{market_ret:.2f}%")
            m3.metric("Win Rate", f"{win_rate:.2f}%")
            m4.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # Plot Equity Curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['Timestamp'], y=results['Equity'], name='Strategy Equity', line=dict(color='#00ff00', width=2)))
            fig.add_trace(go.Scatter(x=results['Timestamp'], y=initial_capital * results['Cum_Market_Ret'], name='Market Benchmark', line=dict(color='#8b949e', dash='dash')))
            
            fig.update_layout(
                title=f"Equity Evolution ({symbol})",
                xaxis_title="Time",
                yaxis_title="Equity (USD)",
                template="plotly_dark",
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown or Prediction Distribution
            st.subheader("Prediction Probability Distribution")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=results['Prediction'], nbinsx=40, marker_color='#7928ca', opacity=0.75))
            fig_dist.add_vline(x=threshold, line_dash="dash", line_color="orange", annotation_text="Active Threshold")
            fig_dist.update_layout(template="plotly_dark", height=400, bargap=0.1)
            st.plotly_chart(fig_dist, use_container_width=True)
            
        except Exception as e:
            st.error(f"Backtest error: {e}")
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("Click the button above to analyze how this threshold performed in the past.")

# Auto-refresh info
st.sidebar.markdown("---")
st.sidebar.caption(f"Last sync (IST): {now_ist.strftime('%H:%M:%S')}")
if st.sidebar.button("Manual Sync Now"):
    st.rerun()
