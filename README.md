# 🛢️ Crude Oil Prediction & Live Trading Dashboard

A hybrid **CNN-LSTM** deep learning project for predicting short-term price movements of Crude Oil Futures (`CL=F`) on 5-minute intervals. The project includes a full research pipeline and a real-time Streamlit dashboard for signal generation and backtesting.

## 🚀 Features
- **Live Signal Predictor**: Real-time direction (BUY/SELL) prediction for the current market candle.

- **CNN-LSTM Architecture**: Deep learning model optimized for spatial and temporal pattern extraction in time-series data.

- **Interactive Backtesting**: Test your strategy against the last 30 days of market data with custom capital settings.

- **Automated Feature Engineering**: Includes RSI, MACD, Volatility, and Log Returns calculation.

## 📂 Project Structure
- `backtests/`: Contains the live prediction engine and Streamlit app.
- `model/`: Saved weights (`.keras`) and feature scaler (`.joblib`).
- `data/`: Data scripts using `yfinance`.
- `experiments/`: Jupyter notebooks with the model training and architecture research.

##  Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/commodity-trading-project.git
   cd commodity-trading-project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

### 1. Launch the Live Dashboard
Run the Streamlit app to view real-time signals and perform backtests:
```bash
streamlit run backtests/live_backtest_app.py
```

### 2. Download Fresh Data
If you want to update the local CSV for research:
```bash
python data/download_data.py
```

### 3. Model Research
Explore the model development in the Jupyter notebook:
`experiments/crude_cnn_lstm_5m/crude_cnn_lstm.ipynb`

## 📊 Strategy Details
- **Interval**: 5-Minute
- **Symbol**: CL=F (Crude Oil)
- **Signal Logic**:
  - **BUY**: Model probability > 0.50
  - **SELL**: Model probability < 0.50

## 📝 Note on Market Hours
The dashboard reflects live market data. Since Crude Oil Futures close on weekends, the dashboard will display the final Friday close signal until the market re-opens on Sunday evening.

---
*Use for educational purposes only.*
