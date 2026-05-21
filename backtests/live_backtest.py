import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from joblib import load
import os
import matplotlib.pyplot as plt

class LiveBacktester:
    def __init__(self, model_path, scaler_path, symbol="CL=F"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self.window_size = 60 # Defaulting to 60 as seen in the notebook

    def load_resources(self):
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Loading scaler from {self.scaler_path}...")
        self.scaler = load(self.scaler_path)

    def fetch_data(self, period="1mo", interval="5m"):
        print(f"Fetching data for {self.symbol} (period={period}, interval={interval})...")
        data = yf.download(self.symbol, period=period, interval=interval, progress=False)
        # Handle multi-index columns if present (recent yfinance change)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Clean data
        data = data.dropna()
        print(f"Data fetched. Shape: {data.shape}")
        return data

    def calculate_features(self, df):
        df = df.copy()
        # Same logic as in enhance_features.py/notebook
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['hl_range'] = df['High'] - df['Low']
        df['co_range'] = df['Close'] - df['Open']
        df['volatility'] = df['log_ret'].rolling(window=14).std()
        df['volume_pct'] = df['Volume'].pct_change()

        # RSI (14 period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Handle inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        feature_cols = ['log_ret', 'hl_range', 'co_range', 'volatility', 'volume_pct', 'rsi', 'macd', 'macd_signal']
        df_features = df[feature_cols].dropna()
        
        # We also need the Close prices to calculate trade returns later
        final_df = df.loc[df_features.index].copy()
        return final_df, df_features

    def run_backtest(self, initial_capital=10000, threshold=0.5):
        raw_data = self.fetch_data()
        full_df, features_df = self.calculate_features(raw_data)
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        X = []
        #  the prices corresponding to the end of each window
        prices = []
        timestamps = []

        for i in range(len(scaled_features) - self.window_size):
            X.append(scaled_features[i : i + self.window_size])
            prices.append(full_df['Close'].iloc[i + self.window_size - 1])
            timestamps.append(full_df.index[i + self.window_size - 1])

        X = np.array(X)
        print(f"Prepared {len(X)} windows for prediction.")

        # Predict
        print("Scoring model...")
        predictions = self.model.predict(X, verbose=0)
        
        # Trading Simulation
        # Prediction i is for the NEXT candle after window i (which ends at index i + window_size - 1)
        # So at timestamp[i], we make a trade for the next 5 min interval.
        
        results = pd.DataFrame({
            'Timestamp': timestamps,
            'Price': prices,
            'Prediction': predictions.flatten()
        })
        
        # Calculate actual next-candle returns
        # The next price for results.iloc[i] is full_df['Close'].iloc[i + window_size]
        next_prices = full_df['Close'].iloc[self.window_size:].values
        # Match lengths if necessary
        n = min(len(results), len(next_prices))
        results = results.iloc[:n].copy()
        results['Next_Price'] = next_prices[:n]
        results['Next_Ret'] = (results['Next_Price'] - results['Price']) / results['Price']
        
        # Strategy: Long if Prediction > threshold, Short otherwise
        results['Signal'] = np.where(results['Prediction'] > threshold, 1, -1)
        results['Strategy_Ret'] = results['Signal'] * results['Next_Ret']
        
        # Cumulative Returns
        results['Cum_Market_Ret'] = (1 + results['Next_Ret']).cumprod()
        results['Cum_Strategy_Ret'] = (1 + results['Strategy_Ret']).cumprod()
        
        results['Equity'] = initial_capital * results['Cum_Strategy_Ret']
        
        # Metrics
        total_return = (results['Cum_Strategy_Ret'].iloc[-1] - 1) * 100
        market_return = (results['Cum_Market_Ret'].iloc[-1] - 1) * 100
        win_rate = (results['Strategy_Ret'] > 0).mean() * 100
        sharpe = np.sqrt(252 * 24 * 12) * results['Strategy_Ret'].mean() / results['Strategy_Ret'].std() # 5m candles per year approx
        
        print("\n" + "="*30)
        print("      BACKTEST RESULTS")
        print("="*30)
        print(f"Symbol:           {self.symbol}")
        print(f"Initial Capital:  ${initial_capital}")
        print(f"Final Equity:     ${results['Equity'].iloc[-1]:.2f}")
        print(f"Total Return:     {total_return:.2f}%")
        print(f"Market Return:    {market_return:.2f}%")
        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Sharpe Ratio:     {sharpe:.2f}")
        print("="*30)
        
        self.plot_results(results)
        return results

    def get_latest_signal(self, threshold=0.5):
        # Fetch just enough data for one window (window_size + padding for features)
        # period="5d" is safe for 5m interval to get enough lookback for features (RSI/MACD need ~30+ rows)
        raw_data = self.fetch_data(period="5d", interval="5m")
        full_df, features_df = self.calculate_features(raw_data)
        
        # Get the last 'window_size' rows
        last_window_features = features_df.tail(self.window_size)
        
        if len(last_window_features) < self.window_size:
            return None, f"Insufficient data for live signal. Got {len(last_window_features)}/{self.window_size}"

        # Scale and Reshape for model
        scaled_window = self.scaler.transform(last_window_features)
        X_live = np.array([scaled_window]) # Shape (1, 60, 8)
        
        # Predict
        prob = self.model.predict(X_live, verbose=0)[0][0]
        
        signal = "BUY (UP)" if prob > threshold else "SELL (DOWN)"
        confidence = prob if prob > threshold else (1 - prob)
        
        return {
            "timestamp": features_df.index[-1],
            "price": full_df['Close'].iloc[-1],
            "probability": prob,
            "signal": signal,
            "confidence": f"{confidence*100:.2f}%"
        }, None

    def plot_results(self, results):
        plt.figure(figsize=(12, 6))
        plt.plot(results['Timestamp'], results['Equity'], label='Strategy Equity', color='blue')
        plt.title(f'Live Backtest: {self.symbol} Performance')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        output_plot = "backtests/equity_curve.png"
        plt.savefig(output_plot)
        print(f"Equity curve saved to {output_plot}")

if __name__ == "__main__":
    # Dynamically determine the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    model_path = os.path.join(base_dir, "model", "crude_cnn_lstm.keras")
    scaler_path = os.path.join(base_dir, "model", "scaler.joblib")
    
    tester = LiveBacktester(model_path, scaler_path)
    tester.load_resources()
    tester.run_backtest()
