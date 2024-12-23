import asyncio
import time
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

class Async_Cache():
    def __init__(self, seconds_til_expire=120, interval=60) -> None:
        self.cache = {}
        self.seconds_til_expire = seconds_til_expire
        self.interval = interval
        asyncio.create_task(self.clean_up())

    async def set_key_value(self, key, value):
        self.cache[key] = [value, time.time()]

    async def get_value_from_key(self, key):
        if key in self.cache:
            value = self.cache[key]
            if time.time() - value[1] < self.seconds_til_expire:
                return value[0]
            else:
                del self.cache[key]
        return None

    async def clean_up(self):
        while True:
            await asyncio.sleep(self.interval)
            current_time = time.time()
            keys_to_delete = [key for key, (_, timestamp) in self.cache.items()
                                if current_time - timestamp >= self.seconds_til_expire]
            for key in keys_to_delete:
                del self.cache[key]

async def get_stock_data(cache, tickers, start, end):
    data = {}

    for ticker in tickers:
        key = f"{ticker}_{start}_{end}"
        cached_data = await cache.get_value_from_key(key)
        if cached_data is not None:
            data[ticker] = cached_data
            continue

        try:
            df = pd.read_csv(f'../CSVs/{ticker}_{start}_{end}_returns.csv', index_col=0, parse_dates=True)
            if cache:
                await cache.set_key_value(key, df)

            data[ticker] = df
        except Exception as e:
            df = yf.download(ticker, start=start, end=end)
            if cache:
                await cache.set_key_value(key, df)
            df.to_csv(f'../CSVs/{ticker}_{start}_{end}_returns.csv')
            data[ticker] = df

    return data

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

class RBF:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2):
        # Compute pairwise squared distances between points in X1 and X2
        dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
        # Apply the RBF kernel formula
        return np.exp(-0.5 * dists / self.length_scale**2)

class GaussianProcessRegressor:
    def __init__(self, kernel, alpha=1e-6):
        self.kernel = kernel
        self.alpha = alpha
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        # Compute the training covariance matrix with noise regularization
        K = self.kernel(X_train, X_train) + self.alpha * np.eye(len(X_train))
        # Precompute the inverse of the covariance matrix for later use
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test):
        # Compute the covariance between training points and test points
        K_s = self.kernel(self.X_train, X_test)
        
        # Compute the predictive mean
        mu = K_s.T @ self.K_inv @ self.y_train
        return mu

async def main():
    cache = Async_Cache()

    tickers = ["QQQ"]
    start_date = '2021-01-01'
    end_date = '2024-01-01'

    stock_data = await get_stock_data(cache, tickers, start_date, end_date)

    for ticker, df in stock_data.items():
        df = df.copy()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()

        log_returns = df['log_returns'].values
        seq_length = 10
        X, y = create_sequences(log_returns, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        # Flatten sequences for regression model
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Gaussian Process Regressor
        kernel = RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        # Train
        gpr.fit(X_train, y_train)

        # Predict
        predictions = gpr.predict(X_test)

        # Plot results
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2)
        ))

        fig.update_layout(
            title=f"Gaussian Process Regression Predictions vs Actual for {ticker}",
            xaxis_title="Index",
            yaxis_title="Log Returns",
            legend=dict(x=0, y=1),
            template="plotly_dark"
        )

        fig.show()

if __name__ == "__main__":
    asyncio.run(main())