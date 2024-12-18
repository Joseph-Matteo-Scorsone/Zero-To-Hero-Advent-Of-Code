import asyncio
import time
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
        cached_data = await cache.get_value_from_key(ticker)
        if cached_data is not None:
            data[ticker] = cached_data
            continue

        try:
            df = pd.read_csv(f'../CSVs/{ticker}_returns.csv', index_col=0, parse_dates=True)
            if cache:
                await cache.set_key_value(ticker, df)

            data[ticker] = df
        except Exception as e:
            df = yf.download(ticker, start=start, end=end)
            if cache:
                await cache.set_key_value(ticker, df)
            df.to_csv(f'../CSVs/{ticker}_returns.csv')
            data[ticker] = df

    return data

def calculate_GARCH(returns):
    def garch_likelihood(params):
        omega, alpha, beta = params
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)  # initialize with the variance of the ts
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        likelihood = 0.5 * np.sum(np.log(2 * np.pi * sigma2) + (returns**2 / sigma2))
        return likelihood

    # initial guesses
    initial_params = [0.1, 0.1, 0.8]
    # bounds for values
    bounds = [(1e-6, None), (0, 1), (0, 1)]
    # use scipy to minimize
    result = minimize(garch_likelihood, initial_params, bounds=bounds)
    return result.x if result.success else initial_params

def GARCH_vol_and_returns(data):
    data = data.copy()
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()

    garch_params = calculate_GARCH(data['log_returns'].values)
    omega, alpha, beta = garch_params

    sigma2 = np.zeros(len(data))
    sigma2[0] = np.var(data['log_returns'])  # initialize with the variance of the ts
    for t in range(1, len(data)):
        sigma2[t] = omega + alpha * data['log_returns'].iloc[t-1]**2 + beta * sigma2[t-1]

    data['GARCH'] = np.sqrt(sigma2)

    annualized_return = data['log_returns'].sum() * (252 / len(data))
    annualized_volatility = data['GARCH'].sum() * (252 / len(data))

    return annualized_return, annualized_volatility

def kmeans(X, n_clusters, max_iters=100):
    #X: numpy array of shape (n_samples, n_features)
    tol=1e-4

    # Randomly initialize centroids by selecting `n_clusters` random points from the dataset
    np.random.seed(2024)
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iters):
        # Calculate the Euclidean distance of each point to all centroids
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=0)
        
        # Save current centroids for convergence check
        prev_centroids = centroids.copy()
        
        # Update centroids as the mean of all points assigned to each cluster
        for i in range(n_clusters):
            if np.any(labels == i):  # Avoid recalculating for empty clusters
                centroids[i] = np.mean(X[labels == i], axis=0)
        
        # Check for convergence: stop if centroids move less than `tol`
        if np.all(np.abs(centroids - prev_centroids) < tol):
            break
    
    return centroids, labels

async def main():
    cache = Async_Cache()
    
    tickers = ["GS", "UNH", "MSFT", "HD", "CAT", "SHW", "CRM", "V", "AXP", "MCD", 
               "AMGN", "TRV", "JPM", "AAPL", "IBM", "HON", "AMZN", "PG", "CVX", 
               "BA", "JNJ", "NVDA", "MMM", "DIS", "MRK", "WMT", "NKE", "KO", "CSCO", "VZ"]
    
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    stock_data = await get_stock_data(cache, tickers, start_date, end_date)
    
    # Prepare data for K-means
    X = []
    for ticker, df in stock_data.items():
        annualized_return, annualized_volatility = GARCH_vol_and_returns(df)
        X.append([annualized_return, annualized_volatility])
        
    X = np.array(X)
    
    n_clusters = 5
    centroids, labels = kmeans(X, n_clusters)
    
    # Create Plotly figure
    fig = go.Figure()

    # Add scatter plot for stock data
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[X[i, 0]], y=[X[i, 1]],
            mode='markers+text',
            marker=dict(color=labels[i], colorscale='Viridis', size=8),
            text=[ticker],
            textposition="top center",
            name=f"Cluster {labels[i]}"
        ))

    # Add centroids
    for centroid in centroids:
        fig.add_trace(go.Scatter(
            x=[centroid[0]], y=[centroid[1]], 
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            showlegend=False
        ))

    fig.update_layout(
        title='K-means Clustering of Stocks by Returns and GARCH Volatility',
        xaxis_title='Annualized Return',
        yaxis_title='Annualized GARCH Volatility',
        template="plotly_dark"
    )

    fig.show()

if __name__ == "__main__":
    asyncio.run(main())