import asyncio
import time
import yfinance as yf
import numpy as np
import pandas as pd
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

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = None

    def fit(self, X, depth=0):
        n_samples, n_features = X.shape
        
        # Stop criteria: if max depth reached or only one sample left
        if depth >= self.max_depth or n_samples <= 1:
            self.size = n_samples
            return

        # Choose random feature for splitting
        np.random.seed(2024)
        self.split_feature = np.random.randint(0, n_features)
        feature_values = X[:, self.split_feature]
        min_val, max_val = feature_values.min(), feature_values.max()
        
        # If all values in the feature are the same, make it a leaf
        if min_val == max_val:
            self.size = n_samples
            return
        
        # Randomly choose a split value within the range of the feature
        self.split_value = np.random.uniform(min_val, max_val)

        # Split data based on the chosen split value
        left_mask = feature_values < self.split_value
        right_mask = ~left_mask # ~ bitwise NOT, every 0 becomes a 1, every 1 becomes a 0

        # Recursively build left and right subtrees
        self.left = IsolationTree(self.max_depth)
        self.right = IsolationTree(self.max_depth)
        self.left.fit(X[left_mask], depth + 1)
        self.right.fit(X[right_mask], depth + 1)

    def path_length(self, X):
        # If this node is a leaf, return the path length based on node size
        if self.size is not None:
            return np.full(X.shape[0], np.log2(self.size + 1) if self.size > 1 else 0)

        # Get the value of the split feature for all samples
        feature_values = X[:, self.split_feature]
        left_mask = feature_values < self.split_value
        right_mask = ~left_mask # ~ bitwise NOT, every 0 becomes a 1, every 1 becomes a 0

        path = np.zeros(X.shape[0])
        # Compute path length for each subtree
        if self.left:
            path[left_mask] = self.left.path_length(X[left_mask])
        if self.right:
            path[right_mask] = self.right.path_length(X[right_mask])

        # Add 1 to account for the current node in the path
        return path + 1

class IsolationForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X):
        self.trees = []
        for _ in range(self.n_trees):
            # Randomly sample half of the data for each tree
            sample_indices = np.random.choice(X.shape[0], size=X.shape[0] // 2, replace=False)
            sample = X[sample_indices]
            tree = IsolationTree(self.max_depth)
            tree.fit(sample)
            self.trees.append(tree)

    def anomaly_score(self, X):
        # Compute average path length across all trees for each sample
        avg_path_length = np.mean([tree.path_length(X) for tree in self.trees], axis=0)
        # Normalization constant for anomaly score calculation
        c_n = 2 * (np.log(X.shape[0] - 1) + 0.5772156649) - (2 * (X.shape[0] - 1) / X.shape[0])
        # Calculate anomaly score based on path length
        return 2 ** (-avg_path_length / c_n)

    def predict(self, X, threshold=0.9):
        scores = self.anomaly_score(X)
        return scores > threshold  # Returns boolean array where True indicates anomaly

async def main():
    cache = Async_Cache()
    
    tickers = ["QQQ"]
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    stock_data = await get_stock_data(cache, tickers, start_date, end_date)

    for ticker, df in stock_data.items():
        df = df.copy()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        
        X = df['log_returns'].values.reshape(-1, 1) # 2d array with 1 column
        
        # Fit Isolation Forest for outlier detection
        isolation_forest = IsolationForest(n_trees=100, max_depth=10)
        isolation_forest.fit(X)

        # Predict outliers
        df.loc[:, 'Outlier'] = isolation_forest.predict(X)
        outlier_dates = df[df['Outlier']].index
        
        # Visualize or analyze outliers
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=outlier_dates, y=df.loc[outlier_dates, 'Close'], mode='markers', name='Outliers', marker=dict(color='red', size=10)))
        fig.update_layout(title=f'{ticker} Price with Outliers', 
                          xaxis_title='Date', 
                          yaxis_title='Price',
                          template="plotly_dark")
        fig.show()

if __name__ == "__main__":
    asyncio.run(main())