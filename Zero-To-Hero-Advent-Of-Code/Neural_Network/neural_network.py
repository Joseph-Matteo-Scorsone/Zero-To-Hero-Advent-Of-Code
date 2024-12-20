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

class NeuralNetwork:
    def __init__(self, inputSize, hiddenLayerSize, outputSize, numHiddenLayers, learningRate=0.01):
        self.inputSize = inputSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputSize = outputSize
        self.numHiddenLayers = numHiddenLayers
        self.learningRate = learningRate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []

        self.weights.append(np.random.randn(self.inputSize, self.hiddenLayerSize))
        self.biases.append(np.zeros((1, self.hiddenLayerSize)))
        
        for _ in range(numHiddenLayers - 1):
            self.weights.append(np.random.randn(self.hiddenLayerSize, self.hiddenLayerSize))
            self.biases.append(np.zeros((1, self.hiddenLayerSize)))
        
        self.weights.append(np.random.randn(self.hiddenLayerSize, self.outputSize))
        self.biases.append(np.zeros((1, self.outputSize)))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def feedforward(self, X):
        self.activations = [X]  # Store activations layer by layer
        for i in range(self.numHiddenLayers + 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]  # Linear transformation
            a = self.tanh(z)  # Apply activation function
            self.activations.append(a)
        return self.activations[-1]  # Output layer activation

    # Backpropagation to update weights and biases
    def backpropagation(self, X, y, output):
        error = y - output  # Compute output error
        deltas = [error * self.tanh_derivative(output)]  # Output layer delta

        # Backpropagate through hidden layers
        for i in range(self.numHiddenLayers, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.tanh_derivative(self.activations[i])
            deltas.insert(0, delta)

        # Update weights and biases
        for i in range(self.numHiddenLayers + 1):
            self.weights[i] += self.learningRate * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] += self.learningRate * np.sum(deltas[i], axis=0, keepdims=True)

    # Train the network over a number of epochs
    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.feedforward(X)  # Compute predictions
            self.backpropagation(X, y, output)  # Update parameters

    def predict(self, X):
        return self.feedforward(X)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

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

        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        nn = NeuralNetwork(inputSize=X_train.shape[1], hiddenLayerSize=32, outputSize=1, numHiddenLayers=2)
        nn.train(X_train, y_train.reshape(-1, 1), epochs=1000)

        predictions = nn.predict(X_test)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            y=predictions.flatten(),
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2)
        ))

        fig.update_layout(
            title=f"Neural Network Predictions vs Actual for {ticker}",
            xaxis_title="Index",
            yaxis_title="Log Returns",
            legend=dict(x=0, y=1),
            template="plotly_dark"
        )

        fig.show()

if __name__ == "__main__":
    asyncio.run(main())