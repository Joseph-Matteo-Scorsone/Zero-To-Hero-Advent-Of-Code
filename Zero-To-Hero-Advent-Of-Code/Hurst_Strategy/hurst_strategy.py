import asyncio
import time
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

class BackTester():

    def __init__(self, configs, cache, start, end):
        self.configs = configs
        self.cache = cache
        self.start = start
        self.end = end


    async def get_stock_data(self, tickers, start, end):
        data = {}

        for ticker in tickers:
            cached_data = await self.cache.get_value_from_key(ticker)
            if cached_data is not None:
                data[ticker] = cached_data
                continue

            try:
                df = pd.read_csv(f'../CSVs/{ticker}_returns.csv', index_col=0, parse_dates=True)
                if self.cache:
                    await self.cache.set_key_value(ticker, df)

                data[ticker] = df
            except Exception as e:
                df = yf.download(ticker, start=start, end=end)
                if self.cache:
                    await self.cache.set_key_value(ticker, df)
                df.to_csv(f'../CSVs/{ticker}_returns.csv')
                data[ticker] = df

        return data

    def calculate_hurst(self, x, max_window=None):
        if max_window is None:
            max_window = len(x) // 2
        
        # Ensure we have enough data points
        if len(x) < 2:
            return np.nan

        # init for R/S analysis
        rs = []
        n = np.arange(2, max_window + 1)

        for i in n:
            # Split the series into non-overlapping subseries
            subseries = np.array_split(x, len(x) // i)
            
            # Compute R/S for each subseries
            r_values, s_values = [], []
            for series in subseries:
                if len(series) < 2:
                    continue

                # Mean adjustment
                mean = np.mean(series)
                z = np.cumsum(series - mean)
                
                # Range
                r = np.max(z) - np.min(z)
                
                # Standard deviation
                s = np.std(series)
                
                if s == 0:
                    continue  # Avoid division by zero
                
                r_values.append(r)
                s_values.append(s)

            if not r_values or not s_values:
                continue

            # Average R/S for this window size
            avg_rs = np.mean(np.array(r_values) / np.array(s_values))
            rs.append(avg_rs)

        if len(rs) < 2:  # Need at least two points to fit a line
            return np.nan

        # Log-log plot and fit
        log_n = np.log(n[:len(rs)])
        log_rs = np.log(rs)

        # Line fit for Hurst exponent
        try:
            slope, _ = np.polyfit(log_n, log_rs, 1)
            hurst = slope
        except np.linalg.LinAlgError:  # If polyfit fails due to numerical issues
            return np.nan

        return hurst

    def add_hurst_strategy(self, data, window):
        data = data.copy()

        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        # Calculate Hurst Exponent for each rolling window
        data['Hurst'] = data['log_returns'].rolling(window=window).apply(lambda x: self.calculate_hurst(x))

        data = data.dropna()
        data["Signal"] = np.where(
            (data["Hurst"] > 0.5) & (data["Hurst"].shift(1) <= 0.5), 1,
            np.where(
                (data["Hurst"] < 0.5) & (data["Hurst"].shift(1) >= 0.5), -1,
                0
            )
        )

        data = data.dropna()
        return data


    def simulate_backtest(self, data, initial_balance=10000, risk=0.10, transaction_cost=0.001, take_profit=0.05, stop_loss=0.02):
        balance = float(initial_balance)
        position_size = 0  # Shares held
        position = 0  # 1 for long, -1 for short, 0 for no position
        entry_price = 0

        data = data.copy()
        data["Balance"] = balance

        for i in range(1, len(data)):
            signal = data.iloc[i - 1]["Signal"]

            # Check for take-profit or stop-loss
            if position != 0:
                price_change = (data.iloc[i]["Close"] - entry_price) / entry_price

                if position == 1 and (price_change >= take_profit or price_change <= -stop_loss):
                    # Close long position
                    balance += position_size * (data.iloc[i]["Close"] - entry_price) * (1 - transaction_cost)
                    position_size = 0
                    position = 0

                elif position == -1 and (-price_change >= take_profit or -price_change <= -stop_loss):
                    # Close short position
                    balance += position_size * (entry_price - data.iloc[i]["Close"]) * (1 - transaction_cost)
                    position_size = 0
                    position = 0

            # Open new positions based on signals
            if signal == 1 and position <= 0:  # Buy signal
                if position == -1:  # Close short position
                    balance += position_size * (entry_price - data.iloc[i]["Close"]) * (1 - transaction_cost)
                    position_size = 0

                # Open long position
                position = 1
                position_size = (risk * balance) / data.iloc[i]["Close"]
                entry_price = data.iloc[i]["Close"]

            elif signal == -1 and position >= 0:  # Sell signal
                if position == 1:  # Close long position
                    balance += position_size * (data.iloc[i]["Close"] - entry_price) * (1 - transaction_cost)
                    position_size = 0

                # Open short position
                position = -1
                position_size = (risk * balance) / data.iloc[i]["Close"]
                entry_price = data.iloc[i]["Close"]

            # Update balance for unrealized P&L
            if position == 1:  # Long
                balance += position_size * (data.iloc[i]["Close"] - data.iloc[i - 1]["Close"])
            elif position == -1:  # Short
                balance += position_size * (data.iloc[i - 1]["Close"] - data.iloc[i]["Close"])

            data.loc[data.index[i], "Balance"] = balance

        return data

    def plot_balance(self, data, plot_title):
        # Create subplots: use 'domain' type for x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, 
                            subplot_titles=("Balance Over Time", "Hurst Exponent"))

        # Add traces
        fig.add_trace(go.Scatter(x=data.index, y=data["Balance"], mode='lines', name='Balance'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["Hurst"], mode='lines', name='Hurst Exponent'), row=2, col=1)

        # Update layout
        fig.update_layout(
            title_text=f"Backtest Results for {plot_title}",
            template="plotly_dark"
        )
        fig.update_yaxes(title_text="Balance", row=1, col=1)
        fig.update_yaxes(title_text="Hurst Exponent", row=2, col=1)

        fig.show()

    async def run_backtest(self, ticker, window, risk, take_profit, stop_loss):
        # function to use as a task to backtest specific data
        backtest_id = f"{ticker}_{window}_{risk}"
        stock_data = await self.get_stock_data([ticker], self.start, self.end)
        data = stock_data[ticker]
        data = self.add_hurst_strategy(data, window)
        results = self.simulate_backtest(data, risk=risk, transaction_cost=0.001, take_profit=take_profit, stop_loss=stop_loss)
        
        results.to_csv(f"../backtests/backtest_{backtest_id}.csv")
        plot_title = f"{ticker} Window {window} Risk {risk} Take Profit {take_profit} Stop loss {stop_loss}"
        self.plot_balance(results, plot_title)
        return results
    
    async def run_all_backtests(self):
        tasks = [
            self.run_backtest(
                config["ticker"], config["window"], config["risk"], 
                config["take_profit"], config["stop_loss"]
            ) for config in self.configs
        ]
        await asyncio.gather(*tasks)

async def main():
    cache = Async_Cache()

    start_date = '2023-01-01'
    end_date = '2024-01-01'
    tickers = ["NVDA", "MSFT", "SQ"]


    # Define varying parameters
    backtest_configs = [
        {"ticker": ticker, "window": window, "risk": risk, "take_profit":take_profit, "stop_loss":stop_loss}
        for ticker in tickers
        for window in [10, 20, 40]
        for risk in [0.10]
        for take_profit in [0.20]
        for stop_loss in [0.10]
    ]

    BT = BackTester(backtest_configs, cache, start_date, end_date)

    await BT.run_all_backtests()

asyncio.run(main())