import asyncio
import time
import yfinance as yf
import pandas as pd
import numpy as np
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

async def get_stock_data(tickers, start, end, cache=None):
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

def add_moving_average_strategy(data, short_window=20, long_window=50):
    # Simple moving avaerage cross strategy
    data = data.copy()
    data["Short_MA"] = data["Close"].rolling(window=int(short_window)).mean()
    data["Long_MA"] = data["Close"].rolling(window=int(long_window)).mean()

    data = data.dropna()
    data["Signal"] = np.where(
        (data["Short_MA"] > data["Long_MA"]) & (data["Short_MA"].shift(1) <= data["Long_MA"].shift(1)), 1,
        np.where(
            (data["Short_MA"] < data["Long_MA"]) & (data["Short_MA"].shift(1) >= data["Long_MA"].shift(1)), -1,
            0
        )
    )
    data = data.dropna()

    return data

def simulate_backtest(data, initial_balance=10000, risk=0.10, transaction_cost=0.001, take_profit=0.05, stop_loss=0.02):
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



def plot_balance(data, plot_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Balance"], mode='lines', name='Balance'))
    fig.update_layout(
        title=f"Balance Over Time for Backtest {plot_title}",
        xaxis_title="Date",
        yaxis_title="Balance",
        template="plotly_dark"
    )
    fig.show()

async def run_backtest(ticker, start, end, short_window, long_window, risk, take_profit, stop_loss, cache):
    # function to use as a task to backtest specific data
    backtest_id = f"{ticker}_{short_window}_{long_window}_{risk}"
    stock_data = await get_stock_data([ticker], start, end, cache)
    data = stock_data[ticker]
    data = add_moving_average_strategy(data, short_window, long_window)
    results = simulate_backtest(data, risk=risk, transaction_cost=0.001, take_profit=take_profit, stop_loss=stop_loss)
    
    results.to_csv(f"../backtests/backtest_{backtest_id}.csv")
    plot_title = f"{ticker} Short Window {short_window} Long Window {long_window} Risk {risk} Take Profit {take_profit} Stop loss {stop_loss}"
    plot_balance(results, plot_title)
    return results

async def main():
    cache = Async_Cache()

    start_date = '2015-01-01'
    end_date = '2024-01-01'
    tickers = ["CRWD"]

    # Define varying parameters
    backtest_configs = [
        {"ticker": ticker, "short_window": short, "long_window": long, "risk": risk, "take_profit":take_profit, "stop_loss":stop_loss}
        for ticker in tickers
        for short in [10, 20]
        for long in [50, 100]
        for risk in [0.01, 0.02]
        for take_profit in [0.05, 0.10]
        for stop_loss in [0.02, 0.05]
    ]

    # Run backtests in parallel
    tasks = [
        run_backtest(config["ticker"], start_date, end_date, config["short_window"], config["long_window"], config["risk"], config["take_profit"], config["stop_loss"], cache)
        for config in backtest_configs
    ]

    # each task is a coroutine running a back test
    await asyncio.gather(*tasks)

asyncio.run(main())
