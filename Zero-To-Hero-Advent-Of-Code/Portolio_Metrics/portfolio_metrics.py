import asyncio
import time
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

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

    def __init__(self, cache, start, end):
        self.cache = cache
        self.start = start
        self.end = end
        self.strategies = {
            "GARCH": self.add_GARCH_strategy,
            "Hurst": self.add_hurst_strategy,
        }

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

    def calculate_GARCH(self, returns):
        # negative log likelihood function we need to minimize with the omega, alpha, and beta parameters
        # in turn we maximize it by minimizing the negative
        def garch_likelihood(params):
            omega, alpha, beta = params
            T = len(returns)
            sigma2 = np.zeros(T)
            sigma2[0] = np.var(returns) # initialize with the variance of the ts
            # Iterate over the returns to calculate conditional variances using the GARCH formula
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            # Compute the negative log-likelihood for the GARCH model
            likelihood = 0.5 * np.sum(np.log(2 * np.pi * sigma2) + (returns**2 / sigma2))
            return likelihood

        # initial guesses
        initial_params = [0.1, 0.1, 0.8]
        # bounds for values
        bounds = [(1e-6, None), (0, 1), (0, 1)]
        # use scipy to minimize
        result = minimize(garch_likelihood, initial_params, bounds=bounds)
        return result.x if result.success else initial_params

    def add_GARCH_strategy(self, data, window):
        data = data.copy()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        garch_params = self.calculate_GARCH(data['log_returns'].values)
        omega, alpha, beta = garch_params

        sigma2 = np.zeros(len(data))
        sigma2[0] = np.var(data['log_returns']) # initialize with the variance of the ts
        # Calculate conditional variances iteratively using the GARCH formula
        for t in range(1, len(data)):
            sigma2[t] = omega + alpha * data['log_returns'].iloc[t-1]**2 + beta * sigma2[t-1]

        data['GARCH'] = np.sqrt(sigma2)
        data['volatility_threshold'] = data['GARCH'].rolling(window=window).mean()
        data = data.dropna()

        data['Signal'] = np.where(
            (data['GARCH'] > data['volatility_threshold']) & (data['GARCH'].shift(1) <= data['volatility_threshold']), 1,
            np.where(
                (data['GARCH'] < data['volatility_threshold']) & (data['GARCH'].shift(1) >= data['volatility_threshold']), -1,
                0
            )
        )
        data["Daily_Return"] = data["Close"].pct_change().dropna()

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
        data["Daily_Return"] = data["Close"].pct_change().dropna()

        data = data.dropna()
        return data
    
    def apply_strategy(self, strategy_name, data, **kwargs):
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found.")
        return self.strategies[strategy_name](data, **kwargs)

    def simulate_portfolio(self, data_dict, allocations, initial_balance=10000.0, transaction_cost=0.001, take_profit=0.20, stop_loss=0.10):
        portfolio_balance = initial_balance
        daily_returns = pd.DataFrame(index=data_dict[list(data_dict.keys())[0]].index)
        daily_returns['Portfolio'] = initial_balance

        positions = {ticker: {"position": 0, "entry_price": None, "size": 0} for ticker in data_dict.keys()}

        for i in range(1, len(daily_returns)):
            portfolio_return = 0

            for ticker, df in data_dict.items():
                row = df.iloc[i]
                prev_row = df.iloc[i - 1]
                signal = prev_row["Signal"]
                position_data = positions[ticker]

                # Check existing position for take-profit or stop-loss
                if position_data["position"] != 0:
                    if (
                        (position_data["position"] == 1 and row["Close"] >= (position_data["entry_price"] * (1+take_profit))) or
                        (position_data["position"] == 1 and row["Close"] <= (position_data["entry_price"] * (1-stop_loss))) or
                        (position_data["position"] == -1 and row["Close"] <= (position_data["entry_price"] * (1-take_profit))) or
                        (position_data["position"] == -1 and row["Close"] >= (position_data["entry_price"] * (1+stop_loss)))
                    ):
                        # Close position
                        portfolio_balance += (
                            position_data["size"] * (row["Close"] - position_data["entry_price"]) *
                            (1 if position_data["position"] == 1 else -1) * (1 - transaction_cost)
                        )
                        positions[ticker] = {"position": 0, "entry_price": None, "size": 0}

                # Handle signals for opening positions
                if signal == 1 and position_data["position"] <= 0:  # Buy signal
                    if position_data["position"] == -1:  # Close short position
                        portfolio_balance += (
                            position_data["size"] * (position_data["entry_price"] - row["Close"]) * (1 - transaction_cost)
                        )
                    # Open long position
                    size = (allocations[ticker] * portfolio_balance) / row["Close"]
                    positions[ticker] = {"position": 1, "entry_price": row["Close"], "size": size}

                elif signal == -1 and position_data["position"] >= 0:  # Sell signal
                    if position_data["position"] == 1:  # Close long position
                        portfolio_balance += (
                            position_data["size"] * (row["Close"] - position_data["entry_price"]) * (1 - transaction_cost)
                        )
                    # Open short position
                    size = (allocations[ticker] * portfolio_balance) / row["Close"]
                    positions[ticker] = {"position": -1, "entry_price": row["Close"], "size": size}

                # Calculate unrealized P&L for active positions
                if position_data["position"] == 1:  # Long
                    portfolio_balance += position_data["size"] * (row["Close"] - prev_row["Close"])
                elif position_data["position"] == -1:  # Short
                    portfolio_balance += position_data["size"] * (prev_row["Close"] - row["Close"])

                # Update daily return
                portfolio_return += allocations[ticker] * df.iloc[i]["Daily_Return"]

            # Update portfolio balance and store in daily returns
            portfolio_balance *= (1 + portfolio_return)
            daily_returns.loc[daily_returns.index[i], 'Portfolio'] = portfolio_balance

        return daily_returns

        
    def calculate_portfolio_metrics(self, portfolio_values):
        # Calculate key performance metrics
        returns = portfolio_values.pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = mean_return / std_return * np.sqrt(252) # relative to portfolio
        max_drawdown = (portfolio_values / portfolio_values.cummax() - 1).min()
        
        return {
            "Mean Return": mean_return,
            "Volatility": std_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Total Return": portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
        }

    async def run_multiple_windows(self, allocations, strategy_params, configs):
        tasks = []
        for config in configs:
            strategy_params_with_window = {ticker: {'name': strategy['name'], 'window': config['window'], 
                                                    'take_profit': config['take_profit'], 
                                                    'stop_loss': config['stop_loss']} 
                                            for ticker, strategy in strategy_params.items() if strategy}
            tasks.append(self.run_portfolio_backtest(allocations, strategy_params_with_window))

        results = await asyncio.gather(*tasks)
        
        # Plot each result in a separate figure and display metrics
        for i, ((result, metrics), config) in enumerate(zip(results, configs), 1):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.index, y=result['Portfolio'], mode='lines', name='Portfolio'))
            fig.update_layout(
                title=f'Portfolio Performance (Window: {config["window"]} Take Profit: {config["take_profit"]} Stop Loss: {config["stop_loss"]})<br>' +
                      f'Mean Return: {(metrics["Mean Return"] * 100):.2f}<br>' +
                      f'Volatility: {(metrics["Volatility"] * 100):.2f}<br>' +
                      f'Sharpe Ratio: {metrics["Sharpe Ratio"]:.4f}<br>' +
                      f'Max Drawdown: {(metrics["Max Drawdown"] * 100):.2f}<br>' +
                      f'Total Return: {(metrics["Total Return"] * 100):.2f}',
                template="plotly_dark"
            )
            fig.show()

    async def run_portfolio_backtest(self, allocations, strategy_params):
        stock_data = await self.get_stock_data(list(allocations.keys()), self.start, self.end)
        data_dict = {}
        
        for ticker, strategy in strategy_params.items():
            if strategy:  # If a strategy is specified for this ticker
                data = stock_data[ticker]
                data = self.apply_strategy(strategy['name'], data, window=strategy['window'])
                data_dict[ticker] = data
            else:  # If no strategy, just use the raw data
                data_dict[ticker] = stock_data[ticker]

        results = self.simulate_portfolio(data_dict, allocations, take_profit=strategy['take_profit'], stop_loss=strategy['stop_loss'])
        
        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(results['Portfolio'])
        return results, metrics


async def main():
    cache = Async_Cache()

    start_date = '2023-01-01'
    end_date = '2024-01-01'

    BT = BackTester(cache, start_date, end_date)

    # allocations and strategies
    allocations = {
        "SPY": 0.80,  
        "SQ": 0.10,   
        "MSFT": 0.10  
    }

    strategy_params = {
        "SPY": None,
        "SQ": {"name": "GARCH"},
        "MSFT": {"name": "Hurst"}
    }

    configs = [
        {'window': 10, 'take_profit': 0.10, 'stop_loss': 0.05},
        {'window': 20, 'take_profit': 0.20, 'stop_loss': 0.10},
    ]
    
    await BT.run_multiple_windows(allocations, strategy_params, configs)

asyncio.run(main())
