import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_stock_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        try:
            df = pd.read_csv(f'../CSVs/{ticker}_returns.csv', index_col=0, parse_dates=True)
            data[ticker] = df
        except Exception as e:
            print(f"Downloading data for {ticker} due to error: {e}")
            df = yf.download(ticker, start=start, end=end)
            data[ticker] = df
            df.to_csv(f'../CSVs/{ticker}_returns.csv')
    return data

def calculate_volatility(data):
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))

    data['Rolling_Std'] = data['log_returns'].rolling(window=20).std()
    data['Annualized_Vol'] = data['Rolling_Std'] * np.sqrt(252)
    return data

def binomial_option_price(S, K, T, r, sigma, n, option_type="call"):

    dt = T / n  # Length of time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral up probability

    # Initialize the option values at maturity (final step)
    option_prices = np.zeros(n + 1)
    for i in range(n + 1):
        if option_type == "call":
            option_prices[i] = max(S * u**i * d**(n - i) - K, 0)  # Call payoff
        elif option_type == "put":
            option_prices[i] = max(K - S * u**i * d**(n - i), 0)  # Put payoff

    # Step back through the tree to calculate the option price
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            option_prices[i] = np.exp(-r * dt) * (p * option_prices[i + 1] + (1 - p) * option_prices[i])

    return option_prices[0]

def calculate_option_price_binomial(data, risk_free_rate=0.03, days_to_expiration=30, n=100):
    T = days_to_expiration / 252
    
    data = data.dropna().copy()
    
    # Calculating Call option prices using the binomial model
    data.loc[:, 'Call_Price'] = data.apply(
        lambda row: binomial_option_price(
            S=row['Close'],
            K=row['Close'] + 1,
            T=T,
            r=risk_free_rate,
            sigma=row['Annualized_Vol'],
            n=n,
            option_type="call"
        ),
        axis=1
    )
    
    # Calculating Put option prices using the binomial model
    data.loc[:, 'Put_Price'] = data.apply(
        lambda row: binomial_option_price(
            S=row['Close'],
            K=row['Close'] - 1,
            T=T,
            r=risk_free_rate,
            sigma=row['Annualized_Vol'],
            n=n,
            option_type="put"
        ),
        axis=1
    )
    
    return data

def plot_with_options(data, ticker):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Candlestick Chart", "Calls", "Puts")
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price'
    ), row=1, col=1)
    
    # Calls
    fig.add_trace(go.Scatter(x=data.index, y=data['Call_Price'], mode='lines', name='Call_Price', line=dict(color='green')), row=2, col=1)
    
    # Puts
    fig.add_trace(go.Scatter(x=data.index, y=data['Put_Price'], mode='lines', name='Put_Price', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis3_title="Date",
        yaxis1_title="Price",
        yaxis2_title="Call_Price",
        yaxis3_title="Put_Price",
        height=900,
        showlegend=True
    )
    
    fig.show()

if __name__ == '__main__':
    tickers = ['AAPL']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    stock_data = get_stock_data(tickers, start=start_date, end=end_date)

    for ticker, data in stock_data.items():
        data = calculate_volatility(data)
        data = calculate_option_price_binomial(data, risk_free_rate=0.03, days_to_expiration=5)
        plot_with_options(data, ticker)