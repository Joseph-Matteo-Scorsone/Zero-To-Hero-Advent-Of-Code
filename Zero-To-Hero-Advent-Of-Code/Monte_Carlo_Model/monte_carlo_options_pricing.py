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

# Calculate volatility
def calculate_volatility(data):
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Rolling_Std'] = data['log_returns'].rolling(window=20).std()
    data['EWMA_Std'] = data['log_returns'].ewm(span=20).std()
    data['Annualized_Vol'] = data['Rolling_Std'] * np.sqrt(252)
    return data

# Monte Carlo simulation for option pricing
def monte_carlo_option_price(S, K, T, r, sigma, option_type="call", num_simulations=10):
    dt = T / 252  # daily steps assuming 252 trading days per year
    random_paths = np.random.normal(loc=0, scale=1, size=(num_simulations, 252))
    S_paths = np.zeros((num_simulations, 252))
    S_paths[:, 0] = S
    
    for t in range(1, 252):
        S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_paths[:, t - 1])
    
    # Option payoff calculation
    if option_type == "call":
        payoffs = np.maximum(S_paths[:, -1] - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_paths[:, -1], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Discount to present value
    discounted_payoffs = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoffs)

def calculate_option_price_monte_carlo(data, risk_free_rate=0.03, days_to_expiration=30, num_simulations=10000):
    T = days_to_expiration / 252  # Convert days to years
    valid_data = data.dropna().copy()  # Use .copy() to ensure a copy is made, not a view.
    
    # Calculating Call option prices using Monte Carlo simulation
    valid_data.loc[:, 'Call_Price_MC'] = valid_data.apply(
        lambda row: monte_carlo_option_price(
            S=row['Close'],
            K=row['Close'] + 1,
            T=T,
            r=risk_free_rate,
            sigma=row['Annualized_Vol'],
            option_type="call",
            num_simulations=num_simulations
        ),
        axis=1
    )
    
    # Calculating Put option prices using Monte Carlo simulation
    valid_data.loc[:, 'Put_Price_MC'] = valid_data.apply(
        lambda row: monte_carlo_option_price(
            S=row['Close'],
            K=row['Close'] - 1,
            T=T,
            r=risk_free_rate,
            sigma=row['Annualized_Vol'],
            option_type="put",
            num_simulations=num_simulations
        ),
        axis=1
    )
    
    return valid_data

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
    fig.add_trace(go.Scatter(x=data.index, y=data['Call_Price_MC'], mode='lines', name='Call_Price_MC', line=dict(color='green')), row=2, col=1)
    
    # Puts
    fig.add_trace(go.Scatter(x=data.index, y=data['Put_Price_MC'], mode='lines', name='Put_Price_MC', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis3_title="Date",
        yaxis1_title="Price",
        yaxis2_title="Call_Price_MC",
        yaxis3_title="Put_Price_MC",
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
        data = calculate_option_price_monte_carlo(data, risk_free_rate=0.03, days_to_expiration=5)
        plot_with_options(data, ticker)
