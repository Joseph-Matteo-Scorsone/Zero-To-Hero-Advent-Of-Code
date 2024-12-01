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

def add_moving_average_strategy(data, short_window=20, long_window=50):
    data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
    data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

    data = data.dropna()
    data["Signal"] = np.where(
        (data["Short_MA"] > data["Long_MA"]) & (data["Short_MA"].shift(1) <= data["Long_MA"].shift(1)), 1,
        np.where(
            (data["Short_MA"] < data["Long_MA"]) & (data["Short_MA"].shift(1) >= data["Long_MA"].shift(1)), -1,
            0
        )
    )

    return data

def calculate_indicators(data):

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["Upper_Band"] = data["MA_20"] + 2 * data["Close"].rolling(window=20).std()
    data["Lower_Band"] = data["MA_20"] - 2 * data["Close"].rolling(window=20).std()

    data = data.dropna()
    return data

def plot_with_strategy(data, ticker):
   
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3],
        subplot_titles=("Candlestick Chart", "RSI")
    )
   
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price'
    ), row=1, col=1)
   
    # Moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data['Short_MA'], mode='lines', name='Short MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Long_MA'], mode='lines', name='Long MA'), row=1, col=1)
   
    # Strategy
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]

    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Close'],
        mode='markers', marker=dict(color='green', size=10), name='Buy'
    ), row=1, col=1)


    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['Close'],
        mode='markers', marker=dict(color='red', size=10), name='Sell'
    ), row=1, col=1)
   
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=[70] * len(data), mode='lines',
        line=dict(dash='dash', color='red'), name='Overbought'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=[30] * len(data), mode='lines',
        line=dict(dash='dash', color='green'), name='Oversold'
    ), row=2, col=1)
   
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2_title="Date",
        yaxis1_title="Price",
        yaxis2_title="RSI",
        height=800,
        showlegend=True
    )
   
    fig.show()


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
   
    stock_data = get_stock_data(tickers, start=start_date, end=end_date)
   
    for ticker, data in stock_data.items():
        data = add_moving_average_strategy(data)
        data = calculate_indicators(data)
        plot_with_strategy(data, ticker)