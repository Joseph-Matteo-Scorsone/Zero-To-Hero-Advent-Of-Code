import yfinance as yf
import pandas as pd
import numpy as np
import random
from scipy.stats import norm

class Contract:
    def __init__(self):
        self.premium = 0
        self.strike = 0
        self.dte = 0
        self.delta = 0
        self.gamma = 0
        self.theta = 0
        self.vega = 0
        self.rho = 0
        self.implied_volatility = 0
        self.intrinsic_value = 0
        self.market_price = 0


def black_scholes(S0, K, r, sigma, T, is_call_option):
    T = T /252

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call_option:
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def find_implied_volatility(market_price, S0, K, r, T, is_call_option):
    sigma = 0.2
    tolerance = 1e-5
    max_iterations = 100
    for _ in range(max_iterations):
        price = black_scholes(S0, K, r, sigma, T, is_call_option)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S0 * np.sqrt(T) * norm.pdf(d1)
        price_difference = market_price - price
        if abs(price_difference) < tolerance:
            break
        sigma += price_difference / vega
    return sigma

def bs_option_chain(S0, K, r, sigma, T, is_call_option):

    chain = []
    for i in range(-5, 5):
        con = Contract()
        days_till_expiry = int(T * 365.2425)
        con.dte = days_till_expiry
        con.strike = K + i
        d1 = (np.log(S0 / (K + i)) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call_option:
            con.premium = S0 * norm.cdf(d1) - (K + i) * np.exp(-r * T) * norm.cdf(d2)
            con.delta = norm.cdf(d1)
            con.intrinsic_value = max(S0 - (K + i), 0)
        else:
            con.premium = (K + i) * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            con.delta = norm.cdf(d1) - 1
            con.intrinsic_value = max((K + i) - S0, 0)

        con.gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        con.theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - (r * (K + i) * np.exp(-r * T) * norm.cdf(d2))
        con.vega = S0 * norm.pdf(d1) * np.sqrt(T)
        con.rho = (K + i) * T * np.exp(-r * T) * norm.cdf(d2)

        market_noise = random.gauss(0, 0.5)
        con.market_price = con.premium + market_noise

        con.implied_volatility = find_implied_volatility(con.market_price, S0, K + i, r, T, is_call_option)

        chain.append(con)
    return chain

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


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    stock_data = get_stock_data(tickers, start=start_date, end=end_date)

    for ticker, data in stock_data.items():
        
        S0 = data.iloc[-1]["Close"]
        K = S0
        r = 0.05
        sigma = 0.2
        T = 30

        call_chain = bs_option_chain(S0, K, r, sigma, T, True)
        put_chain = bs_option_chain(S0, K, r, sigma, T, False)

        for con in call_chain:
            print(f"Strike: {con.strike}, European Call Option Price: {con.premium}, Market Price: {con.market_price}, "
                f"dte: {con.dte}, delta: {con.delta}, gamma: {con.gamma}, theta: {con.theta}, "
                f"vega: {con.vega}, rho: {con.rho}, implied volatility: {con.implied_volatility}, "
                f"intrinsic value: {con.intrinsic_value}")

        print()

        for con in put_chain:
            print(f"Strike: {con.strike}, European Put Option Price: {con.premium}, Market Price: {con.market_price}, "
                f"dte: {con.dte}, delta: {con.delta}, gamma: {con.gamma}, theta: {con.theta}, "
                f"vega: {con.vega}, rho: {con.rho}, implied volatility: {con.implied_volatility}, "
                f"intrinsic value: {con.intrinsic_value}")
