import yfinance as yf
import pandas as pd
import numpy as np
import random
from scipy.stats import norm

class Contract:
    def __init__(self, strike, premium, dte, delta, gamma, theta, vega, rho, implied_volatility, intrinsic_value, market_price):
        self.strike = strike
        self.premium = premium
        self.dte = dte
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho
        self.implied_volatility = implied_volatility
        self.intrinsic_value = intrinsic_value
        self.market_price = market_price

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if option_type == "call" else  K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_implied_volatility(market_price, S0, K, r, T, is_call_option):
    sigma, tolerance, max_iterations = 0.2, 1e-5, 100
    for _ in range(max_iterations):
        price = black_scholes(S0, K, T, r, sigma, is_call_option)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S0 * np.sqrt(T) * norm.pdf(d1)
        if abs(market_price - price) < tolerance:
            break
        sigma += (market_price - price) / vega
    return sigma

def bs_option_chain(S0, K, r, sigma, T, option_type="call"):
    T /= 365
    
    chain = []
    for i in range(-5, 5):
        strike = K + i
        d1 = (np.log(S0 / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        premium = black_scholes(S0, strike, T, r, sigma, option_type)

        intrinsic_value = max(S0 - strike, 0) if option_type == "call" else max(strike - S0, 0)
        delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        theta = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - (r * strike * np.exp(-r * T) * norm.cdf(d2))
        vega = S0 * norm.pdf(d1) * np.sqrt(T)
        rho = strike * T * np.exp(-r * T) * norm.cdf(d2)

        market_noise = random.gauss(0, 0.5)
        implied_volatility = find_implied_volatility(premium + market_noise, S0, strike, r, T, option_type)

        chain.append(Contract(strike, premium, int(T * 365), delta, gamma, theta, vega, rho, implied_volatility, intrinsic_value, premium + market_noise))
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
    tickers = ['AAPL']
    start_date, end_date = '2023-01-01', '2023-12-31'
    stock_data = get_stock_data(tickers, start=start_date, end=end_date)
    
    for ticker, data in stock_data.items():
        S0, K, r, sigma, T = np.floor(data.iloc[-1]["Close"]), np.floor(data.iloc[-1]["Close"]), 0.05, 0.2, 30

        call_chain = bs_option_chain(S0, K, r, sigma, T, "call")
        put_chain = bs_option_chain(S0, K, r, sigma, T, "put")
        
        for con in call_chain + put_chain:
            print(f"Strike: {con.strike}, Option Price: {con.premium}, Market Price: {con.market_price}, dte: {con.dte}, "
                  f"delta: {con.delta}, gamma: {con.gamma}, theta: {con.theta}, vega: {con.vega}, rho: {con.rho}, "
                  f"implied volatility: {con.implied_volatility}, intrinsic value: {con.intrinsic_value}")
            print("\n")
