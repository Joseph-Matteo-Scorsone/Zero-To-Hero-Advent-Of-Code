import asyncio
import time
import yfinance as yf
import pandas as pd

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
        start_time = time.time()
        cached_data = await cache.get_value_from_key(ticker)
        if cached_data is not None:

            print(f"Cache hit for ticker: {ticker}")
            time_taken = time.time() - start_time
            print(f"Time taken for cache hit for {ticker}: {time_taken:.4f} seconds")

            data[ticker] = cached_data
            continue

        try:
            start_time = time.time()
            df = pd.read_csv(f'../CSVs/{ticker}_returns.csv', index_col=0, parse_dates=True)
            if cache:
                await cache.set_key_value(ticker, df)

            time_taken = time.time() - start_time
            print(f"Time taken for {ticker} file read: {time_taken:.4f} seconds")

            data[ticker] = df
            

        except Exception as e:

            start_time = time.time()
            df = yf.download(ticker, start=start, end=end)
            if cache:
                cache.set_key_value(ticker, df)
            df.to_csv(f'../CSVs/{ticker}_returns.csv')

            time_taken = time.time() - start_time
            print(f"Time taken for {ticker} download: {time_taken:.4f} seconds")

            data[ticker] = df
            

    return data

async def main():
    cache = Async_Cache()
    asyncio.create_task(cache.clean_up())

    start_date = '2015-01-01'
    end_date = '2024-1-1'
    tickers = ["AAPL", "MSFT", "AAPL"]

    stock_data = await get_stock_data(tickers, start=start_date, end=end_date, cache=cache)
    print(stock_data)

# delete AAPL csv
asyncio.run(main())