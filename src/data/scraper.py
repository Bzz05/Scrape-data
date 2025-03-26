import asyncio
import aiohttp
import pandas as pd
import yaml
import logging
import json
from datetime import datetime, timedelta
import os

class BinanceDataScraper:
    def __init__(self, config_path):
        # init scraper
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.api_url = self.config.get("api_url", "https://testnet.binancefuture.com")
        self.session = aiohttp.ClientSession()
        logging.info("BinanceDataScraper initialized.")

    async def fetch_historical_klines(self, symbol, interval, start_time, end_time=None):
        # fetch historical from Binance
        all_data = []
        current_time = start_time
        while current_time < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(current_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
            logging.info(f"Fetching klines for {symbol} from {current_time} to {end_time}")
            try:
                async with self.session.get(f"{self.api_url}/fapi/v1/klines", params=params) as response:
                    data = await response.json()
                if not data:
                    break
                all_data.extend(data)
                # Set current_time to just after the last kline timestamp
                current_time = datetime.fromtimestamp(data[-1][0] / 1000) + timedelta(milliseconds=1)
            except Exception as e:
                logging.error(f"Error fetching historical klines for {symbol}: {e}")
                raise
        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        logging.info(f"Fetched {len(df)} klines for {symbol}.")
        return df

    def save_data_to_disk(self, data, data_type, symbol, timeframe):
        # save scraped data to disk to csv
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        filename = f"{symbol}_{timeframe}_{timestamp}.csv"
        directory = os.path.join("data", data_type)
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        try:
            if isinstance(data, pd.DataFrame):
                # Convert datetime columns to string format without milliseconds.
                for col in data.select_dtypes(include=['datetime64[ns]']).columns:
                    data[col] = data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                data.to_csv(filepath, index=False)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            logging.info(f"Data saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving data to disk: {e}")
            raise

    async def close(self):
        await self.session.close()

async def main():
    config_path = "config/config.yaml"
    scraper = BinanceDataScraper(config_path)
    
    symbol = "BTCUSDT" # set this urself
    intervals = ["1d", "1h"] # this too
    start_time = datetime(2019, 9, 1)
    end_time = datetime(2025, 3, 24)
    
    for interval in intervals:
        klines = await scraper.fetch_historical_klines(symbol, interval, start_time, end_time)
        
        # save the df
        output_directory = "data/historical"
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f"klines_{interval}.csv")
        
        # remove miliseconds, close time have microseconds and it kinda messes up reading in excel bcz of formatting
        klines.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Data saved to {output_path}")
    
    await scraper.close()

if __name__ == "__main__":
    asyncio.run(main())