'''
run in terminal use this command to start scrape:
python3 scr_binance.py --coins BTC,ETH,XRP,SOL,ICP --resolutions 1h,1d --start_time 2023-01-01T00:00:00 
--end_time 2024-12-31T23:59:59 --endpoint_file_paths endpoints.json --save_folder data/scraped --mode historical
'''

import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
import os
from binance.client import Client
import numpy as np
from pathlib import Path
import warnings
from apscheduler.schedulers.blocking import BlockingScheduler
import argparse
import json
import logging
import keys
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class BinanceScraper:
    CONFIG = {
        "INTERVALS": {"1h": 60, "1d": 60 * 24},
        "LOOKBACK": 168,  # hours lookback for live data
        "TA_LOOKBACK": 48,  # hours lookback for technical analysis
    }

    def __init__(self, coins: List[str], resolutions: List[str],
                 start_time: Optional[datetime], end_time: Optional[datetime],
                 save_folder: str, endpoint_file_paths: Dict[str, Dict[str, str]],
                 mode: str):
        self.coins = coins
        self.resolutions = resolutions
        self.start_time = start_time
        self.end_time = end_time
        self.save_folder = Path(save_folder)
        self.endpoint_file_paths = endpoint_file_paths
        self.endpoints_df = self.load_endpoints()
        self.client = Client()
        self.mode = mode

    def load_endpoints(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        '''
        load TA metrics from csv files
        '''
        endpoints = {}
        for coin in self.coins:
            endpoints[coin] = {}
            for resolution in self.resolutions:
                try:
                    df = pd.read_csv(self.endpoint_file_paths[coin][resolution], index_col=0)
                    endpoints[coin][resolution] = df
                except Exception as e:
                    logging.error(f"Failed to load endpoints for {coin} {resolution}: {e}")
        return endpoints

    def save_data_to_csv(self, df: pd.DataFrame, coin: str, resolution: str, endpoint: str = "binance"):
        '''
        save df to csv
        '''
        folder_path = self.save_folder / coin / resolution
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f"{endpoint}_{coin}_{resolution}_{self.file_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        file_path = folder_path / filename
        try:
            df.to_csv(file_path)
            logging.info(f"Data saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save data to {file_path}: {e}")

    def fetch_klines(self, coin: str, resolution: str) -> pd.DataFrame:
        '''
        fetch klines data from Binance
        '''
        try:
            # For historical mode, add one interval so that the final candle is included.
            if self.mode == "historical":
                adjusted_end = self.end_time + timedelta(minutes=self.CONFIG["INTERVALS"][resolution])
            else:
                adjusted_end = self.end_time

            klines = self.client.futures_historical_klines(
                symbol=f"{coin}USDT",
                interval=resolution,
                start_str=self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_str=adjusted_end.strftime('%Y-%m-%d %H:%M:%S'),
            )
            return pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
            ])
        except Exception as e:
            logging.error(f"Failed to fetch klines for {coin} {resolution}: {e}")
            return pd.DataFrame()

    def process_klines(self, klines: pd.DataFrame, coin: str, resolution: str) -> pd.DataFrame:
        '''
        process klines data and perform TA, then fill missing values.
        '''
        
        # convert columns to float64
        klines = klines[['open_time', 'open', 'high', 'low', 'close']].astype(np.float64)
        if self.mode == "live":
            klines = klines.iloc[:-1]  # Ignore the last incomplete candle

        # convert timestamp and set index
        klines['open_time'] = pd.to_datetime(klines['open_time'], unit='ms')
        klines = klines.set_index("open_time")
        klines.index.name = None

        # apply TA indicators
        metrics = sorted(self.endpoints_df[coin][resolution]["0"].tolist())
        strategy = ta.Strategy(name="MyStrategy", ta=[{"kind": item} for item in metrics])
        klines.ta.strategy(strategy)

        # rename columns to include resolution and coin
        klines.columns = [f"{col}_{self.CONFIG['INTERVALS'][resolution]:04}_{coin}" for col in klines.columns]

        # create a placeholder index covering the entire time period
        # use a frequency matching the resolution ('1h' becomes '1H', '1d' becomes '1D')
        freq = resolution.upper()
        start_str = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_str = self.end_time.strftime('%Y-%m-%d %H:%M:%S')
        

        end_adjusted = pd.to_datetime(end_str) - pd.Timedelta("1ns")
        placeholder = pd.DataFrame(index=pd.date_range(start=start_str, end=end_adjusted, freq=freq))
        placeholder.index = placeholder.index.tz_localize(None)

        # combine the placeholder with the computed klines and fill missing values
        combined = placeholder.combine_first(klines)
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        return combined

    def scrape(self):
        '''
        scraping function
        '''
        current_time = datetime.now(timezone.utc)
        self.file_time = current_time.replace(second=0, microsecond=0, minute=(current_time.minute // 5) * 5)

        for coin in self.coins:
            for resolution in self.resolutions:
                if self.mode == "live":
                    self.end_time = current_time.replace(minute=0, second=0, microsecond=0)
                    self.start_time = self.end_time - timedelta(
                        hours=self.CONFIG["LOOKBACK"],
                        minutes=self.CONFIG["TA_LOOKBACK"] * self.CONFIG["INTERVALS"][resolution]
                    )

                logging.info(f"Fetching data for {coin} {resolution} from {self.start_time} to {self.end_time}")

                # Fetch and process klines
                klines = self.fetch_klines(coin, resolution)
                if klines.empty:
                    continue

                processed_data = self.process_klines(klines, coin, resolution)

                # Save data
                if self.mode == "live":
                    # In live mode, we only save the most recent LOOKBACK period
                    processed_data.iloc[-self.CONFIG["LOOKBACK"]:].to_csv(
                        self.save_folder / f"{coin}_{resolution}.csv"
                    )
                elif self.mode == "historical":
                    processed_data.to_csv(
                        self.save_folder / f"{coin}_{resolution}.csv"
                    )
                    logging.info(f"Scraped historical data for {coin} {resolution} up to {self.end_time}")

    def run_periodic_scrape(self):
        '''
        periodic scraping for live mode.
        '''
        scheduler = BlockingScheduler()
        scheduler.add_job(self.scrape, 'cron', hour='*/1')
        scheduler.start()


def main(args):
    save_folder_path = Path(args.save_folder)
    save_folder_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.endpoint_file_paths) as f:
            endpoint_file_paths = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load endpoint file paths: {e}")
        return

    if args.mode == "live":
        start_time = end_time = None
    elif args.mode == "historical":
        # Attach UTC timezone to the parsed datetimes if needed.
        start_time = datetime.fromisoformat(args.start_time).replace(tzinfo=timezone.utc)
        end_time = datetime.fromisoformat(args.end_time).replace(tzinfo=timezone.utc)

    scraper = BinanceScraper(
        coins=args.coins.split(','),
        resolutions=args.resolutions.split(','),
        start_time=start_time,
        end_time=end_time,
        endpoint_file_paths=endpoint_file_paths,
        save_folder=save_folder_path,
        mode=args.mode,
    )

    warnings.filterwarnings("ignore")

    if args.mode == "live":
        scraper.run_periodic_scrape()
    elif args.mode == "historical":
        scraper.scrape()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance data scraper")
    parser.add_argument("--coins", required=True, help='e.g., "BTC,ETH,FTM,BAT"')
    parser.add_argument("--resolutions", required=True, help='e.g., "1h,1d"')
    parser.add_argument("--start_time", required=False, help='e.g., "2020-07-01T00:00:00"')
    parser.add_argument("--end_time", required=False, help='e.g., "2024-12-31T23:59:59"')
    parser.add_argument("--endpoint_file_paths", required=True, help="endpoints_path_binance.json")
    parser.add_argument("--save_folder", required=True, help='File path to save scrape result')
    parser.add_argument("--mode", required=True, help="Scrape mode", choices=["historical", "live"])

    args = parser.parse_args()
    main(args)

'''
1d csv size is to be expexted to be 24x smaller than 1h csv size
'''