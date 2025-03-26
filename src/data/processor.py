import pandas as pd
import os
import logging
import yaml
from feature_engineering import FeatureEngineer

class DataProcessor:
    def __init__(self, config_path):
        # init processor
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # Define directories for raw and processed data.
        self.raw_dir = os.path.join("data", "historical")
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.feature_engineer = FeatureEngineer(config_path)
        logging.info("DataProcessor initialized.")

    def load_raw_data(self, filename):
        # load raw csv data
        filepath = os.path.join(self.raw_dir, filename)
        try:
            # parse date time columns
            df = pd.read_csv(filepath, parse_dates=["open_time", "close_time"])
            logging.info(f"Loaded raw data from {filepath}")
            return df
        except Exception as e:
            logging.error(f"Error loading raw data from {filepath}: {e}")
            raise

    def process_data(self, raw_df):
        # process raw data
        try:
            # Ensure datetime columns are properly formatted.
            raw_df["open_time"] = pd.to_datetime(raw_df["open_time"]) 
            raw_df['close_time'] = pd.to_datetime(raw_df['close_time'])          
            # Drop 'ignore' column if present.
            if "ignore" in raw_df.columns:
                raw_df = raw_df.drop("ignore", axis=1)

            # Apply technical indicators.
            df = self.feature_engineer.create_technical_indicators(raw_df)
            # Create temporal features (using open_time).
            df = self.feature_engineer.create_temporal_features(df)
            # Normalize numeric features.
            df = self.feature_engineer.normalize_features(df)
            logging.info("Data processing completed.")
            return df
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            raise

    def save_processed_data(self, df, filename):
        # save data to CSV
        filepath = os.path.join(self.processed_dir, filename)
        try:
            print(f'File size for {filepath}\n===========\n{df.shape}\n===========')
            df.to_csv(filepath, index=False)
            logging.info(f"Processed data saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving processed data to {filepath}: {e}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_path = "config/config.yaml"
    processor = DataProcessor(config_path)
    
    # list raws
    raw_data_files = ["klines_1d.csv", "klines_1h.csv"]
    
    for raw_data_file in raw_data_files:
        # load raw data.
        raw_df = processor.load_raw_data(raw_data_file)
        # process data.
        processed_df = processor.process_data(raw_df)
        
        # filename generate
        processed_filename = raw_data_file.split("_")[-1]
        processor.save_processed_data(processed_df, processed_filename)
