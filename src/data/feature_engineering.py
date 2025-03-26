import pandas as pd
import numpy as np
import yaml
import logging
from ta import add_all_ta_features

class FeatureEngineer:
    def __init__(self, config_path):
        # load configuration.
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info("FeatureEngineer initialized.")

    def create_technical_indicators(self, df):
        # add and append TA
        try:
            df = add_all_ta_features(df,
                                     open="open",
                                     high="high",
                                     low="low",
                                     close="close",
                                     volume="volume",
                                     fillna=True)
            logging.info("Technical indicators added.")
            return df
        except Exception as e:
            logging.error(f"Error creating technical indicators: {e}")
            raise

    def create_futures_specific_features(self, df, funding_data, open_interest_data):
        # 4 futures-specific features like funding rate momentum & open interest velocity
        try:
            df["funding_rate"] = pd.to_numeric(funding_data["fundingRate"], errors="coerce")
            df["funding_momentum"] = df["funding_rate"].diff()
            oi = pd.to_numeric(open_interest_data["sumOpenInterest"], errors="coerce")
            df["oi_velocity"] = oi.diff()
            df["oi_acceleration"] = oi.diff().diff()
            logging.info("Futures-specific features created.")
            return df
        except Exception as e:
            logging.error(f"Error in futures-specific feature creation: {e}")
            raise

    def create_liquidation_features(self, df, liquidation_data):
        # 4 liquidation features
        try:
            liq_df = pd.DataFrame(liquidation_data)
            liq_df["liq_timestamp"] = pd.to_datetime(liq_df["time"], unit="ms")
            liq_counts = liq_df.groupby(pd.Grouper(key="liq_timestamp", freq="1H")).size().rename("liq_count")
            df = df.merge(liq_counts, left_on="open_time", right_index=True, how="left")
            df["liq_count"] = df["liq_count"].fillna(0)
            logging.info("Liquidation features created.")
            return df
        except Exception as e:
            logging.error(f"Error in liquidation feature creation: {e}")
            raise

    def create_orderbook_features(self, df, orderbook_data):
        # orderbook features
        try:
            df["orderbook_imbalance"] = orderbook_data.get("imbalance", 0)
            logging.info("Orderbook features created.")
            return df
        except Exception as e:
            logging.error(f"Error in orderbook feature creation: {e}")
            raise

    def create_temporal_features(self, df):
        # temporal features
        try:
            df["open_time"] = pd.to_datetime(df["open_time"])
            df["day_of_week"] = df["open_time"].dt.dayofweek
            df["hour_of_day"] = df["open_time"].dt.hour
            logging.info("Temporal features created.")
            return df
        except Exception as e:
            logging.error(f"Error creating temporal features: {e}")
            raise

    def normalize_features(self, df):
        # normalize columns
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            logging.info("Features normalized.")
            return df
        except Exception as e:
            logging.error(f"Error normalizing features: {e}")
            raise
