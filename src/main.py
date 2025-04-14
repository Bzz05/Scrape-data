import argparse
import logging
import sys
import time
import os
import torch
import yaml
import numpy as np
import pandas as pd
import asyncio
import threading
import queue
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer as RllibPPOTrainer
from ray.rllib.models import ModelCatalog
from models.rllib_policy import RllibTransformerPolicy
from environment.trading_env import TradingEnv
from data.processor import DataProcessor
from data.scraper import BinanceDataScraper
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

log_queue = queue.Queue()
log_lines = []
app = FastAPI()

@app.get("/")
def read_root():
    logging.info("Launching FastAPI log server...")
    return {"message": "Training bot server running."}

@app.get("/logs")
def get_logs():
    # Return logs as a JSON response
    return JSONResponse(content={"logs": log_lines})

def launch_log_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

def parse_arguments():
    parser = argparse.ArgumentParser(description="RL Training for Binance Trading Bot")
    parser.add_argument("--mode", type=str, choices=["train", "test", "api"], required=True, help="Mode of execution")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="checkpoint.pt", help="Path to model checkpoint (for resuming)")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

def load_configuration(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded.")
    return config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging is set up.")


async def scrape_data(config_path):
    scraper = BinanceDataScraper(config_path)
    symbol = "BTCUSDT"
    intervals = ["1d", "1h"]
    start_time = pd.to_datetime("2019-09-01")
    end_time = pd.to_datetime("2025-03-24")

    for interval in intervals:
        klines = await scraper.fetch_historical_klines(symbol, interval, start_time, end_time)
        output_path = os.path.join("data/historical", f"klines_{interval}.csv")
        klines.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logging.info(f"Data saved to {output_path}")
    
    await scraper.close()


def process_data(config_path):
    processor = DataProcessor(config_path)
    raw_data_files = ["klines_1d.csv", "klines_1h.csv"]
    
    for raw_data_file in raw_data_files:
        raw_df = processor.load_raw_data(raw_data_file)
        # process data.
        processed_df = processor.process_data(raw_df)
        
        processed_filename = raw_data_file.split("_")[-1]
        processor.save_processed_data(processed_df, processed_filename)


def train_model(config_path, resume_checkpoint=None):
    """
    for training
    """
    config = load_configuration(config_path)
    ModelCatalog.register_custom_model("rllib_transformer", RllibTransformerPolicy)
    
    # Get the processed data file from config.
    data_file = config.get("data_file", None)
    
    # Base RLlib configuration.
    rllib_config = config.get("rllib", {})

    # Check for GPU availability.
    if torch.cuda.is_available():
        rllib_config["num_gpus"] = 1
        logging.info("GPU detected. Using 1 GPU.")
    else:
        rllib_config["num_gpus"] = 0
        logging.info("No GPU detected. Training on CPU.")

    # Merge additional settings.
    rllib_config.update({
        "env": "TradingEnv",
        "framework": "torch",
        "model": {
            "custom_model": "rllib_transformer", 
            "custom_model_config": config.get("model", {})
        },
        "horizon": config.get("horizon", 500),
    })
    
    # Register the custom environment.
    tune.register_env("TradingEnv", lambda cfg: TradingEnv(config_path, data_file=data_file))
    
    # Create the PPO trainer.
    agent = RllibPPOTrainer(config=rllib_config)
    
    # Optionally resume from a checkpoint.
    if resume_checkpoint:
        agent.restore(resume_checkpoint)
        logging.info(f"Resumed training from checkpoint: {resume_checkpoint}")
    
    total_timesteps = config.get("total_timesteps", 50000)
    timesteps = 0

    progress_log = []
    progress_dir = os.path.join("data", "progress")
    os.makedirs(progress_dir, exist_ok=True)
    progress_filepath = os.path.join(progress_dir, "progress.csv")
    
    try:
        while timesteps < total_timesteps:
            result = agent.train()
            timesteps_in_iter = result.get("time_this_iter_s") or result.get("timesteps_this_batch")
            if timesteps_in_iter is None:
                raise KeyError("No timesteps key found in training result. Keys: " + str(result.keys()))
            timesteps += timesteps_in_iter
            episode_rewards = result.get("hist_stats", {}).get("episode_reward", [])
            median_reward = np.median(episode_rewards) if episode_rewards else 0
            max_reward = max(episode_rewards) if episode_rewards else 0
            min_reward = min(episode_rewards) if episode_rewards else 0
            current_balance = result.get("custom_metrics", {}).get("portfolio_value", None)
            
            progress_log.append({
                 "iteration": result["training_iteration"],
                 "timesteps": timesteps,
                 "mean_reward": result["episode_reward_mean"],
                 "median_reward": median_reward,
                 "max_reward": max_reward,
                 "min_reward": min_reward,
                 "current_balance": current_balance,
                 "timestamp": time.time()
             })
            
            log_msg = (
                f"Iteration: {result['training_iteration']} | "
                f"Timesteps: {timesteps} | "
                f"Mean Reward: {result['episode_reward_mean']:.2f} | "
                f"Median: {median_reward:.2f} | "
                f"Max: {max_reward:.2f} | "
                f"Min: {min_reward:.2f}"
            )
            logging.info(log_msg)
            log_lines.append(log_msg)
            log_queue.put(log_msg)
            
            # Save a checkpoint every 10 iterations.
            if result["training_iteration"] % 10 == 0:
                checkpoint = agent.save()
                logging.info(f"Checkpoint saved at iteration {result['training_iteration']}: {checkpoint}")
            
            # Save progress log after every iteration.
            try:
                pd.DataFrame(progress_log).to_csv(progress_filepath, index=False)
                # logging.info(f"Progress log saved to {progress_filepath}")
            except Exception as csv_e:
                logging.error(f"Error saving progress log: {csv_e}")
                
        final_checkpoint = agent.save()
        logging.info(f"Final checkpoint saved at end of training: {final_checkpoint}")
                
    except Exception as e:
        logging.error(f"Training interrupted: {e}. Saving final checkpoint.")
        checkpoint = agent.save()
        logging.info(f"Final checkpoint saved: {checkpoint}")
        try:
            pd.DataFrame(progress_log).to_csv(progress_filepath, index=False)
            logging.info(f"Final progress log saved to {progress_filepath}")
        except Exception as csv_e:
            logging.error(f"Error saving final progress log: {csv_e}")


def main():
    args = parse_arguments()
    setup_logging()

    if args.mode == "train":
        logging.info("Starting log server thread...")
        try:
            t = threading.Thread(target=launch_log_server, daemon=True)
            t.start()
            time.sleep(2)
            logging.info("Log server thread started.")
        except Exception as e:
            logging.error(f"Failed to start log server: {e}")

        logging.info("start data scraping...")
        asyncio.run(scrape_data(args.config))

        logging.info("start data processing...")
        process_data(args.config)

        logging.info("start training...")
        train_model(args.config)

    else:
        logging.error("Invalid mode selected.")

    logging.info("Process complete.")


if __name__ == "__main__":
    main()