import argparse
import logging
import sys
import time
import os
import torch
import yaml
import pandas as pd
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer as RllibPPOTrainer
from ray.rllib.models import ModelCatalog
from models.rllib_policy import RllibTransformerPolicy
from environment.trading_env import TradingEnv

def parse_arguments():
    parser = argparse.ArgumentParser(description="RL Training for Binance Trading Bot")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="checkpoint.pt", help="Path to model checkpoint (for resuming)")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging is set up.")

def load_configuration(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded.")
    return config

def train_model(config_path, resume_checkpoint=None):
    '''
    this 4 training
    '''
    config = load_configuration(config_path)
    ModelCatalog.register_custom_model("rllib_transformer", RllibTransformerPolicy)
    
    # get the processed data file from config.
    data_file = config.get("data_file", None)
    
    # base RLlib configuration.
    rllib_config = config.get("rllib", {})

    # check for GPU
    if torch.cuda.is_available():
        rllib_config["num_gpus"] = 1
        logging.info("GPU detected. Using 1 GPU.")
    else:
        rllib_config["num_gpus"] = 0
        logging.info("No GPU detected. Training on CPU.")

    # merge additional settings.
    rllib_config.update({
        "env": "TradingEnv",
        "framework": "torch",
        "model": {
            "custom_model": "rllib_transformer", 
            "custom_model_config": config.get("model", {})
        },
        "horizon": config.get("horizon", 500),
    })
    
    # register the custom environment.
    tune.register_env("TradingEnv", lambda cfg: TradingEnv(config_path, data_file=data_file))
    
    # create the PPO trainer.
    agent = RllibPPOTrainer(config=rllib_config)
    
    # to start from scratch, do NOT restore from a checkpoint.
    if resume_checkpoint:
        agent.restore(resume_checkpoint)
        logging.info(f"Resumed training from checkpoint: {resume_checkpoint}")
    
    total_timesteps = config.get("total_timesteps", 50000)
    timesteps = 0
    last_checkpoint_time = time.time()

    cumulative_reward = 0.0
    progress_log = []
    progress_dir = os.path.join("data", "progress")
    os.makedirs(progress_dir, exist_ok=True)
    progress_filepath = os.path.join(progress_dir, "progress.csv")
    
    try:
        while timesteps < total_timesteps:
            result = agent.train()
            # use available timesteps key.
            timesteps_in_iter = result.get("time_this_iter_s") or result.get("timesteps_this_batch")
            if timesteps_in_iter is None:
                raise KeyError("No timesteps key found in training result. Keys: " + str(result.keys()))
            timesteps += timesteps_in_iter
            cumulative_reward += result["episode_reward_mean"]

            # attempt to extract current balance from custom metrics.
            current_balance = result.get("custom_metrics", {}).get("portfolio_value", None)
            
            # append progress for this iteration.
            progress_log.append({
                "iteration": result["training_iteration"],
                "timesteps": timesteps,
                "episode_reward_mean": result["episode_reward_mean"],
                "cumulative_reward": cumulative_reward,
                "current_balance": current_balance,
                "timestamp": time.time()
            })
            logging.info(
                f"Iteration: {result['training_iteration']}, timesteps: {timesteps:.2f}, "
                f"reward: {result['episode_reward_mean']:.2f}, cumulative reward: {cumulative_reward:.2f}, "
                f"current balance: {current_balance}"
            )
            
            # Save a checkpoint every 10 iterations.
            if result["training_iteration"] % 10 == 0:
                checkpoint = agent.save()
                last_checkpoint_time = time.time()
                logging.info(f"Checkpoint saved at iteration {result['training_iteration']}: {checkpoint}")
            
            # Save progress log after every iteration.
            try:
                pd.DataFrame(progress_log).to_csv(progress_filepath, index=False)
                logging.info(f"Progress log saved to {progress_filepath}")
            except Exception as csv_e:
                logging.error(f"Error saving progress log: {csv_e}")
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
    setup_logging(args.log_level)
    try:
        train_model(args.config)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()