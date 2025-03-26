import numpy as np
import pandas as pd
import yaml
import logging
import gym
from gym import spaces

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config_path, data_file='data/processed/1h.csv', **kwargs):
        # setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

        # load config        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # environment params
        self.obs_dim = self.config.get("obs_dim", 90)
        self._max_episode_steps = self.config.get("horizon", 500)
        
        # trade params
        self.trade_fraction = self.config.get("trade_fraction", 0.1)
        self.transaction_cost = self.config.get("transaction_cost", 0.002)  # Slightly increased
        
        # init portfolio parameters
        self.initial_cash = self.config.get("initial_cash", 10000.0)
        
        # define Gym spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

        # load market data
        if data_file:
            try:
                self.data = pd.read_csv(data_file, parse_dates=["open_time", "close_time"])
                logging.info(f"Loaded market data from {data_file}")
            except Exception as e:
                logging.warning(f"Error loading data: {e}. Generating synthetic data.")
                self.data = self._load_market_data()
        else:
            self.data = self._load_market_data()
        
        # performance tracking
        self.initial_portfolio_value = None
        self.max_portfolio_value = None
        
        self.reset()
        logging.info("TradingEnv initialized.")

    def reset(self):
        # reset with enhanced initialization
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0.0
        
        # track initial and max portfolio value
        self.initial_portfolio_value = self.cash
        self.max_portfolio_value = self.cash
        
        return self._get_observation()

    # need improvement i think, plz chat if u have any ideas for this
    def step(self, action):
        # current price
        price = self.data.iloc[self.current_step]["close"]
        
        # calculate current portfolio value
        prev_value = self.cash + self.holdings * price

        transaction_fee = 0.0

        # execute action (keeping the existing action logic)
        if action == 1:  # Buy
            amount_to_spend = self.trade_fraction * self.cash
            transaction_fee = self.transaction_cost * amount_to_spend
            amount_to_spend -= transaction_fee
            quantity = amount_to_spend / price
            self.holdings += quantity
            self.cash -= (amount_to_spend + transaction_fee)
        elif action == 2 and self.holdings > 0:  # Sell
            quantity = self.trade_fraction * self.holdings
            trade_value = quantity * price
            transaction_fee = self.transaction_cost * trade_value
            self.holdings -= quantity
            self.cash += (trade_value - transaction_fee)

        # to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # new price and portfolio value
        new_price = self.data.iloc[self.current_step]["close"] if not done else price
        new_value = self.cash + self.holdings * new_price
        
        # update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, new_value)
        
        # calculate returns and volatility
        # use a rolling window of recent returns to estimate volatility
        recent_window = max(10, min(50, self.current_step))
        recent_prices = self.data['close'].iloc[max(0, self.current_step - recent_window):self.current_step]
        returns = recent_prices.pct_change().dropna()
        
        # calculate volatility (standard deviation of returns)
        volatility = returns.std() if len(returns) > 0 else 0.01
        
        # porto return
        portfolio_return = (new_value - prev_value) / prev_value
        
        # risk-free rate (annualized, convert to per-step)
        # assuming a conservative risk-free rate of 2% annually
        risk_free_rate = 0.02
        steps_per_year = 365 * 24  # hourly data
        risk_free_rate_per_step = (1 + risk_free_rate) ** (1/steps_per_year) - 1
        
        # Sharpe-like reward calculation
        # add small epsilon so no division by zero
        if volatility > 0:
            sharpe_reward = (portfolio_return - risk_free_rate_per_step) / (volatility + 1e-8)
        else:
            sharpe_reward = portfolio_return
        
        # penalize transaction costs
        transaction_penalty = transaction_fee / self.initial_portfolio_value
        
        # final reward combines Sharpe-like metric with transaction penalty
        reward = sharpe_reward - transaction_penalty
        
        # add a small exploration bonus/penalty
        if action == 0:  # hold action
            reward -= 0.001
        
        obs = self._get_observation()
        return obs, reward, done, {
            'portfolio_value': new_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'portfolio_return': portfolio_return,
            'volatility': volatility
        }

    def _get_observation(self):
        # get observation vector (only numeric columns)
        numeric_data = self.data.select_dtypes(include=[np.number])
        obs = numeric_data.iloc[self.current_step].values[:self.obs_dim].astype("float32")
        return obs

    # only 4 testing initially
    # def _load_market_data(self):
    #     dates = pd.date_range(start="2020-01-01", periods=500, freq="H")
        
    #     base_price = 100
        
    #     # trend component
    #     trend = np.linspace(0, 30, 500)
        
    #     # seasonality component (simulating market cycles)
    #     seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, 500))
        
    #     # volatility component
    #     volatility = np.random.normal(0, 5, 500)
        
    #     # combine components
    #     prices = base_price + trend + seasonality + volatility
        
    #     # non-negative prices
    #     prices = np.maximum(prices, 1)
        
    #     data = pd.DataFrame({
    #         "open": prices,
    #         "high": prices + np.abs(np.random.normal(0, 3, 500)),
    #         "low": prices - np.abs(np.random.normal(0, 3, 500)),
    #         "close": prices,
    #         "volume": np.random.rand(500) * 1000,
    #         "open_time": dates,
    #         "close_time": dates + pd.Timedelta(hours=1)
    #     })
    #     return data.reset_index(drop=True)