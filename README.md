Scraper + Processing + Trade Bot
================================

**Project Structure**
```
project
├── 📁 config
│   ├── 🔧 config.yaml                # main configuration file
│   └── 🛠️ model_hyperparameters.yaml # model-specific hyperparameters
│
├── 📂 src
│   ├── 📊 data
│   │   ├── 🧮 feature_engineering.py  # feature engineering
│   │   ├── 🔬 processor.py            # processing utilities
│   │   └── 🕸️ scraper.py              # scraping scripts
│   │
│   ├── 🤖 environment
│   │   └── 🎮 trading_env.py          # trading environment for RL
│   │
│   ├── 🧠 models
│   │   └── 📈 rllib_policy.py         # RLlib policy implementation
│   │
│   └── 🚀 main.py                     # main training/execution script
```

Using Python3.9 is recommended

**Setup**
```
# clone the repo
git clone https://github.com/Bzz05/Scraping-TradeBot.git
cd Scraping-TradeBot

# create virtual env (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install --upgrade pip==21.0
pip install --upgrade setuptools==57.5.0
pip install --upgrade wheel==0.37.0
pip install -r requirements.txt
```

**Scrape**
```
python src/data/scraper.py
```

**Processing**
```
python src/data/processor.py
```

**Start Training**
```
python3 src/main.py --mode train --config config/config.yaml
```

## License
This project is licensed under the terms of the **MIT** license.
