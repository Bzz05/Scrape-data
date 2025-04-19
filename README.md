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


## docker
docker build -t tradebot .
docker run tradebot
## run with gpu
docker run --gpus all -it tradebot



## docker
docker build -t btcbot .
docker run tradebot
## run with gpu
docker run --gpus all -it btcbot



## cuda
nvidia-smi



## save 
docker start keen_proskuriakova
docker exec -it keen_proskuriakova /bin/bash

ls /app
ls /workspace

docker cp keen_proskuriakova:/app/data "C:\2. Code Repository\200. Projects\PeanutFund\Scraping-TradeBot\data"

docker cp keen_proskuriakova:/app/ray_results "C:\2. Code Repository\200. Projects\PeanutFund\Scraping-TradeBot\ray_results"

## run and save output in local and docker
docker run -v /absolute/path/on/host:/app/output my_image
docker run -v "C:\myresults":/app/output btcbot


is how you mount a volume — a shared folder between your host machine and the Docker container.
“Hey Docker, anything written to /app/output inside the container should actually be saved at /absolute/path/on/host on my computer.”