# Scraper + Processing + Trade Bot

Using Python3.9 is recommended

**Project Structure**

```
project
├── 📁 config
│   ├── 🔧 config.yaml                  # main configuration file
│   └── 🛠️ model_hyperparameters.yaml   # model-specific hyperparameters
│
├── 📁 kind
│   └── ⚙️ kind-config.yaml             # kind cluster configuration
│
├── 📁 k8s
│   ├── 📦 deployment.yaml              # Kubernetes Deployment spec
│   └── 🌐 service.yaml                 # Kubernetes Service spec
│
├── 🐳 Dockerfile                       # Docker image definition
│
├── 📂 src
│   ├── 📊 data
│   │   ├── 🧮 feature_engineering.py    # feature engineering
│   │   ├── 🔬 processor.py              # processing utilities
│   │   └── 🕸️ scraper.py                # scraping scripts
│   │
│   ├── 🤖 environment
│   │   └── 🎮 trading_env.py            # trading environment for RL
│   │
│   ├── 🧠 models
│   │   └── 📈 rllib_policy.py           # RLlib policy implementation
│   │
│   └── 🚀 main.py                       # main training/execution script
```

**Setup**

```
# clone the repo
git clone https://github.com/Bzz05/Scraping-TradeBot.git
cd Scraping-TradeBot

# create virtual env (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install --upgrade pip==22.0
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

**Run with Docker**

```
# build image
docker build -t trading-bot:latest .

# run container
docker run --gpus all -it trading-bot:latest
```

**To use Kubernetes (local testing with kind)**
```
# build image for Kubernetess to use
docker build -t trading-bot:latest

# make cluster
kind create cluster --config kind-config.yaml

# load the docker image to kind
kind load docker-image trading-bot:latest

# deploy and expose the service
kubectl apply -f k8s/

# port forward
kubectl port-forward svc/trading-bot-service 8080:80

# browse live logs
http://localhost:8080/logs
```

## License

This project is licensed under the terms of the **MIT** license.
