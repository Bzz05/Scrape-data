FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential git

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade \
    pip==21.0 \
    setuptools==57.5.0 \
    wheel==0.37.0 \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN mkdir -p /app/data/historical

CMD ["python", "src/main.py", "--mode", "train"]