FROM nvidia/cuda:12.3.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

WORKDIR /workspace