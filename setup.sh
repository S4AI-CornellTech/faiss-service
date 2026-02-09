#!/bin/bash
set -e  # stop on any error

# Conda
conda env create -f environment.yml -y
conda activate faiss-service

# System dependencies
sudo apt update
sudo apt install -y civetweb libcivetweb-dev intel-mkl-full protobuf-compiler-grpc \
    libssl-dev cmake libgrpc++-dev protobuf-compiler libspdlog-dev \
    libgrpc-dev libgflags-dev libgtest-dev libc-ares-dev libprotobuf-dev

# Python dependencies
pip install --quiet gdown

# Download dataset
gdown --id 1xFBQnltn_KtwSjE-aGIChgwxtyKroneJ -O triviaqa_encodings.npy

# Build index
python create_synthetic_index.py --index-size 10m --dim 768
make cppclient

# GRPC cmake setup
sudo mkdir -p /usr/lib/x86_64-linux-gnu/cmake/grpc
sudo cp grpc-config.cmake /usr/lib/x86_64-linux-gnu/cmake/grpc

# Build server
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../

# Build docker image
docker build -t faiss-server .
