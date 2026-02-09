#!/bin/bash
set -e  # stop on errors

# Initialize conda for this shell session
source "$HOME/anaconda3/etc/profile.d/conda.sh"

# Activate the environment
conda activate faiss-service

# Run your FAISS server
./build/bin/faiss_server -file_path synthetic_index_monolithic_10m.faiss -on_cpu true
