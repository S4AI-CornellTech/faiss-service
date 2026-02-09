#!/bin/bash
set -e  # stop on errors

# Initialize conda for this shell session
source "$HOME/anaconda3/etc/profile.d/conda.sh"

# Activate the environment
conda activate faiss-service

make pyclient
python query_server.py