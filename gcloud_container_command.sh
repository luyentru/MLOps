#!/usr/bin/env bash

# Strict error handling
set -euo pipefail
IFS=$'\n\t'

# # Check required environment variables
# if [ -z "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ]; then
#     echo "Error: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is not set"
#     exit 1
# fi

# Create required directories
mkdir -p ./reports/logs
mkdir -p ./reports/figures
mkdir -p ./models
mkdir -p ./vertex_train/queue
mkdir -p ./vertex_train/completed

# Initialize DVC and configure remote
dvc init --no-scm || { echo "DVC initialization failed"; exit 1; }
dvc remote add -d pet-fac-rec-bucket gs://pet-fac-rec-bucket/
# dvc remote modify pet-fac-rec-bucket credentialpath /tmp/gcp-credentials.json

# Pull .dvc files from git
git clone https://${GITHUB_TOKEN}@https://github.com/luyentru/MLOps/tree/dvc_revisions /workspace
cd /workspace

# Pull data using DVC
dvc pull || { echo "DVC pull failed"; exit 1; }
dvc status || { echo "DVC status check failed"; exit 1; }

# Move a config file from queue to configs
CONFIG_FILE=$(find ./vertex_train/queue -type f | head -n 1)
if [ -z "$CONFIG_FILE" ]; then
  echo "No config file found in ./vertex_train/queue."
  exit 1
fi
mv "$CONFIG_FILE" ./src/pet_fac_rec/configs/experiment/

# Create output folder
FILENAME=$(basename "$CONFIG_FILE" .yaml)
mkdir -p "./vertex_train/completed/${FILENAME}"

# Run training
python3.11 -u src/pet_fac_rec/train.py experiment="${FILENAME}"

# Move logs and models to completed folder
mv /reports/logs/* /reports/figures/* /src/pet_fac_rec/configs/* /models/* /vertex_train/completed/"${FILENAME}"/

# Push data using DVC
dvc add ./vertex_train/
dvc push