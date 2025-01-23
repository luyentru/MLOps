#!/usr/bin/env bash

# Strict error handling
set -euo pipefail
IFS=$'\n\t'

# Create required directories
mkdir -p ./reports/logs
mkdir -p ./reports/figures
mkdir -p ./models
mkdir -p ./vertex_train/queue
mkdir -p ./vertex_train/completed

# Fetch files from remote
gsutil cp "${STORAGE_URI}"/vertex_train/queue/* ./vertex_train/queue/ || { echo "Fetching config files from remote failed"; exit 1; }

# Copy data
gsutil -m cp -r "${STORAGE_URI}/data" ./ || { echo "Fetching data from remote failed"; exit 1; }

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
python3.11 -u ./src/pet_fac_rec/train.py experiment="${FILENAME}"

# Move logs and models to completed folder
mv ./reports/logs/* ./reports/figures/* ./src/pet_fac_rec/configs/* ./models/* ./vertex_train/completed/"${FILENAME}"/
gsutil cp -r ./vertex_train/completed/"${FILENAME}" "${STORAGE_URI}"/vertex_train/completed/

# Clean up config and dvc files in remote
gsutil rm "${STORAGE_URI}"/vertex_train/queue/"$(basename "$CONFIG_FILE")"

