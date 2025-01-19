#!/usr/bin/env bash

# Strict error handling
set -euo pipefail
IFS=$'\n\t'

# Check required environment variables
if [ -z "${STORAGE_URI:-}" ]; then
    echo "Error: STORAGE_URI environment variable is not set"
    exit 1
fi

if [ -z "${CONFIG_NAME:-}" ]; then
    echo "Error: CONFIG_NAME environment variable is not set"
    exit 1
fi

# Create required directories
mkdir -p /reports/logs
mkdir -p /reports/figures
mkdir -p /models

# Copy config file
gsutil cp "${STORAGE_URI}/vertex_train/queue/${CONFIG_NAME}.yaml" /src/pet_fac_rec/configs/experiment

# Copy data
gsutil -m cp -r "${STORAGE_URI}/data" /

# Run training
python3 -u src/pet_fac_rec/train.py experiment="${CONFIG_NAME}"

# Copy logs and models to completed folder
gsutil cp -r /reports/logs "${STORAGE_URI}/vertex_train/completed/${CONFIG_NAME}/logs"
gsutil cp -r /reports/figures "${STORAGE_URI}/vertex_train/completed/${CONFIG_NAME}/figures"
gsutil cp -r /models "${STORAGE_URI}/vertex_train/completed/${CONFIG_NAME}/models"

# Copy and remove config file
gsutil cp "${STORAGE_URI}/vertex_train/queue/${CONFIG_NAME}.yaml" "${STORAGE_URI}/vertex_train/completed/${CONFIG_NAME}/"
gsutil rm "${STORAGE_URI}/vertex_train/queue/${CONFIG_NAME}.yaml"