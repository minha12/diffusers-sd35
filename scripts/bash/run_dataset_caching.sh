#!/bin/bash

export DATASET_NAME="drsk"

export BASE_DIR="/home/ubuntu"
export MODEL_DIR="${BASE_DIR}/models/stabilityai/stable-diffusion-3-medium-diffusers"
export CACHE_DIR="${BASE_DIR}/datasets/${DATASET_NAME}/cache-sd3.5-${DATASET_NAME}"
export DATASET_DIR="${BASE_DIR}/datasets/${DATASET_NAME}"
export SCRIPT_PATH="./src/datasets/${DATASET_NAME}.py"

# Run the dataset caching script
accelerate launch --config_file configs/accelerate_config.yaml src/caching/dataset_cache.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$CACHE_DIR \
    --train_data_dir=$DATASET_DIR \
    --dataset_cache_dir=$CACHE_DIR \
    --dataset_script_path=$SCRIPT_PATH \
    --resolution=512 \
    --dataset_preprocess_batch_size=16 \
    --max_sequence_length=77 \
    --mixed_precision bf16