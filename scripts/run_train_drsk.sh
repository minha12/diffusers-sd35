#!/bin/bash

export BASE_DIR="/home/ubuntu"
export MODEL_DIR="${BASE_DIR}/models/stabilityai/stable-diffusion-3-medium-diffusers"
export OUTPUT_DIR="sd3.5-controlnet-out-drsk"
export CACHE_DIR="${BASE_DIR}/datasets/drsk/cache-sd3.5-full"
export DATASET_DIR="${BASE_DIR}/datasets/drsk"
export SCRIPT_PATH="./scripts/drsk.py"

accelerate launch --config_file accelerate_config.yaml train_controlnet_sd35.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir=$DATASET_DIR \
    --dataset_cache_dir=$CACHE_DIR \
    --dataset_script_path=$SCRIPT_PATH \
    --resolution=512 \
    --learning_rate=1e-5 \
    --dataset_preprocess_batch_size=64 \
    --max_train_steps=15000 \
    --validation_image "./validation_images/control_image_1.png" "./validation_images/control_image_2.png" \
    --validation_prompt "pathology image: tissue unknown 12.73%, dermis normal skin 22.91%, skin appendage structure normal skin 61.62%" "pathology image: dermis normal skin 47.76%, skin appendage structure normal skin 51.06%" \
    --validation_steps=300 \
    --num_validation_images=4 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4 \
    --use_8bit_adam

