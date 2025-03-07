#!/bin/bash

# Configure environment variables
export MODEL_PATH="/proj/berzelius-2023-296/users/x_lemin/models/stabilityai/stable-diffusion-3.5-medium-diffusers"
export CONTROLNET_PATH="/proj/berzelius-2023-296/users/x_lemin/diffusers/examples/controlnet/sd3.5-controlnet-out-drsk/checkpoint-15000/controlnet"
export OUTPUT_DIR="./sd3_large_outputs"
export PROMPT="pathology image of skin with a focus on the dermis and skin appendage structures"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Example 1: Generate a standard large image without ControlNet
# python sampling_random.py \
#   --prompt="$PROMPT" \
#   --model_path="$MODEL_PATH" \
#   --output_dir="${OUTPUT_DIR}/no_control" \
#   --image_size=2048 \
#   --patch_size=512 \
#   --guidance_scale=5.0 \
#   --sampling_steps=30 \
#   --seed=42 \
#   --n_images=1 \
#   --dtype_str="float16"

# Example 2: Generate a large image with ControlNet
# Uncomment if you have a control image ready
CONTROL_IMAGE="./large-content/squamous_mask_colored.png"
python sampling_random.py \
  --prompt="$PROMPT" \
  --model_path="$MODEL_PATH" \
  --controlnet_path="$CONTROLNET_PATH" \
  --control_image="$CONTROL_IMAGE" \
  --output_dir="${OUTPUT_DIR}/with_control" \
  --image_size=2560 \
  --patch_size=512 \
  --guidance_scale=5 \
  --sampling_steps=30 \
  --n_images=1 \
  --dtype_str="float32"

# Example 3: Generate multiple images with different seeds
# python sampling_random.py \
#   --prompt="$PROMPT" \
#   --model_path="$MODEL_PATH" \
#   --output_dir="${OUTPUT_DIR}/multi_seed" \
#   --image_size=1536 \
#   --patch_size=512 \
#   --guidance_scale=7.5 \
#   --sampling_steps=25 \
#   --seed=100 \
#   --n_images=3 \
#   --dtype_str="float16"

echo "Generation complete. Results saved to $OUTPUT_DIR"
