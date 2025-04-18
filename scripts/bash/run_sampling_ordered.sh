#!/bin/bash

# Configure environment variables
export MODEL_PATH="/proj/berzelius-2023-296/users/x_lemin/models/stabilityai/stable-diffusion-3.5-medium-diffusers"
export CONTROLNET_PATH="/proj/berzelius-2023-296/users/x_lemin/diffusers/examples/controlnet/sd3.5-controlnet-out-drsk/checkpoint-15000/controlnet"
export OUTPUT_DIR="./multi_tile_results"
export PROMPT="pathology image of skin with a focus on the dermis and skin appendage structures"
export CONTROL_IMAGE="./large-content/squamous_mask_colored.png"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run multi-tile generation with ControlNet
python sampling_ordered.py \
  --prompt="$PROMPT" \
  --model_path="$MODEL_PATH" \
  --controlnet_path="$CONTROLNET_PATH" \
  --control_image="$CONTROL_IMAGE" \
  --output_dir="${OUTPUT_DIR}/with_control" \
  --image_size=2048 \
  --patch_size=512 \
  --overlap_ratio=4 \
  --guidance_scale=5.0 \
  --sampling_steps=30 \
  --n_images=1 \
  --dtype_str="float32" \
  --tile_order="corner_to_center" \
  --use_region_prompt=True \
  --use_hann_blending=False \
  --debug=True

# Optional: Run another variation without region prompts and with center-to-corner ordering
# python sampling_ordered.py \
#   --prompt="$PROMPT" \
#   --model_path="$MODEL_PATH" \
#   --controlnet_path="$CONTROLNET_PATH" \
#   --control_image="$CONTROL_IMAGE" \
#   --output_dir="${OUTPUT_DIR}/center_to_corner" \
#   --image_size=2048 \
#   --patch_size=512 \
#   --overlap_ratio=4 \
#   --guidance_scale=5.0 \
#   --sampling_steps=30 \
#   --seed=42 \
#   --n_images=1 \
#   --dtype_str="float16" \
#   --tile_order="center_to_corner" \
#   --use_region_prompt=False \
#   --debug=True

echo "Multi-tile generation complete. Results saved to $OUTPUT_DIR"