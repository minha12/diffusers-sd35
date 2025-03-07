python debug_sd3_controlnet_simple.py \
  --model_path="/proj/berzelius-2023-296/users/x_lemin/models/stabilityai/stable-diffusion-3.5-medium-diffusers" \
  --controlnet_path="/proj/berzelius-2023-296/users/x_lemin/diffusers/examples/controlnet/sd3.5-controlnet-out-drsk/checkpoint-15000/controlnet" \
  --control_image="validation_images/control_image_1.png" \
  --prompt="pathology image: tissue unknown 12.73%, dermis normal skin 22.91%, skin appendage structure normal skin 61.62%" \
  --height=512 \
  --width=512 \
  --guidance_scale=5.0 \
  --fp16 \
  --seed=42