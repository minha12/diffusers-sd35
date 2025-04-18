from huggingface_hub import snapshot_download
import os

# Define the path where you want to save the model
local_dir = "/proj/berzelius-2023-296/users/x_lemin/models/stabilityai/stable-diffusion-3-medium-diffusers"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Download the model with proper structure
model_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-3.5-medium",
    local_dir=local_dir,
    token=True,
    use_safetensors=True,
)
print(f"Model downloaded to: {model_path}")