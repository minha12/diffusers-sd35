import torch
import fire
from PIL import Image
import os
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
)
from transformers import CLIPTokenizer, T5TokenizerFast
from train_controlnet_sd3 import import_model_class_from_model_name_or_path
from SD35SinglePatch import sd3_controlnet_inference

def load_models(model_path, controlnet_path, device="cuda", dtype=torch.float16):
    """Load all required models for SD3 ControlNet inference"""
    
    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    
    print("Loading text encoders and tokenizers...")
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer"
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer_2"
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        model_path, subfolder="tokenizer_3"
    )
    
    # Get text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        model_path, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        model_path, None, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        model_path, None, subfolder="text_encoder_3"
    )
    
    # Load text encoders with dtype
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        model_path, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    text_encoder_three = text_encoder_cls_three.from_pretrained(
        model_path, subfolder="text_encoder_3", torch_dtype=dtype
    ).to(device)
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    ).to(device)
    
    print("Loading transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    
    print(f"Loading ControlNet from {controlnet_path}...")
    controlnet = SD3ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=dtype
    ).to(device)
    
    return {
        "transformer": transformer,
        "controlnet": controlnet,
        "vae": vae,
        "text_encoders": [text_encoder_one, text_encoder_two, text_encoder_three],
        "tokenizers": [tokenizer_one, tokenizer_two, tokenizer_three],
        "scheduler": scheduler
    }

def run_inference(
    controlnet_path,
    control_image,
    prompt="a photo of an astronaut riding a horse on mars",
    model_path="stabilityai/stable-diffusion-3-medium",
    output_path="sd3_controlnet_output.png",
    num_inference_steps=20,
    guidance_scale=5.0,
    height=512,
    width=512,
    max_sequence_length=77,
    seed=None,
    num_images_per_prompt=1,
    output_dir=None,
    dtype=torch.float16,  # Add dtype parameter
):
    """
    Run SD3 ControlNet inference with the provided parameters.
    
    Args:
        controlnet_path (str): Path to trained ControlNet checkpoint
        control_image (str or List[str]): Path to control image(s)
        prompt (str or List[str]): Prompt(s) for image generation
        model_path (str): Path to SD3 model or model ID from huggingface.co
        output_path (str): Path to save the generated image (used when generating a single image)
        num_inference_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale for classifier-free guidance
        height (int): Output image height
        width (int): Output image width
        max_sequence_length (int): Max sequence length for T5 encoder
        seed (int, optional): Random seed for reproducibility
        num_images_per_prompt (int): Number of images to generate per prompt
        output_dir (str, optional): Directory to save multiple generated images
        dtype (torch.dtype): Data type for model parameters
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set generator for reproducibility
    if seed is not None:
        if isinstance(seed, int):
            generator = torch.Generator(device).manual_seed(seed)
        elif isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(s) for s in seed]
        else:
            raise ValueError("Seed must be an integer or a list of integers")
    else:
        generator = None
    
    # Process inputs to support lists
    if isinstance(prompt, str):
        prompt = [prompt]
        
    if isinstance(control_image, str):
        control_image = [control_image]
    
    # Validate inputs
    if len(prompt) != len(control_image) and len(control_image) != 1:
        raise ValueError("Number of prompts and control images must be the same, or provide a single control image")
    
    # If a single control image is provided, repeat it for all prompts
    if len(control_image) == 1 and len(prompt) > 1:
        control_image = control_image * len(prompt)
    
    # Load models with dtype
    models = load_models(model_path, controlnet_path, device, dtype)
    
    # Set evaluation mode
    for model_name, model in models.items():
        if isinstance(model, list):
            for m in model:
                if isinstance(m, torch.nn.Module):
                    m.eval()
        elif isinstance(model, torch.nn.Module):
            model.eval()
    
    # Load control images
    control_images_pil = []
    for img_path in control_image:
        control_images_pil.append(Image.open(img_path).convert("RGB"))
    
    print(f"Running inference with {len(prompt)} prompt(s), generating {num_images_per_prompt} image(s) per prompt")
    # Run inference
    generated_images = sd3_controlnet_inference(
        prompt=prompt,
        control_image=control_images_pil,
        transformer=models["transformer"],
        controlnet=models["controlnet"],
        vae=models["vae"],
        text_encoders=models["text_encoders"],
        tokenizers=models["tokenizers"],
        scheduler=models["scheduler"],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        max_sequence_length=max_sequence_length,
        generator=generator,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        dtype=dtype
    )
    
    # Save results
    if len(generated_images) == 1 and output_dir is None:
        # Single image case
        generated_images[0].save(output_path)
        print(f"Image saved to {output_path}")
    else:
        # Multiple images case
        if output_dir is None:
            # Default to current directory
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
            base_filename = os.path.basename(output_path).split(".")[0]
        else:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = "sd3_controlnet_output"
            
        for i, image in enumerate(generated_images):
            img_path = os.path.join(output_dir, f"{base_filename}_{i}.png")
            image.save(img_path)
            print(f"Image saved to {img_path}")

if __name__ == "__main__":
    fire.Fire(run_inference)