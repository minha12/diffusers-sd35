import os
import sys
import fire
import time
import torch
import numpy as np
from PIL import Image

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline
)
from transformers import CLIPTokenizer, T5TokenizerFast

from SD35OrderedPatching import SD3MultiTieDiffusion, save_tensor_as_png

def load_models(
    model_path, 
    controlnet_path=None, 
    device="cuda", 
    dtype=torch.float16
):
    """
    Load all required models for SD3 with optional ControlNet.
    
    Args:
        model_path: Path to SD3 model
        controlnet_path: Optional path to ControlNet model
        device: Device to load models on
        dtype: Data type for model parameters
    
    Returns:
        Pipeline with loaded models
    """
    print(f"Loading models from {model_path}...")
    
    # Use controlnet pipeline if controlnet_path is provided
    if controlnet_path:
        print(f"Using ControlNet from {controlnet_path}")
        pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
            model_path,
            controlnet=SD3ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype),
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
    else:
        from diffusers import StableDiffusion3Pipeline
        print("Using standard SD3 pipeline without ControlNet")
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
    
    pipeline.to(device)
    
    return pipeline

def main(
    prompt="",
    model_path="stabilityai/stable-diffusion-3-medium",
    controlnet_path=None,
    control_image=None,
    n_images=1,
    image_size=2048,
    patch_size=512,
    guidance_scale=5.0,
    sampling_steps=20,
    seed=None,
    output_dir="./results/sd3_multi_tiles",
    dtype_str="float16",
    tile_order="corner_to_center",  # Options: "corner_to_center" or "center_to_corner"
    overlap_ratio=4,  # patch_size // overlap_ratio is the overlap size
    use_region_prompt=True,
    use_hann_blending=True,
    debug=False
):
    """
    Generate large images using SD3 with tiled diffusion.
    
    Args:
        prompt: Text prompt for generation
        model_path: Path to SD3 model
        controlnet_path: Optional path to ControlNet model
        control_image: Path to control image for ControlNet
        n_images: Number of images to generate
        image_size: Size of the output image (must be divisible by 8)
        patch_size: Size of each patch (must be divisible by 8)
        guidance_scale: Classifier-free guidance scale
        sampling_steps: Number of diffusion steps
        seed: Random seed for reproducibility
        output_dir: Directory to save generated images
        dtype_str: Data type ("float16" or "float32")
        tile_order: How to order the tile processing ("corner_to_center" or "center_to_corner")
        overlap_ratio: Determines overlap size (patch_size // overlap_ratio)
        use_region_prompt: Whether to use region-specific prompts
        use_hann_blending: Whether to use Hann window blending for decoding
        debug: Whether to print debug information
    """
    # Input validation
    assert image_size % 8 == 0, "Image size must be divisible by 8"
    assert patch_size % 8 == 0, "Patch size must be divisible by 8"
    assert tile_order in ["corner_to_center", "center_to_corner"], "Invalid tile order option"
    assert overlap_ratio > 0, "Overlap ratio must be positive"
    
    # Print generation parameters
    print(f"Multi-Tile Diffusion Parameters:")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Overlap ratio: {overlap_ratio} (overlap = {patch_size//overlap_ratio} pixels)")
    print(f"  - Tile order: {tile_order}")
    print(f"  - Using region prompts: {use_region_prompt}")
    print(f"  - Using Hann window blending: {use_hann_blending}")
    print(f"  - Debug mode: {debug}")
    
    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dtype_str == "float16" else torch.float32
    
    # Set seed for reproducibility
    if seed is not None:
        if isinstance(seed, int):
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            print(f"Seed must be an integer, got {seed}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    pipeline = load_models(model_path, controlnet_path, device, dtype)
    
    # Load control image if specified
    control_img = None
    use_controlnet = controlnet_path is not None
    
    if use_controlnet and control_image:
        print(f"Loading control image from {control_image}")
        try:
            control_img = Image.open(control_image).convert("RGB")
            
            # Print image info for debugging
            print(f"Control image mode: {control_img.mode}, size: {control_img.width}x{control_img.height}")
            
            # Resize control image if needed
            if control_img.width != image_size or control_img.height != image_size:
                print(f"Resizing control image from {control_img.width}x{control_img.height} to {image_size}x{image_size}")
                control_img = control_img.resize((image_size, image_size), Image.BICUBIC)
        except Exception as e:
            print(f"Error loading control image: {e}")
            return
    
    # Print model configuration
    if debug:
        print(f"VAE input channels: {pipeline.vae.config.in_channels}")
        print(f"VAE latent channels: {pipeline.vae.config.latent_channels}")
        print(f"VAE scaling factor: {pipeline.vae.config.scaling_factor}")
        print(f"Transformer in_channels: {pipeline.transformer.config.in_channels}")
        
        if use_controlnet:
            # Print ControlNet configuration
            if hasattr(pipeline.controlnet.config, "extra_conditioning_channels"):
                print(f"ControlNet extra conditioning channels: {pipeline.controlnet.config.extra_conditioning_channels}")
            else:
                print(f"ControlNet config keys: {list(pipeline.controlnet.config.keys())}")
    
    # Initialize the multi-tile diffusion model
    multi_tile_diffusion = SD3MultiTieDiffusion(
        pipeline=pipeline,
        patch_size=patch_size,
        sampling_steps=sampling_steps,
        guidance_scale=guidance_scale,
        device=device,
        dtype=dtype,
        vae_scale_factor=8,  # SD3.5 uses a VAE with scale factor of 8
        control_image=control_img,
        use_controlnet=use_controlnet,
        image_size=image_size,
        tile_order=tile_order,
        overlap_ratio=overlap_ratio,
    )
    
    # Generate images
    print(f"Generating {n_images} image(s) with prompt: '{prompt}'")
    for i in range(n_images):
        print(f"Generating image {i+1}/{n_images}...")
        
        # Set per-image seed if needed
        if seed is not None:
            current_seed = seed + i if isinstance(seed, int) else seed
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            print(f"Using seed: {current_seed}")
        
        # Generate the image
        start_time = time.time()
        try:
            latents, image = multi_tile_diffusion(
                prompt=prompt, 
                control_image=control_img, 
                batch_size=1, 
                use_region_prompt=use_region_prompt,
                use_hann_blending=use_hann_blending,
                debug=debug
            )
            generation_time = time.time() - start_time
            
            # Convert to PIL image
            if isinstance(image, torch.Tensor):
                image = image[0].cpu()
            
            # Save the generated image
            filename_base = f"sd3_multi_tile_{i:03d}"
            if isinstance(seed, int):
                filename_base += f"_seed{seed+i}"
            
            # Create the output filename
            output_filename = os.path.join(output_dir, f"{filename_base}.png")
            
            # Save as PNG
            save_tensor_as_png(image, output_filename)
            
            print(f"Generated image saved to {output_filename}")
            print(f"Generation took {generation_time:.2f} seconds")
            
            # Save the latents for potential future use (optional)
            latents_path = os.path.join(output_dir, f"{filename_base}_latents.pt")
            torch.save(latents.cpu(), latents_path)
            print(f"Latents saved to {latents_path}")
            
            # If using controlnet, also save a copy of the control image for reference
            if use_controlnet and control_img:
                control_path = os.path.join(output_dir, f"{filename_base}_control.png")
                control_img.save(control_path)
                print(f"Control image saved to {control_path}")
                
            # Save generation parameters
            with open(os.path.join(output_dir, f"{filename_base}_params.txt"), "w") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Model path: {model_path}\n")
                f.write(f"Image size: {image_size}x{image_size}\n")
                f.write(f"Patch size: {patch_size}\n")
                f.write(f"Overlap ratio: {overlap_ratio} (overlap = {patch_size//overlap_ratio} pixels)\n")
                f.write(f"Guidance scale: {guidance_scale}\n")
                f.write(f"Sampling steps: {sampling_steps}\n")
                f.write(f"Seed: {current_seed if isinstance(seed, int) else seed}\n")
                f.write(f"Tile order: {tile_order}\n")
                f.write(f"Region prompt: {use_region_prompt}\n")
                f.write(f"Hann blending: {use_hann_blending}\n")
                f.write(f"Generation time: {generation_time:.2f} seconds\n")
            
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully generated {n_images} images at {image_size}x{image_size} resolution")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    fire.Fire(main)
