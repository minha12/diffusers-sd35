# SD3 ControlNet Inference Guide

This guide explains how to use a trained SD3 ControlNet model for inference, detailing the process of generating images from control images.

## Setup

```python
import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.utils import load_image

# Load the pipeline
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",  # Base model
    controlnet="path/to/your/controlnet",               # Your trained controlnet
    torch_dtype=torch.float16                           # Use float16 for efficiency
)
pipe.to("cuda")
```

## Steps for Inference

1. **Prepare Control Image**
   - Load your control image (e.g., canny edge map, depth map, etc.)
   ```python
   control_image = load_image("path/to/control_image.png")
   ```

2. **Basic Generation**
   ```python
   prompt = "Your prompt here"
   image = pipe(
       prompt,
       control_image=control_image,
       height=1024,                           # Default resolution
       width=1024,
       num_inference_steps=28,                # Default steps
       controlnet_conditioning_scale=0.7      # Control strength
   ).images[0]
   ```

3. **Advanced Parameters**

   a. Control Guidance Timing:
   ```python
   # When the controlnet starts and stops affecting generation
   image = pipe(
       prompt,
       control_image=control_image,
       control_guidance_start=0.0,  # Start from beginning (0.0) to end (1.0)
       control_guidance_end=1.0,
   ).images[0]
   ```

   b. Multiple Control Images:
   ```python
   # For multiple controlnets
   control_images = [control_image1, control_image2]
   control_scales = [0.7, 0.5]  # Different scales for each control
   
   image = pipe(
       prompt,
       control_image=control_images,
       controlnet_conditioning_scale=control_scales,
   ).images[0]
   ```

4. **Save Results**
   ```python
   image.save("output.png")
   ```

## Important Parameters

- `controlnet_conditioning_scale`: Controls strength of the conditioning (0-1)
- `num_inference_steps`: More steps = higher quality but slower
- `guidance_scale`: Controls how closely to follow the prompt (higher = closer)
- `control_guidance_start/end`: When the control affects the generation
- `height/width`: Output image dimensions (must be multiples of 8)

## Tips

1. **Control Strength**: 
   - Start with `controlnet_conditioning_scale=0.7`
   - Increase for stronger control, decrease for more creative freedom

2. **Resolution**:
   - Must be multiples of 8
   - Higher resolution = more VRAM usage
   - Common sizes: 1024x1024, 768x768

3. **Memory Management**:
   ```python
   # Enable memory efficient attention
   pipe.enable_xformers_memory_efficient_attention()
   
   # Or use torch.compile for better performance
   pipe.transformer = torch.compile(pipe.transformer)
   ```

4. **Batch Processing**:
   ```python
   # Generate multiple images
   images = pipe(
       prompt,
       control_image=control_image,
       num_images_per_prompt=4,
   ).images
   ```

## Common Issues

1. **Out of Memory**:
   - Reduce batch size
   - Use lower resolution
   - Enable memory efficient attention
   - Use float16 precision

2. **Poor Results**:
   - Try different `controlnet_conditioning_scale` values
   - Adjust `guidance_scale`
   - Increase `num_inference_steps`
   - Improve control image quality

3. **Slow Generation**:
   - Use float16 precision
   - Enable xformers
   - Use torch.compile
   - Reduce number of steps

## Example Full Script

```python
import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.utils import load_image

def setup_pipeline():
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet="path/to/your/controlnet",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

def generate_image(pipe, control_image, prompt):
    return pipe(
        prompt,
        control_image=control_image,
        height=1024,
        width=1024,
        num_inference_steps=28,
        controlnet_conditioning_scale=0.7,
        guidance_scale=7.0
    ).images[0]

def main():
    pipe = setup_pipeline()
    control_image = load_image("control_image.png")
    prompt = "Your detailed prompt here"
    
    image = generate_image(pipe, control_image, prompt)
    image.save("output.png")

if __name__ == "__main__":
    main()
```

# SD3 ControlNet Pipeline Internals

This document details what happens inside the `StableDiffusion3ControlNetPipeline` during image generation.

## Pipeline Components

The pipeline consists of several key components:
- VAE (Variational Autoencoder)
- Text Encoders (CLIP x2 and T5)
- ControlNet model
- Transformer (Diffusion model)
- Flow-based scheduler

## Step-by-Step Generation Process

1. **Text Encoding**
   ```python
   prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
       prompt=prompt,
       prompt_2=prompt_2,  # Optional secondary prompt
       prompt_3=prompt_3,  # Optional T5 prompt
       device=device,
       num_images_per_prompt=num_images_per_prompt,
       do_classifier_free_guidance=do_guidance
   )
   ```
   - Encodes prompts using CLIP (x2) and T5 encoders
   - Combines embeddings for joint attention
   - Creates negative embeddings for classifier-free guidance

2. **Control Image Processing**
   ```python
   # Process control image through VAE
   control_image = pipeline.prepare_image(
       image=control_image,
       width=width,
       height=height,
       batch_size=batch_size,
       num_images_per_prompt=num_images_per_prompt,
       device=device,
       dtype=dtype
   )
   control_image = pipeline.vae.encode(control_image).latent_dist.sample()
   control_image = (control_image - vae_shift_factor) * vae.config.scaling_factor
   ```
   - Resizes and normalizes control image
   - Encodes to latent space using VAE
   - Applies scaling and shifting

3. **Initial Latents Generation**
   ```python
   latents = pipeline.prepare_latents(
       batch_size * num_images_per_prompt,
       num_channels_latents,
       height,
       width,
       prompt_embeds.dtype,
       device,
       generator,
   )
   ```
   - Creates random noise as starting point
   - Matches shape requirements for denoising

4. **Denoising Loop**
   For each timestep:

   a. **ControlNet Conditioning**
   ```python
   control_block_samples = pipeline.controlnet(
       hidden_states=latent_model_input,
       timestep=timestep,
       encoder_hidden_states=controlnet_encoder_hidden_states,
       pooled_projections=controlnet_pooled_projections,
       controlnet_cond=control_image,
       conditioning_scale=controlnet_conditioning_scale,
   )
   ```
   - Processes current latents with control image
   - Outputs conditioning signals for transformer

   b. **Transformer Denoising**
   ```python
   noise_pred = pipeline.transformer(
       hidden_states=latent_model_input,
       timestep=timestep,
       encoder_hidden_states=prompt_embeds,
       pooled_projections=pooled_prompt_embeds,
       block_controlnet_hidden_states=control_block_res_samples,
   )
   ```
   - Uses text embeddings and ControlNet signals
   - Predicts noise to remove

   c. **Guidance Scale Application** (if using classifier-free guidance)
   ```python
   noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
   noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
   ```
   
   d. **Scheduler Step**
   ```python
   latents = pipeline.scheduler.step(noise_pred, timestep, latents).prev_sample
   ```
   - Updates latents using predicted noise
   - Uses flow-matching scheduler

5. **Final Image Decoding**
   ```python
   # Scale and decode the image latents with vae
   latents = (1 / vae.config.scaling_factor) * latents
   image = pipeline.vae.decode(latents).sample
   image = (image / 2 + 0.5).clamp(0, 1)
   ```
   - Rescales latents
   - Decodes through VAE
   - Converts to pixel space

## Key Parameters

1. **controlnet_conditioning_scale** (float, default=1.0)
   - Controls strength of ControlNet conditioning
   - Higher values = stronger adherence to control image

2. **guidance_scale** (float, default=7.0)
   - Controls text prompt adherence
   - Higher values = closer to prompt but potentially less quality

3. **num_inference_steps** (int, default=28)
   - Number of denoising steps
   - More steps = higher quality but slower

4. **control_guidance_start/end** (float, 0.0-1.0)
   - When ControlNet starts/stops affecting generation
   - Allows fine control over timing of conditioning

## Memory Optimization Features

1. **Memory-Efficient Attention**
   ```python
   pipeline.enable_xformers_memory_efficient_attention()
   ```

2. **Model Offloading**
   ```python
   pipeline.enable_model_cpu_offload()
   ```

3. **Gradient Checkpointing**
   ```python
   pipeline.enable_gradient_checkpointing()
   ```

## Example Usage with Parameters

```python
image = pipeline(
    prompt="a bird in space",
    control_image=control_image,
    num_inference_steps=28,
    controlnet_conditioning_scale=0.7,
    guidance_scale=7.0,
    control_guidance_start=0.0,
    control_guidance_end=1.0,
    height=1024,
    width=1024
).images[0]
```