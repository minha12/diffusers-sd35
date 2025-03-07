import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from diffusers.utils import numpy_to_pil

from diffusers import StableDiffusion3ControlNetPipeline

def sd3_controlnet_inference(
    prompt,
    control_image,
    transformer,
    controlnet,
    vae,
    text_encoders,
    tokenizers,
    scheduler,
    generator=None,
    num_inference_steps=20,
    height=512,
    width=512,
    guidance_scale=5.0,
    max_sequence_length=77,
    device="cuda",
    dtype=torch.float16,
    num_images_per_prompt=1,
    vae_scale_factor = 8  # Default VAE scale factor
):
    """
    Custom SD3 ControlNet inference function without using the pipeline.
    
    Args:
        prompt (str): The text prompt to guide the image generation
        control_image (PIL.Image.Image): The conditioning image for ControlNet
        transformer (SD3Transformer2DModel): The transformer model for diffusion
        controlnet (SD3ControlNetModel): The controlnet model
        vae (AutoencoderKL): The VAE model for encoding/decoding images
        text_encoders (list): List of three text encoders [text_encoder_one, text_encoder_two, text_encoder_three]
        tokenizers (list): List of three tokenizers [tokenizer_one, tokenizer_two, tokenizer_three]
        scheduler (FlowMatchEulerDiscreteScheduler): The noise scheduler
        generator (torch.Generator, optional): Random number generator
        num_inference_steps (int): Number of denoising steps
        height (int): Output image height
        width (int): Output image width
        guidance_scale (float): Scale for classifier-free guidance
        max_sequence_length (int): Max token sequence length for T5
        device (str): Device to run inference on
    
    Returns:
        List[PIL.Image.Image]: The generated images
    """
    # 1. Check for devices and dtype
    device = torch.device(device)
    
    # Move models to correct dtype
    vae = vae.to(dtype=dtype)
    transformer = transformer.to(dtype=dtype)
    controlnet = controlnet.to(dtype=dtype)
    for encoder in text_encoders:
        if encoder is not None:
            encoder.to(dtype=dtype)
    
    # 2. Process inputs
    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)
    
    if not isinstance(control_image, list):
        control_image = [control_image]
    
    if len(control_image) == 1 and batch_size > 1:
        control_image = control_image * batch_size
        
    if len(control_image) != batch_size:
        raise ValueError(
            f"Number of prompts ({batch_size}) and control images ({len(control_image)}) must match"
        )
    
    # 3. Generate text embeddings
    do_classifier_free_guidance = guidance_scale > 1.0
    
    # We need to import and use the pipeline's implementation
    
    
    # Create a minimal temporary pipeline to use its encode_prompt
    pipeline = StableDiffusion3ControlNetPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoders[0],
        text_encoder_2=text_encoders[1],
        text_encoder_3=text_encoders[2],
        tokenizer=tokenizers[0],
        tokenizer_2=tokenizers[1],
        tokenizer_3=tokenizers[2],
        scheduler=scheduler,
        controlnet=controlnet,
    )
    
    # Use the pipeline's encode_prompt
    (prompt_embeds, 
        negative_prompt_embeds, 
        pooled_prompt_embeds, 
        negative_pooled_prompt_embeds) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,  # Use the same prompt for all encoders
        prompt_3=prompt,  # Use the same prompt for all encoders
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=[""] * batch_size if do_classifier_free_guidance else None,
        max_sequence_length=max_sequence_length,
    )
    
    # Combine into the format expected by the rest of the function
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Process control images
    # Use pipeline's image processor if available
    if pipeline is not None and hasattr(pipeline, "image_processor"):
        control_image_processor = pipeline.image_processor
    else:
        control_image_processor = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
    
    # Process all control images in batch
    control_image_tensors = []
    for image in control_image:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(control_image_processor, transforms.Compose):
            image_tensor = control_image_processor(image).unsqueeze(0)
        else:
            # Using pipeline's VaeImageProcessor
            image_tensor = control_image_processor.preprocess(image, height=height, width=width).unsqueeze(0)
        control_image_tensors.append(image_tensor)
    
    # Stack all control images into a single batch tensor
    control_image_tensor = torch.cat(control_image_tensors, dim=0).to(device=device, dtype=dtype)
    
    # Repeat for num_images_per_prompt
    if num_images_per_prompt > 1:
        control_image_tensor = control_image_tensor.repeat_interleave(num_images_per_prompt, dim=0)
    
    # If using classifier-free guidance, duplicate the control image
    if do_classifier_free_guidance:
        control_image_tensor = torch.cat([control_image_tensor] * 2, dim=0)
    
    # 5. Create initial noise for latent - USE EXACT SAME APPROACH AS PIPELINE
    
    latents_shape = (batch_size * num_images_per_prompt, transformer.config.in_channels, height // vae_scale_factor, width // vae_scale_factor)
    
    # Use the same generator and random state as the pipeline
    if generator is None:
        latents = torch.randn(latents_shape, device=device, dtype=dtype)
    else:
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
    
    # 6. Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # 7. Encode control image with VAE
    with torch.no_grad():
        # Match the pipeline's VAE encoding
        control_image_latents = vae.encode(control_image_tensor).latent_dist.sample()
        # Apply scaling factor
        vae_shift_factor = getattr(vae.config, "shift_factor", 0)
        control_image_latents = (control_image_latents - vae_shift_factor) * vae.config.scaling_factor
    
    # Check if controlnet has specific pooled projection requirements
    controlnet_config = getattr(controlnet, "config", None)
    if controlnet_config and getattr(controlnet_config, "force_zeros_for_pooled_projection", False):
        # Use zeros for pooled projections as specified in config
        controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
    else:
        # Use normal pooled projections
        controlnet_pooled_projections = pooled_prompt_embeds
    
    # 8. Denoising loop
    for i, t in enumerate(timesteps):
        # Expand timestep
        timestep = t
        timestep_batch = timestep.expand(latents.shape[0])
        
        # Get ControlNet conditioning
        with torch.no_grad():
            # ControlNet forward pass
            control_block_samples = controlnet(
                hidden_states=latents,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=controlnet_pooled_projections,
                controlnet_cond=control_image_latents,
                return_dict=False,
            )[0]
        
        # Predict the noise residual with the transformer
        with torch.no_grad():
            # Transformer forward pass with controlnet blocks
            model_pred = transformer(
                hidden_states=latents,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,
                return_dict=False,
            )[0]
        
        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample: x_t -> x_t-1
        latents = scheduler.step(model_pred, t, latents).prev_sample
    
    # 9. Decode the image using VAE
    with torch.no_grad():
        # Scale latents back to VAE range
        latents = 1 / vae.config.scaling_factor * latents
        
        # Add VAE shift factor if present
        if hasattr(vae.config, "shift_factor"):
            latents = latents + vae.config.shift_factor
            
        # Decode latents to images
        images = vae.decode(latents).sample
    
    # 10. Convert to PIL Images
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    
    return numpy_to_pil(images)