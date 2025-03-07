import sys
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import logging
import copy

class SD3MultiTieDiffusion:
    """
    Enables tiled generation of large images using a patch-by-patch approach.
    """
    def __init__(
        self, 
        pipeline, 
        patch_size=512, 
        sampling_steps=20, 
        guidance_scale=5.0, 
        device='cuda', 
        dtype=torch.float16,
        vae_scale_factor=8,
        control_image=None,
        use_controlnet=False,
        image_size=2048,
        tile_order="corner_to_center",  # New parameter: "corner_to_center" or "center_to_corner"
        overlap_ratio=4,  # New parameter: patch_size // overlap_ratio is the overlap size
    ):
        """
        Initialize the SD3RandomDiffusion.
        
        Args:
            pipeline: Complete SD3.5 pipeline (with ControlNet if use_controlnet=True)
            patch_size: Size of each patch in pixel space (should be divisible by 8)
            sampling_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            device: Device to perform computations on
            dtype: Data type for tensors
            vae_scale_factor: VAE downsampling factor (usually 8)
            control_image: Control image for ControlNet (required if use_controlnet=True)
            use_controlnet: Whether to use ControlNet for generation
            image_size: Size of the output image
            tile_order: How to order the tile processing ("corner_to_center" or "center_to_corner")
            overlap_ratio: Determines overlap size (patch_size // overlap_ratio)
        """
        self.pipeline = pipeline
        self.patch_size = patch_size
        self.sampling_steps = sampling_steps
        self.guidance_scale = guidance_scale
        self.device = device
        self.dtype = dtype
        self.vae_scale_factor = vae_scale_factor
        self.control_image = control_image
        self.use_controlnet = use_controlnet
        self.image_size = image_size
        
        # Initialize control image processor
        if hasattr(self.pipeline, "image_processor"):
            self.control_image_processor = self.pipeline.image_processor
        else:
            self.control_image_processor = T.Compose([
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ])
        
        # Store ControlNet configuration for reuse
        self.controlnet_config = None
        self.force_zeros_for_pooled_projection = False
        if self.use_controlnet and hasattr(self.pipeline, 'controlnet'):
            self.controlnet_config = getattr(self.pipeline.controlnet, "config", None)
            self.force_zeros_for_pooled_projection = (
                self.controlnet_config is not None and 
                getattr(self.controlnet_config, "force_zeros_for_pooled_projection", False)
            )
        
        # Set up models in evaluation mode
        self.pipeline.transformer.eval()
        self.pipeline.vae.eval()
        
        for encoder in [self.pipeline.text_encoder, 
                        self.pipeline.text_encoder_2, 
                        self.pipeline.text_encoder_3]:
            if encoder is not None:
                encoder.eval()
                
        if self.use_controlnet and hasattr(self.pipeline, 'controlnet'):
            self.pipeline.controlnet.eval()
        
        # New parameters
        self.tile_order = tile_order
        self.overlap_ratio = overlap_ratio
        self.overlap_size = patch_size // overlap_ratio
        self.overlap_size_latent = self.overlap_size // vae_scale_factor
        
        # Print scheduler info for debugging
        print(f"Using scheduler of type: {type(self.pipeline.scheduler).__name__}")
        self.scheduler_config = self.pipeline.scheduler.config
        
        # Prepare timesteps
        self.pipeline.scheduler.set_timesteps(self.sampling_steps, device=self.device)
        self.timesteps = self.pipeline.scheduler.timesteps
        self.sigmas = getattr(self.pipeline.scheduler, 'sigmas', None)

    @torch.no_grad()
    def encode_control_image(self, control_image, height, width):
        """
        Encode a control image to latent space for ControlNet conditioning.
        
        Args:
            control_image: PIL image for control
            height, width: Target dimensions
        
        Returns:
            Tensor of encoded control image
        """
        if not self.use_controlnet:
            return None
            
        # Use the class control image processor
        if isinstance(self.control_image_processor, T.Compose):
            image_tensor = self.control_image_processor(control_image).unsqueeze(0)
        else:
            # Using pipeline's VaeImageProcessor
            image_tensor = self.control_image_processor.preprocess(control_image, height=height, width=width)
        
        # Make sure image tensor has the correct shape
        if image_tensor.dim() != 4:
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            else:
                raise ValueError(f"Control image tensor has unexpected shape: {image_tensor.shape}")
        
        # Log shape for debugging
        # print(f"Control image tensor shape: {image_tensor.shape}")
        
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
        
        # Encode with VAE
        control_image_latents = self.pipeline.vae.encode(image_tensor).latent_dist.sample()
        vae_shift_factor = getattr(self.pipeline.vae.config, "shift_factor", 0)
        control_image_latents = (control_image_latents - vae_shift_factor) * self.pipeline.vae.config.scaling_factor
        
        return control_image_latents

    @torch.no_grad()
    def encode_prompt(self, prompt, do_classifier_free_guidance=True):
        """
        Encodes text prompt into embeddings.
        
        Args:
            prompt: Text prompt(s)
            do_classifier_free_guidance: Whether to create uncond/cond embedding pairs
        
        Returns:
            prompt_embeds, pooled_prompt_embeds
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Use the pipeline's encode_prompt
        (prompt_embeds, 
         negative_prompt_embeds, 
         pooled_prompt_embeds, 
         negative_pooled_prompt_embeds) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,  # Use the same prompt for all encoders
            prompt_3=prompt,  # Use the same prompt for all encoders
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=[""] * len(prompt) if do_classifier_free_guidance else None,
        )
        
        # Combine into the format expected by the transformer
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def decode(self, latents):
        """
        Decode latent representation to image using the VAE.
        
        Args:
            latents: Latent tensors to decode
        
        Returns:
            Decoded images as tensors in [0,1] range
        """
        # Scale latents back to VAE range
        latents = 1 / self.pipeline.vae.config.scaling_factor * latents
        
        # Add VAE shift factor if present
        if hasattr(self.pipeline.vae.config, "shift_factor"):
            latents = latents + self.pipeline.vae.config.shift_factor
            
        # Decode latents to images
        images = self.pipeline.vae.decode(latents).sample
        
        # Post-process
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return images

    def get_ordered_tile_coordinates(self, latent_size):
        """
        Generate coordinates for tiles in a specific order.
        
        Args:
            latent_size: Size of the latent image
            
        Returns:
            List of (i, j) coordinate tuples for tile centers
        """
        # Calculate effective patch size in latent space with overlap
        effective_patch_size = self.patch_size // self.vae_scale_factor - self.overlap_size_latent
        
        # Calculate how many tiles we need in each dimension
        num_tiles = max(1, (latent_size - self.overlap_size_latent) // effective_patch_size)
        
        # Create grid of tile centers
        coordinates = []
        
        # Calculate half patch size in latent space
        latent_patch_size = self.patch_size // self.vae_scale_factor
        half_patch = latent_patch_size // 2
        
        # Adjust the starting positions to ensure exact fit at left/top edges
        # Instead of starting at overlap_size_latent, start at half_patch
        for i in range(num_tiles):
            for j in range(num_tiles):
                # Calculate center of the tile in latent space with exact edge fit
                center_i = half_patch + i * effective_patch_size
                center_j = half_patch + j * effective_patch_size
                
                # Ensure the center is valid and not too close to the edge
                if center_i < latent_size - half_patch and center_j < latent_size - half_patch:
                    coordinates.append((center_i, center_j))
        
        # Left edge centers, evenly spaced
        left_edge = half_patch
        for i in range(1, num_tiles):  # Skip the corners which are handled below
            center_i = half_patch + i * effective_patch_size
            if center_i < latent_size - half_patch:
                coordinates.append((center_i, left_edge))
        
        # Top edge centers, evenly spaced
        top_edge = half_patch
        for j in range(1, num_tiles):  # Skip the corners which are handled below
            center_j = half_patch + j * effective_patch_size
            if center_j < latent_size - half_patch:
                coordinates.append((top_edge, center_j))
        
        # Right edge centers, evenly spaced
        right_edge = latent_size - half_patch
        for i in range(1, num_tiles):  # Skip the corners which are handled below
            center_i = half_patch + i * effective_patch_size
            if center_i < latent_size - half_patch:
                coordinates.append((center_i, right_edge))
        
        # Bottom edge centers, evenly spaced
        bottom_edge = latent_size - half_patch
        for j in range(1, num_tiles):  # Skip the corners which are handled below
            center_j = half_patch + j * effective_patch_size
            if center_j < latent_size - half_patch:
                coordinates.append((bottom_edge, center_j))
        
        # Add all four corners
        coordinates.append((left_edge, top_edge))      # Top-left corner
        coordinates.append((left_edge, right_edge))    # Top-right corner
        coordinates.append((bottom_edge, left_edge))   # Bottom-left corner
        coordinates.append((bottom_edge, right_edge))  # Bottom-right corner
        
        # Order the coordinates based on the chosen ordering
        if self.tile_order == "corner_to_center":
            # Sort by distance from center
            center = latent_size // 2
            coordinates.sort(key=lambda x: -((x[0] - center)**2 + (x[1] - center)**2))
        else:  # center_to_corner
            # Sort by distance to center
            center = latent_size // 2
            coordinates.sort(key=lambda x: ((x[0] - center)**2 + (x[1] - center)**2))
        
        return coordinates

    def get_crop(self, image, i, j, latent=True):
        """
        Extract a crop from an image at position (i,j).
        
        Args:
            image: Input tensor
            i, j: Center coordinates
            latent: Whether the input is in latent space
        
        Returns:
            Cropped tensor
        """
        if latent:
            p = self.patch_size // self.vae_scale_factor // 2
            return image[..., i-p:i+p, j-p:j+p]
        else:
            p = self.patch_size // 2
            return image[..., i*self.vae_scale_factor-p:i*self.vae_scale_factor+p, j*self.vae_scale_factor-p:j*self.vae_scale_factor+p]

    @torch.no_grad()
    def sample_one(self, x_t, prompt_embeds, pooled_prompt_embeds, control_image_latents=None, debug=False):
        """
        Sample one patch by fully denoising from noise to image.
        
        Args:
            x_t: Initial noise tensor
            prompt_embeds: Text embeddings
            pooled_prompt_embeds: Pooled text embeddings
            control_image_latents: Control image latents (if using ControlNet)
            debug: Whether to print debug information
            
        Returns:
            Denoised patch latent
        """
        # Start from the noisy latent
        latents = x_t.clone()
        
        if debug:
            print(f"Starting sampling with shape: {latents.shape}, device: {latents.device}, dtype: {latents.dtype}")
            print(f"Using guidance scale: {self.guidance_scale}")
            print(f"Prompt embedding shape: {prompt_embeds.shape}")
            if control_image_latents is not None:
                print(f"Control image latent shape: {control_image_latents.shape}")
        
        # Check if we're using classifier-free guidance
        do_classifier_free_guidance = self.guidance_scale > 1.0
        
        # If using classifier-free guidance, we need to duplicate latents to match the batch size of prompt_embeds
        if do_classifier_free_guidance:
            latents = torch.cat([latents] * 2, dim=0)
            
            # Also duplicate control image latents if they exist
            if control_image_latents is not None:
                control_image_latents = torch.cat([control_image_latents] * 2, dim=0)
                if debug:
                    print(f"Expanded control image latent shape: {control_image_latents.shape}")
            
            if debug:
                print(f"Expanded latents shape for guidance: {latents.shape}")
        
        # Reset timesteps for each patch generation - now using our reset_scheduler method
        # This ensures that the scheduler's internal state is reset properly for any scheduler type
        scheduler = self.reset_scheduler()
        timesteps = scheduler.timesteps
        
        if debug:
            print(f"Scheduler type: {type(scheduler).__name__}")
            print(f"Number of timesteps: {len(timesteps)}")
            print(f"First few timesteps: {timesteps[:3].cpu().tolist()}")
        
        # Process through all timesteps
        for i, t in enumerate(timesteps):
            try:
                if debug and i % 5 == 0:  # Only log every 5 steps to avoid too much output
                    print(f"Step {i}/{len(timesteps)}, timestep value: {t.item()}, latent stats: "
                          f"mean={latents.mean().item():.3f}, std={latents.std().item():.3f}")
                
                # Expand timestep to match batch size of latents
                timestep = t
                timestep_batch = timestep.expand(latents.shape[0])
                
                # Get ControlNet conditioning if applicable
                control_block_samples = None
                if self.use_controlnet and control_image_latents is not None:
                    # Use stored configuration for pooled projections
                    if self.force_zeros_for_pooled_projection:
                        controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
                    else:
                        controlnet_pooled_projections = pooled_prompt_embeds
                    
                    # ControlNet forward pass
                    try:
                        control_block_samples = self.pipeline.controlnet(
                            hidden_states=latents,
                            timestep=timestep_batch,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=controlnet_pooled_projections,
                            controlnet_cond=control_image_latents,
                            return_dict=False
                        )[0]
                        if debug and i == 0:
                            print(f"ControlNet produced blocks with shape: {control_block_samples[0].shape}")
                    except Exception as e:
                        print(f"ControlNet error at step {i}: {str(e)}")
                        if debug:
                            print(f"ControlNet inputs - hidden_states: {latents.shape}, timestep: {timestep_batch.shape}, "
                                  f"encoder_hidden_states: {prompt_embeds.shape}, "
                                  f"controlnet_cond: {control_image_latents.shape}")
                        raise
                
                # Predict the noise residual with the transformer
                try:
                    model_pred = self.pipeline.transformer(
                        hidden_states=latents,
                        timestep=timestep_batch,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        block_controlnet_hidden_states=control_block_samples,
                        return_dict=False,
                    )[0]
                    if debug and i == 0:
                        print(f"Transformer output shape: {model_pred.shape}")
                except Exception as e:
                    print(f"Transformer error at step {i}: {str(e)}")
                    if debug:
                        print(f"Transformer inputs - hidden_states: {latents.shape}, timestep: {timestep_batch.shape}")
                    raise
                
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
                    model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if debug and i == 0:
                        print(f"Applied guidance with scale {self.guidance_scale}")
                
                # Step with the scheduler - Using our fresh scheduler instance for each tile
                try:
                    # Use the freshly created scheduler instance
                    latents = scheduler.step(model_pred, t, latents).prev_sample
                    
                    if debug and i % 5 == 0:
                        print(f"Scheduler step complete, new latent stats: "
                              f"mean={latents.mean().item():.3f}, std={latents.std().item():.3f}")
                except Exception as e:
                    print(f"Scheduler error at step {i}: {str(e)}")
                    if debug:
                        print(f"Scheduler inputs - model_pred: {model_pred.shape}, "
                              f"timestep: {timestep.item()}, latents: {latents.shape}")
                        print(f"Scheduler info - type: {type(scheduler).__name__}")
                        if hasattr(scheduler, 'step_index'):
                            print(f"Scheduler step index: {scheduler.step_index}, timesteps length: {len(scheduler.timesteps)}")
                    raise
                    
            except Exception as e:
                print(f"Error during denoising step {i}: {str(e)}")
                if debug:
                    # Print stack trace for better debugging
                    import traceback
                    traceback.print_exc()
                raise
        
        # For classifier-free guidance, return only the conditional part of the output
        if do_classifier_free_guidance:
            final_output = latents[latents.shape[0]//2:]
            if debug:
                print(f"Returning conditional output with shape: {final_output.shape}")
            return final_output
        else:
            if debug:
                print(f"Returning output with shape: {latents.shape}")
            return latents

    def reset_scheduler(self):
        """
        Reset the scheduler to its initial state or create a new instance.
        Works with any scheduler type whether it has clone() or not.
        
        Returns:
            A reset scheduler instance
        """
        # Get the scheduler class
        scheduler_class = self.pipeline.scheduler.__class__
        
        # Create a new instance with the same config
        # Use from_config if available, otherwise use direct init
        if hasattr(scheduler_class, "from_config"):
            new_scheduler = scheduler_class.from_config(self.scheduler_config)
        else:
            # Fall back to direct initialization with config dict
            config_dict = vars(self.scheduler_config)
            # Filter out non-init params if needed
            init_params = {k: v for k, v in config_dict.items() 
                          if not k.startswith('_') and k != 'timesteps'}
            new_scheduler = scheduler_class(**init_params)
        
        # Set timesteps on the new scheduler
        new_scheduler.set_timesteps(self.sampling_steps, device=self.device)
        
        return new_scheduler

    def create_weight_map(self, size, overlap_size, is_edge_tile=None, debug=False):
        """
        Create a weight map for blending overlapping patches.
        
        Args:
            size: Size of the patch (H, W)
            overlap_size: Size of the overlap region
            is_edge_tile: Dict indicating which edges of this tile are at image boundaries
                        {'left': bool, 'right': bool, 'top': bool, 'bottom': bool}
            debug: Whether to print debug information
            
        Returns:
            Weight tensor for blending
        """
        if debug:
            print(f"Creating weight map with size {size} and overlap {overlap_size}")
        
        # Create a tensor of ones with the given size
        weight = torch.ones(size, device=self.device)
        
        # If is_edge_tile is not provided, default to assuming this is not an edge tile
        if is_edge_tile is None:
            is_edge_tile = {'left': False, 'right': False, 'top': False, 'bottom': False}
        
        # Apply linear blending in the overlap regions
        if overlap_size > 0:
            # Left edge - apply gradient UNLESS this tile is at the left edge of the full image
            if not is_edge_tile.get('left', False):
                ramp = torch.linspace(0, 1, overlap_size, device=self.device)
                weight[:, :overlap_size] *= ramp.view(1, -1)
            
            # Right edge - apply gradient UNLESS this tile is at the right edge of the full image
            if not is_edge_tile.get('right', False):
                ramp = torch.linspace(1, 0, overlap_size, device=self.device)
                weight[:, -overlap_size:] *= ramp.view(1, -1)
            
            # Top edge - apply gradient UNLESS this tile is at the top edge of the full image
            if not is_edge_tile.get('top', False):
                ramp = torch.linspace(0, 1, overlap_size, device=self.device)
                weight[:overlap_size, :] *= ramp.view(-1, 1)
            
            # Bottom edge - apply gradient UNLESS this tile is at the bottom edge of the full image
            if not is_edge_tile.get('bottom', False):
                ramp = torch.linspace(1, 0, overlap_size, device=self.device)
                weight[-overlap_size:, :] *= ramp.view(-1, 1)
            
            if debug:
                print(f"Weight map min: {weight.min().item()}, max: {weight.max().item()}")
                print(f"Edge status: {is_edge_tile}")
                print(f"Corners: TL={weight[0,0].item():.3f}, TR={weight[0,-1].item():.3f}, BL={weight[-1,0].item():.3f}, BR={weight[-1,-1].item():.3f}")
        
        return weight

    @torch.no_grad()
    def __call__(self, prompt, control_image=None, batch_size=1, use_region_prompt=True, use_hann_blending=False, debug=False):
        """
        Generate a large image using tiled diffusion.
        
        Args:
            prompt: Text prompt for generation
            control_image: Optional control image for ControlNet (overrides the one set in init)
            batch_size: Batch size for generation
            use_region_prompt: Whether to use region-specific prompts
            use_hann_blending: Whether to use Hann window blending for decoding (alternative to weight-based blending)
            debug: Whether to print debug information
        
        Returns:
            Tuple of (latent_tensor, decoded_image)
        """
        try:
            # Use provided control image or the one from init
            if control_image is not None:
                self.control_image = control_image
            
            # Get the shape of the output latent
            c = self.pipeline.transformer.config.in_channels
            latent_size = self.image_size // self.vae_scale_factor
            
            print(f"Output latent shape will be: [batch={batch_size}, channels={c}, height={latent_size}, width={latent_size}]")
            print(f"Final image size will be: {self.image_size}x{self.image_size}")
            print(f"Using tile order: {self.tile_order}, overlap size: {self.overlap_size} pixels")
            
            if debug and self.use_controlnet:
                if self.control_image is not None:
                    print(f"Control image: {type(self.control_image)}, size: {self.control_image.size if hasattr(self.control_image, 'size') else 'unknown'}")
                else:
                    print("Warning: ControlNet is enabled but no control image is provided")
            
            # Initialize latent canvas and weight map for blending
            latent_canvas = torch.zeros(batch_size, c, latent_size, latent_size, 
                                       device=self.device, dtype=self.dtype)
            weight_map = torch.zeros(batch_size, 1, latent_size, latent_size, 
                                    device=self.device, dtype=self.dtype)
            
            # Calculate patch size in latent space
            latent_patch_size = self.patch_size // self.vae_scale_factor
            half_patch = latent_patch_size // 2
            
            if debug:
                print(f"Patch size in pixel space: {self.patch_size}, in latent space: {latent_patch_size}")
                print(f"Effective patch overlap in latent space: {self.overlap_size_latent} pixels")
            
            # Encode global prompt once if region prompt is disabled
            global_prompt_embeds = None
            global_pooled_prompt_embeds = None
            if not use_region_prompt:
                try:
                    if debug:
                        print(f"Encoding global prompt: '{prompt[:50]}...' once for all tiles")
                    
                    global_prompt_embeds, global_pooled_prompt_embeds = self.encode_prompt(
                        prompt, 
                        do_classifier_free_guidance=(self.guidance_scale > 1.0)
                    )
                    
                    if debug:
                        print(f"Global prompt embed shapes: {global_prompt_embeds.shape}, {global_pooled_prompt_embeds.shape}")
                except Exception as e:
                    print(f"Error encoding global prompt: {str(e)}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    raise
            
            # Get ordered tile coordinates
            coordinates = self.get_ordered_tile_coordinates(latent_size)
            total_tiles = len(coordinates)
            
            if debug:
                print(f"Generated {total_tiles} tile coordinates in {self.tile_order} order")
                first_coords = coordinates[:min(3, len(coordinates))]
                last_coords = coordinates[-min(3, len(coordinates))]
                print(f"First few tile centers: {first_coords}")
                print(f"Last few tile centers: {last_coords}")
            
            print(f"Processing {total_tiles} tiles...")
            
            # Process each tile
            for idx, (i, j) in enumerate(coordinates):
                try:
                    # Generate and encode prompt for this region
                    current_prompt = prompt
                    if use_region_prompt and isinstance(prompt, str) and self.control_image is not None:
                        try:
                            region_prompt = self.generate_prompt_from_region(
                                self.control_image, i, j, latent_size)
                            # Combine with original prompt
                            current_prompt = f"{region_prompt}. {prompt}"
                            if debug and idx % 10 == 0:  # Log only occasionally to avoid spam
                                print(f"\nRegion prompt for tile ({i},{j}): '{region_prompt}'")
                        except Exception as e:
                            print(f"\nError generating region prompt for tile ({i},{j}): {str(e)}")
                            # Continue with original prompt in case of failure
                    
                    # Print progress with tile details
                    progress_info = f"[{idx+1}/{total_tiles}] Processing tile at ({i}, {j})"
                    if isinstance(current_prompt, str) and len(current_prompt) > 70:
                        prompt_preview = current_prompt[:70] + "..."
                    else:
                        prompt_preview = current_prompt
                    print(f"\r{progress_info} - Prompt: {prompt_preview}", end="", flush=True)
                    
                    # Get prompt embeddings
                    if use_region_prompt or global_prompt_embeds is None:
                        try:
                            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                                current_prompt, 
                                do_classifier_free_guidance=(self.guidance_scale > 1.0)
                            )
                        except Exception as e:
                            print(f"\nError encoding prompt for tile ({i},{j}): {str(e)}")
                            if debug:
                                print(f"Problematic prompt: '{current_prompt}'")
                            raise
                    else:
                        # Use global prompt embeddings
                        prompt_embeds, pooled_prompt_embeds = global_prompt_embeds, global_pooled_prompt_embeds
                    
                    # Get control image patch if using ControlNet
                    control_latent = None
                    if self.use_controlnet and self.control_image is not None:
                        try:
                            # Get full pixel coordinates for this patch
                            i_pixel, j_pixel = i * self.vae_scale_factor, j * self.vae_scale_factor
                            p_half = self.patch_size // 2
                            
                            # Calculate region from full control image
                            y_start = max(0, i_pixel - p_half)
                            y_end = min(self.image_size, i_pixel + p_half)
                            x_start = max(0, j_pixel - p_half)
                            x_end = min(self.image_size, j_pixel + p_half)
                            
                            if debug and idx % 10 == 0:  # Log only occasionally
                                print(f"\nExtracting control region: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}]")
                            
                            # Extract control image region
                            control_img = self.control_image.crop((x_start, y_start, x_end, y_end))
                            
                            # Resize if necessary to match patch size
                            if control_img.width != self.patch_size or control_img.height != self.patch_size:
                                control_img = control_img.resize((self.patch_size, self.patch_size))
                                
                            # Encode control image to latent
                            control_latent = self.encode_control_image(
                                control_img, self.patch_size, self.patch_size)
                            
                            if debug and idx == 0:  # Only log the first one
                                print(f"\nControl latent shape: {control_latent.shape}")
                        except Exception as e:
                            print(f"\nError processing control image for tile ({i},{j}): {str(e)}")
                            if debug:
                                import traceback
                                traceback.print_exc()
                    
                    # Initialize noise for this patch
                    patch_latent = torch.randn(
                        (batch_size, c, latent_patch_size, latent_patch_size),
                        device=self.device,
                        dtype=self.dtype
                    )
                    
                    # Sample the patch (full denoising)
                    denoised_patch = self.sample_one(
                        patch_latent, prompt_embeds, pooled_prompt_embeds, control_latent, 
                        debug=(debug and idx == 0)  # Only debug the first patch
                    )
                    
                    if debug and idx % 10 == 0:  # Log occasionally
                        print(f"\nDenoised patch stats: shape={denoised_patch.shape}, "
                              f"min={denoised_patch.min().item():.3f}, max={denoised_patch.max().item():.3f}, "
                              f"mean={denoised_patch.mean().item():.3f}, std={denoised_patch.std().item():.3f}")
                    
                    # Determine which edges of this tile are at the image boundaries
                    is_edge_tile = {
                        'left': j - half_patch <= 0,  # Left edge of tile touches left edge of image
                        'right': j + half_patch >= latent_size,  # Right edge of tile touches right edge of image
                        'top': i - half_patch <= 0,  # Top edge of tile touches top edge of image
                        'bottom': i + half_patch >= latent_size  # Bottom edge of tile touches bottom edge of image
                    }
                    
                    # Calculate patch weight for blending with edge information
                    patch_weight = self.create_weight_map(
                        (latent_patch_size, latent_patch_size),
                        self.overlap_size_latent,
                        is_edge_tile=is_edge_tile,  # Pass edge information
                        debug=(debug and idx == 0)  # Only debug the first weight map
                    ).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    # Calculate region coordinates for the latent canvas
                    y_start = max(0, i - half_patch)
                    y_end = min(latent_size, i + half_patch)
                    x_start = max(0, j - half_patch)
                    x_end = min(latent_size, j + half_patch)
                    
                    # Calculate corresponding region in the patch
                    p_y_start = max(0, half_patch - i)
                    p_y_end = latent_patch_size - max(0, (i + half_patch) - latent_size)
                    p_x_start = max(0, half_patch - j)
                    p_x_end = latent_patch_size - max(0, (j + half_patch) - latent_size)
                    
                    # Debugging for edge cases
                    if debug and (i > latent_size - half_patch - 5 or j > latent_size - half_patch - 5):
                        print(f"\nDEBUG - Edge tile at ({i},{j}):")
                        print(f"Canvas region: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
                        print(f"Patch region: y=[{p_y_start}:{p_y_end}], x=[{p_x_start}:{p_x_end}]")
                        print(f"Latent size: {latent_size}, Half patch: {half_patch}")
                        print(f"Canvas region dimensions: {y_end-y_start}x{x_end-x_start}")
                        print(f"Patch region dimensions: {p_y_end-p_y_start}x{p_x_end-p_x_start}")
                    
                    # Check if the regions have correct dimensions
                    if (y_end - y_start) != (p_y_end - p_y_start) or (x_end - x_start) != (p_x_end - p_x_start):
                        print(f"\nWARNING: Dimension mismatch for tile at ({i},{j})!")
                        print(f"Canvas region: {y_end-y_start}x{x_end-x_start}, Patch region: {p_y_end-p_y_start}x{p_x_end-p_x_start}")
                        # Skip this tile if dimensions don't match
                        continue
                    
                    # Get the patch region to use
                    patch_region = denoised_patch[..., p_y_start:p_y_end, p_x_start:p_x_end]
                    weight_region = patch_weight[..., p_y_start:p_y_end, p_x_start:p_x_end]
                    
                    # Update the canvas and weight map
                    # Check for NaNs in patch
                    if torch.isnan(patch_region).any():
                        print(f"\nWARNING: NaN values found in patch at ({i},{j}). Skipping this patch.")
                        continue
                    
                    latent_canvas[..., y_start:y_end, x_start:x_end] += patch_region * weight_region
                    weight_map[..., y_start:y_end, x_start:x_end] += weight_region
                    
                    # Periodic NaN check
                    if idx % 5 == 0:
                        if torch.isnan(latent_canvas).any():
                            print(f"\nWARNING: NaN values detected in canvas after tile {idx}")
                
                except Exception as e:
                    print(f"\nError processing tile at ({i},{j}): {str(e)}")
                    if debug:
                        print(f"Scheduler type: {type(self.pipeline.scheduler).__name__}")
                        import traceback
                        traceback.print_exc()
                    continue
            
            print("\nFinished processing all tiles. Blending final image...")
            
            # Check if we have usable data
            if torch.isnan(latent_canvas).any():
                print("WARNING: NaN values detected in the latent canvas before normalization")
                
                # Try to fix NaN values
                latent_canvas = torch.nan_to_num(latent_canvas, nan=0.0)
                print("Replaced NaN values with zeros in the canvas")
            
            # Normalize the latent canvas by the accumulated weights
            # Avoid division by zero or very small numbers more aggressively
            weight_map = torch.clamp(weight_map, min=0.5)  # More conservative minimum
            
            # Check if the weight map has any zeros or NaNs
            if torch.isnan(weight_map).any() or (weight_map == 0).any():
                print("WARNING: Weight map contains zeros or NaNs. Fixing...")
                weight_map = torch.nan_to_num(weight_map, nan=1.0)
                weight_map = torch.clamp(weight_map, min=0.5)
            
            # Normalize with careful checking
            final_latents = latent_canvas / weight_map
            
            # Check for NaNs after normalization
            if torch.isnan(final_latents).any():
                print("WARNING: NaN values detected after normalization. Attempting to recover...")
                final_latents = torch.nan_to_num(final_latents, nan=0.0)
                
                # If we still have NaNs, try a different approach
                if torch.isnan(final_latents).any():
                    print("Still have NaNs. Falling back to simple averaging...")
                    # Create a new weight map with ones only where we have valid values
                    valid_map = (~torch.isnan(latent_canvas)).float()
                    weight_sum = torch.sum(valid_map, dim=1, keepdim=True)
                    weight_sum = torch.clamp(weight_sum, min=1.0)
                    final_latents = torch.sum(torch.nan_to_num(latent_canvas, nan=0.0), dim=1, keepdim=True) / weight_sum
            
            if debug:
                nan_count = torch.isnan(final_latents).sum().item()
                print(f"Final latents stats: shape={final_latents.shape}, " 
                      f"min={final_latents.min().item() if not torch.isnan(final_latents).all() else 'nan'}, "
                      f"max={final_latents.max().item() if not torch.isnan(final_latents).all() else 'nan'}, "
                      f"mean={final_latents.mean().item() if not torch.isnan(final_latents).all() else 'nan'}, "
                      f"std={final_latents.std().item() if not torch.isnan(final_latents).all() else 'nan'}, "
                      f"NaN count={nan_count}")
                print(f"Weight map stats: min={weight_map.min().item()}, max={weight_map.max().item()}")
            
            # After generating final_latents, either use simple decoding or Hann window blending
            if use_hann_blending:
                print("Using Hann window blending for final decoding...")
                try:
                    final_image = self.hann_tile_overlap(final_latents)
                except Exception as e:
                    print(f"Error during Hann window blending: {str(e)}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    # Fall back to direct decoding
                    print("Falling back to direct decoding...")
                    use_hann_blending = False
            
            # Check if we still have NaNs in final_latents before trying to decode
            if torch.isnan(final_latents).any():
                print("WARNING: Still have NaNs in final latents. Generating random latents as fallback.")
                # Create new random latents as a last resort
                final_latents = torch.randn_like(final_latents)
            
            if not use_hann_blending:
                # Use direct decoding
                print("Decoding final image...")
                with torch.no_grad():
                    try:
                        # Scale latents back to VAE range
                        scaled_latents = 1 / self.pipeline.vae.config.scaling_factor * final_latents
                        
                        # Add VAE shift factor if present
                        if hasattr(self.pipeline.vae.config, "shift_factor"):
                            scaled_latents = scaled_latents + self.pipeline.vae.config.shift_factor
                            
                        # Final check for NaNs before decoding
                        if torch.isnan(scaled_latents).any():
                            print("WARNING: NaNs detected before VAE decoding, replacing with zeros")
                            scaled_latents = torch.nan_to_num(scaled_latents, nan=0.0)
                        
                        # Decode latents to images
                        final_image = self.pipeline.vae.decode(scaled_latents).sample
                        
                        # Check for NaNs in the decoded image
                        if torch.isnan(final_image).any():
                            print("WARNING: NaNs detected in decoded image, replacing with zeros")
                            final_image = torch.nan_to_num(final_image, nan=0.0)
                        
                        # Post-process to [0,1] range
                        final_image = (final_image / 2 + 0.5).clamp(0, 1)
                        
                        if debug:
                            nan_count = torch.isnan(final_image).sum().item()
                            print(f"Decoded image shape: {final_image.shape}, "
                                  f"min={final_image.min().item() if not torch.isnan(final_image).all() else 'nan'}, "
                                  f"max={final_image.max().item() if not torch.isnan(final_image).all() else 'nan'}, "
                                  f"NaN count={nan_count}")
                    except Exception as e:
                        print(f"Error during VAE decoding: {str(e)}")
                        if debug:
                            import traceback
                            traceback.print_exc()
                        raise
                
            print("Generation complete.")
            return final_latents, final_image
            
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty tensors to avoid complete failure
            return torch.zeros((batch_size, c, latent_size, latent_size)), torch.zeros((batch_size, 3, self.image_size, self.image_size))

    @staticmethod
    def latent_to_pixel_coords(i, j, p, vae_scale_factor):
        """
        Convert latent space coordinates to pixel space coordinates.
        
        Args:
            i, j: Center coordinates in latent space
            p: Patch size in latent space
            vae_scale_factor: VAE scaling factor (usually 8)
        
        Returns:
            tuple: (i_pixel, j_pixel, patch_size_pixel)
        """
        i_pixel = i * vae_scale_factor
        j_pixel = j * vae_scale_factor
        patch_size_pixel = p * vae_scale_factor
        return i_pixel, j_pixel, patch_size_pixel

    @staticmethod
    def rgb_to_class_mask(rgb_image):
        """
        Convert RGB control image back to class mask.
        
        Args:
            rgb_image: RGB numpy array [H, W, 3]
            
        Returns:
            numpy array with class labels
        """
        # Define color to class mapping (inverse of what's in color_masks.py)
        color_to_class = {
            (255, 255, 255): 0,  # White -> Unknown
            (255, 165, 0): 1,    # Orange -> Background/Artifact
            (0, 255, 255): 2,    # Cyan -> Inflammatory/Reactive
            (255, 0, 0): 3,      # Red -> Carcinoma
            (0, 128, 0): 4       # Green -> Normal Tissue
        }
        
        # Initialize output mask
        mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        
        # Convert each RGB color to class
        for color, class_idx in color_to_class.items():
            color_match = np.all(rgb_image == color, axis=2)
            mask[color_match] = class_idx
            
        return mask

    def generate_prompt_from_region(self, control_image, i, j, p):
        """
        Generate prompt from a region in the control image.
        
        Args:
            control_image: Full control image (PIL Image or numpy array)
            i, j: Center coordinates in latent space
            p: Patch size in latent space
        
        Returns:
            str: Generated prompt
        """
        # Convert control image to numpy if it's PIL
        if isinstance(control_image, Image.Image):
            control_array = np.array(control_image)
        else:
            control_array = control_image
            
        # Convert latent coordinates to pixel space
        i_pixel, j_pixel, patch_size_pixel = self.latent_to_pixel_coords(
            i, j, p, self.vae_scale_factor)
        
        # Extract region from control image
        h, w = control_array.shape[:2]
        i_start = max(0, i_pixel - patch_size_pixel//2)
        i_end = min(h, i_pixel + patch_size_pixel//2)
        j_start = max(0, j_pixel - patch_size_pixel//2)
        j_end = min(w, j_pixel + patch_size_pixel//2)
        
        region = control_array[i_start:i_end, j_start:j_end]
        
        # Convert RGB region to class mask
        mask = self.rgb_to_class_mask(region)
        
        # Calculate class percentages
        unique_labels, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        # Map class indices to names
        class_names = {
            0: "unknown tissue",
            1: "background artifact",
            2: "inflammatory reactive",
            3: "carcinoma",
            4: "normal tissue"
        }
        
        # Build class percentages list (similar to create_text_prompt.py)
        class_percentages = []
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_pixels) * 100
            if percentage > 1 and label in class_names:  # Skip classes with < 1% presence
                class_percentages.append((class_names[label], percentage))
        
        # Sort by percentage
        class_percentages.sort(key=lambda x: x[1])
        
        # Create prompt
        if not class_percentages:
            return "pathology image showing tissue sample"
            
        descriptions = ", ".join(
            f"{name} {percentage:.2f}%" 
            for name, percentage in class_percentages
        )
        
        prompt = f"pathology image: {descriptions}"
        
        return prompt

    def decoding_tiled_image(self, z, size):
        """
        Decode a large latent tensor using tiling.
        
        Args:
            z: Latent tensor to decode
            size: Target size of the output image
            
        Returns:
            Decoded image tensor
        """
        p = self.patch_size
        tile_size = p // self.vae_scale_factor
        
        # Tile operation
        tiled = tile(z, tile_size)
        
        # Decode tiles
        decoded_tiles = []
        n_tiles = len(tiled)
        print(f"\nDecoding {n_tiles} tiles...")
        for idx, t in enumerate(tiled):
            decoded = self.decode(t[None])
            decoded_tiles.append(decoded)
            print(f"\rProcessed tile {idx+1}/{n_tiles}", end="")
        print()  # New line after progress
        
        img = torch.cat(decoded_tiles, 0)
        
        # Untile
        return untile(img, size, p)

    def hann_tile_overlap(self, z):
        """
        Apply Hann window blending to avoid visible tile boundaries.
        
        Args:
            z: Latent tensor to blend
            
        Returns:
            Blended image tensor
        """
        assert z.shape[-1] % (self.patch_size // self.vae_scale_factor) == 0, \
            'Input must be divisible by patch size in latent space'
        
        windows = hann_window(self.patch_size)
        b, c, h, w = z.shape
        
        # Scale latents for VAE
        z_scaled = z.clone()
        if self.pipeline.vae.config.scaling_factor != 1.0:
            z_scaled = z_scaled * self.pipeline.vae.config.scaling_factor
        
        p = self.patch_size

        p_overlap = self.patch_size // self.vae_scale_factor // 2  
        
        try:
            # Calculate dimensions for slices
            full_size = (1, 3, h * self.vae_scale_factor, w * self.vae_scale_factor)
            vertical_size = (1, 3, h * self.vae_scale_factor, w * self.vae_scale_factor - p)
            horizontal_size = (1, 3, h * self.vae_scale_factor - p, w * self.vae_scale_factor)
            cross_size = (1, 3, h * self.vae_scale_factor - p, w * self.vae_scale_factor - p)
            
            # Full image decoding
            print("Decoding full image...")
            img = self.decoding_tiled_image(z_scaled, full_size)
            
            # Vertical slice
            print("Decoding vertical slice...")
            v_slice = z_scaled[..., p_overlap:-p_overlap]
            img_v = self.decoding_tiled_image(v_slice, vertical_size)
            
            # Horizontal slice
            print("Decoding horizontal slice...")
            h_slice = z_scaled[..., p_overlap:-p_overlap, :]
            img_h = self.decoding_tiled_image(h_slice, horizontal_size)
            
            # Cross slice
            print("Decoding cross slice...")
            cross_slice = z_scaled[..., p_overlap:-p_overlap, p_overlap:-p_overlap]
            img_cross = self.decoding_tiled_image(cross_slice, cross_size)
            
            # Window application
            print("Applying window blending...")
            b, c, h, w = img.shape
            
            # Creating repeat tensors
            repeat_v = windows['vertical'].repeat(b, c, h // p, w // p - 1).to(self.device)
            repeat_h = windows['horizontal'].repeat(b, c, h // p - 1, w // p).to(self.device)
            repeat_c = windows['center'].repeat(b, c, h // p - 1, w // p - 1).to(self.device)
            
            # Final blending
            img[..., p//2:-p//2] = img[..., p//2:-p//2] * (1 - repeat_v) + img_v * repeat_v
            img[..., p//2:-p//2, :] = img[..., p//2:-p//2, :] * (1 - repeat_h) + img_h * repeat_h
            img[..., p//2:-p//2, p//2:-p//2] = img[..., p//2:-p//2, p//2:-p//2] * (1 - repeat_c) + img_cross * repeat_c
            
            return img
            
        except Exception as e:
            print(f"Error during window blending: {str(e)}")
            raise


# Helper functions

def tile(x, p):
    """
    Split a tensor into tiles of size p×p.
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        p: Tile size
    
    Returns:
        Tensor of tiles
    """
    B, C, H, W = x.shape
    x_tiled = x.unfold(2, p, p).unfold(3, p, p)
    x_tiled = x_tiled.reshape(B, C, -1, p, p)
    x_tiled = x_tiled.permute(0, 2, 1, 3, 4).contiguous()
    x_tiled = x_tiled.reshape(-1, C, p, p)
    return x_tiled

def untile(x_tiled, original_shape, p):
    """
    Reconstruct a tensor from tiles.
    
    Args:
        x_tiled: Tensor of tiles
        original_shape: Target shape [B, C, H, W]
        p: Tile size
    
    Returns:
        Reconstructed tensor
    """
    B, C, H, W = original_shape
    
    # Calculate number of tiles in each dimension
    H_p = H // p
    W_p = W // p
    
    # Reshape and permute to reconstruct
    x_tiled = x_tiled.reshape(B, H_p * W_p, C, p, p)
    x_tiled = x_tiled.permute(0, 2, 1, 3, 4)
    x_tiled = x_tiled.reshape(B, C, H_p, W_p, p, p)
    x_untiled = x_tiled.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
    
    return x_untiled

def save_tensor_as_png(tensor, filename):
    """
    Save a tensor as a PNG image.
    
    Args:
        tensor: Image tensor [C, H, W]
        filename: Output filename
    """
    # Convert the tensor to a NumPy array
    tensor_np = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Check for NaNs
    if np.isnan(tensor_np).any():
        print(f"WARNING: Image contains NaN values, replacing with zeros")
        tensor_np = np.nan_to_num(tensor_np, nan=0.0)
    
    # Check min and max
    print(f"Image range before conversion: min={np.min(tensor_np):.4f}, max={np.max(tensor_np):.4f}")
    
    # Normalize if needed
    if np.max(tensor_np) > 1.0 or np.min(tensor_np) < 0.0:
        print("Normalizing image to [0,1] range")
        if np.max(tensor_np) != np.min(tensor_np):
            tensor_np = (tensor_np - np.min(tensor_np)) / (np.max(tensor_np) - np.min(tensor_np))
        else:
            tensor_np = np.zeros_like(tensor_np)

    # Convert the NumPy array to an 8-bit unsigned integer array
    tensor_uint8 = (tensor_np * 255).astype(np.uint8)
    
    # Create a PIL image from the 8-bit unsigned integer array
    image = Image.fromarray(tensor_uint8)
    
    # Check image dimensions
    print(f"Final image dimensions: {image.width}x{image.height}")
    
    # Save the image as a PNG file
    image.save(filename, format='PNG', optimize=True, compress_level=9)
    
def corners(subwindows: list):
    """
    Create a composite window from corner subwindows.
    
    Args:
        subwindows: List of 4 corner windows [upleft, upright, downright, downleft]
    
    Returns:
        Combined window
    """
    (w_upleft, w_upright, w_downright, w_downleft) = subwindows
    window = torch.ones_like(w_upleft)
    size = window.shape[0]
    window[:size//2, :size//2] = w_upleft[:size//2, :size//2]
    window[:size//2, size//2:] = w_upright[:size//2, size//2:]
    window[size//2:, size//2:] = w_downright[size//2:, size//2:]
    window[size//2:, :size//2] = w_downleft[:size//2, :size//2]
    return window

def hann_window(size=512):
    """
    Create Hann windows for blending overlapping tiles.
    
    Args:
        size: Size of the window
    
    Returns:
        Dictionary of different window types
    """
    i = torch.arange(size, dtype=torch.float)
    w = 0.5 * (1 - torch.cos(2 * torch.pi * i / (size-1)))
    window_center = w[:, None] * w
    window_up = torch.where(torch.arange(size)[:, None] < size//2, w, w[:, None] * w)
    window_down = torch.where(torch.arange(size)[:, None] > size//2, w, w[:, None] * w)
    window_right = torch.where(torch.arange(size) > size//2, w[:, None], w[:, None] * w)
    window_left = torch.where(torch.arange(size) < size//2, w[:, None], w[:, None] * w)
    window_upleft = corners([torch.ones((size, size)), window_up, window_center, window_left])
    window_upright = corners([window_up, torch.ones((size, size)), window_right, window_center])
    window_downright = corners([window_center, window_right, torch.ones((size, size)), window_down])
    window_downleft = corners([window_left, window_center, window_down, torch.ones((size, size))])
    
    window_rightright = corners([torch.ones((size, size)), window_up, window_down, torch.ones((size, size))])
    window_leftleft = corners([window_up, torch.ones((size, size)), torch.ones((size, size)), window_down])
    window_upup = corners([window_left, window_right, torch.ones((size, size)), torch.ones((size, size))])
    window_downdown = corners([torch.ones((size, size)), torch.ones((size, size)), window_right, window_left])

    window_vertical = corners([window_up, window_up, window_down, window_down])
    window_horizontal = corners([window_left, window_right, window_right, window_left])
    
    return {'up-left': window_upleft, 'up': window_up, 'up-right': window_upright, 
            'left': window_left, 'center': window_center,  'right': window_right, 
            'down-left': window_downleft, 'down': window_down, 'down-right': window_downright,
           'up-up': window_upup, 'down-down': window_downdown, 
            'left-left': window_leftleft, 'right-right': window_rightright,
           'vertical': window_vertical, 'horizontal': window_horizontal}