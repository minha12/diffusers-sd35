import sys
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

class SD3RandomDiffusion:
    """
    Implementation of RandomDiffusion for SD3.5 architecture.
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
        
        # Prepare timesteps
        self.pipeline.scheduler.set_timesteps(self.sampling_steps, device=self.device)
        self.timesteps = self.pipeline.scheduler.timesteps
        self.sigmas = self.pipeline.scheduler.sigmas
        self._step_index = 0  # Add explicit step index tracking

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

    def get_value_coordinates(self, tensor):
        """
        Find coordinates with minimum value in a tensor.
        Used for selecting the next patch to process.
        
        Args:
            tensor: Input tensor to search
        
        Returns:
            Tensor of coordinates with minimum value
        """
        value_indices = torch.nonzero(tensor == tensor.min(), as_tuple=False)
        random_indices = value_indices[torch.randperm(value_indices.size(0))]
        return random_indices

    def random_crop(self, image, i, j, latent=True):
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
    def sample_one(self, x_t, x_stack, control_image, times, prompt_embeds, pooled_prompt_embeds, control_image_latents=None):
        """
        Sample one patch by denoising from current state.
        """
        img = x_t.clone()
        uniques = torch.unique(times)
        
        vmin = uniques[0]
        if len(uniques) > 1:
            for unique in uniques[1:]:
                to_change = torch.where(times == unique, 1, 0)
                x_t = x_stack[:, 0] * to_change + x_t * (to_change == 0)
        
        # Map negative timestep to valid index
        raw_t_index = int(vmin.item())
        t_index = max(0, min(raw_t_index, len(self.timesteps) - 1))  # Clamp between 0 and max index
        
        timestep = self.timesteps[t_index]
        timestep_batch = timestep.expand(x_t.shape[0])
        
        # Get ControlNet conditioning if applicable
        control_block_samples = None
        if self.use_controlnet and control_image is not None:
            # Use stored configuration for pooled projections
            if self.force_zeros_for_pooled_projection:
                controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
            else:
                controlnet_pooled_projections = pooled_prompt_embeds
            
            if control_image_latents is None:
                # Encode the control image patch only if not provided
                control_image_latents = self.encode_control_image(
                    Image.fromarray(control_image.squeeze(0).permute(1, 2, 0).cpu().numpy()),
                    height=control_image.shape[2],
                    width=control_image.shape[3]
                )
            
            # ControlNet forward pass using provided or encoded control_image_latents
            control_block_samples = self.pipeline.controlnet(
                hidden_states=x_t,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=controlnet_pooled_projections,
                controlnet_cond=control_image_latents,
                return_dict=False
            )[0]
        
        # Predict the noise residual with the transformer
        model_pred = self.pipeline.transformer(
            hidden_states=x_t,
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
            return_dict=False,
        )[0]
        
        # Perform guidance
        if self.guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Handle scheduler step
        if t_index >= len(self.sigmas) - 1:
            # For the last step, just return the prediction
            denoised = model_pred
        else:
            # Create a copy of the scheduler to avoid modifying global state
            scheduler = self.pipeline.scheduler
            
            # Manually set the step index for this iteration
            scheduler._step_index = t_index  # Use internal attribute
            
            # Perform the step
            try:
                step_result = scheduler.step(model_pred, timestep, x_t)
                denoised = step_result.prev_sample
            except Exception as e:
                print(f"Scheduler step failed: {e}")
                # Fallback: use simple denoising
                denoised = model_pred
        
        # Handle the blending between timesteps
        time_mask = torch.where(times == vmin, 1, 0).to(self.device)
        return denoised * time_mask + img * (time_mask == 0)

    def decoding_tiled_image(self, z, size):
        """
        Decode a large latent tensor using tiling.
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

    @torch.no_grad()
    def __call__(self, prompt, control_image=None, batch_size=1, use_region_prompt=True):
        """
        Generate a large image using tiled diffusion.
        
        Args:
            prompt: Text prompt for generation
            control_image: Optional control image for ControlNet (overrides the one set in init)
        
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
            
            # Initialize tracking tensors
            print("Initializing generation process...")
            img_stack = torch.randn(batch_size, 2, c, latent_size, latent_size).to(device=self.device, dtype=self.dtype)
            times = torch.zeros((1, 1, latent_size, latent_size)).int().to(self.device) - 1  # Start at -1
            img0 = torch.randn(batch_size, c, latent_size, latent_size).to(self.device, dtype=self.dtype)
            
            latent_overlap = self.patch_size // self.vae_scale_factor // 2
            # print(f"Patch size in latent space: {p}x{p}")
            
            # Encode global prompt once if region prompt is disabled
            global_prompt_embeds = None
            global_pooled_prompt_embeds = None
            if not use_region_prompt:
                global_prompt_embeds, global_pooled_prompt_embeds = self.encode_prompt(
                    prompt, 
                    do_classifier_free_guidance=(self.guidance_scale > 1.0)
                )

            # Main sampling loop
            print("Starting tiled sampling...")
            target_step = self.sampling_steps - 1  # Last timestep index
            
            while times.float().mean() < target_step:
                sys.stdout.flush()
                random_indices = self.get_value_coordinates(times[0, 0])[0]
                i, j = torch.clamp(random_indices, latent_overlap, latent_size-latent_overlap).tolist()
                
                # Generate and encode prompt for this region
                if use_region_prompt and isinstance(prompt, str) and self.control_image is not None:
                    region_prompt = self.generate_prompt_from_region(
                        self.control_image, i, j, latent_size)
                    # Combine with original prompt
                    current_prompt = f"{region_prompt}. {prompt}"
                    prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                        current_prompt, 
                        do_classifier_free_guidance=(self.guidance_scale > 1.0)
                    )
                else:
                    # Use global prompt embeddings if region prompt is off
                    prompt_embeds, pooled_prompt_embeds = (
                        global_prompt_embeds, global_pooled_prompt_embeds
                    )

                # Calculate progress...
                progress = (times.float().mean() + 1) * 100 / (target_step + 1)
                print(f"\r Generation {progress:.2f}%, "
                      f"indices=[{(i-latent_overlap)*self.vae_scale_factor:04d}:{(i+latent_overlap)*self.vae_scale_factor:04d},"
                      f"{(j-latent_overlap)*self.vae_scale_factor:04d}:{(j+latent_overlap)*self.vae_scale_factor:04d}], "
                      f"prompt: {current_prompt[:70]}...", 
                      end="")
                
                # Crop latents and times
                sub_img = self.random_crop(img0, i, j)
                sub_img_stack = self.random_crop(img_stack, i, j)
                sub_time = self.random_crop(times, i, j)
                
                # Get control image patch if using ControlNet
                submask = None
                if self.use_controlnet and self.control_image is not None:
                    # Crop the control image directly (not in latent space)
                    # Make sure to scale the crop size based on the latent dimensions
                    p_pixel = self.patch_size  # Full pixel size for the patch
                    submask = self.random_crop(
                        torch.tensor(np.array(self.control_image)).permute(2, 0, 1).unsqueeze(0).to(self.device), 
                        i, j,
                        latent=False
                    )
                
                # Sample if we haven't reached the final timestep
                if sub_time.float().mean() != target_step:
                    sub_img = self.sample_one(
                        sub_img, sub_img_stack, submask, 
                        sub_time, prompt_embeds, pooled_prompt_embeds)

                    mask_changed = torch.where(sub_time == sub_time.min(), 1, 0).to(self.device)

                    # Update tracking tensors
                    img0[..., i-latent_overlap:i+latent_overlap, j-latent_overlap:j+latent_overlap] = sub_img
                    img_stack[:, 1, :, i-latent_overlap:i+latent_overlap, j-latent_overlap:j+latent_overlap] = sub_img * mask_changed + \
                                                           sub_img_stack[:, 1] * (mask_changed == 0)
                    times[..., i-latent_overlap:i+latent_overlap, j-latent_overlap:j+latent_overlap] = torch.where(
                        sub_time == sub_time.min(), sub_time+1, sub_time)
                                                          
                    if torch.all(times == target_step):
                        img_stack[:, 0] = img_stack[:, 1]
            
            # Decode final latent with Hann window blending
            print("\nDecoding final image...")
            final_image = self.hann_tile_overlap(img0)
                
            return img0, final_image
            
        except Exception as e:
            print(f"\nError during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty tensors to avoid complete failure
            return torch.zeros((1, c, latent_size, latent_size)), torch.zeros((1, 3, self.image_size, self.image_size))

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
    H_p = H // p
    W_p = W // p
    x_tiled = x_tiled.reshape(B, (H//p * W//p), C, p, p).permute(0, 2, 1, 3, 4).reshape(B, C, H//p, W//p, p, p)
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

    # Convert the NumPy array to an 8-bit unsigned integer array
    tensor_uint8 = (tensor_np * 255).astype(np.uint8)

    # Create a PIL image from the 8-bit unsigned integer array
    image = Image.fromarray(tensor_uint8)

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
    window[size//2:, :size//2] = w_downleft[size//2:, :size//2]
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