#!/usr/bin/env python
import torch
import argparse
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.utils import load_image
from diffusers import SD3ControlNetModel

# Create a debug pipeline that extends the original pipeline with logging
class DebugStableDiffusion3ControlNetPipeline(StableDiffusion3ControlNetPipeline):
    def __call__(self, *args, **kwargs):
        # Add debug flag
        self.debug = kwargs.pop("debug", False)
        return super().__call__(*args, **kwargs)
    
    def encode_prompt(self, *args, **kwargs):
        # Call original method
        outputs = super().encode_prompt(*args, **kwargs)
        
        # Debug output
        if self.debug:
            print("\n=== Text Embeddings Debug ===")
            if len(outputs) == 4:
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = outputs
            else:
                prompt_embeds, pooled_prompt_embeds = outputs
                negative_prompt_embeds = negative_pooled_prompt_embeds = None
                
            print(f"Prompt embeddings shape: {prompt_embeds.shape}")
            print(f"Pooled prompt embeddings shape: {pooled_prompt_embeds.shape}")
            if negative_prompt_embeds is not None:
                print(f"Negative prompt embeddings shape: {negative_prompt_embeds.shape}")
            
        return outputs
    
    def prepare_image(self, *args, **kwargs):
        # Call original method
        image = super().prepare_image(*args, **kwargs)
        
        # Debug output
        if self.debug:
            print("\n=== Control Image Debug ===")
            print(f"Prepared control image shape: {image.shape}")
            print(f"Prepared control image dtype: {image.dtype}")
            print(f"Prepared control image range: {image.min().item():.4f} to {image.max().item():.4f}")
            
        return image

    def prepare_latents(self, *args, **kwargs):
        # Call original method
        latents = super().prepare_latents(*args, **kwargs)
        
        # Debug output
        if self.debug:
            print("\n=== Initial Latents Debug ===")
            print(f"Initial latents shape: {latents.shape}")
            print(f"Initial latents dtype: {latents.dtype}")
            
        return latents

# Add debugging to ControlNet forward pass
def debug_controlnet_forward(self, *args, **kwargs):
    original_forward = self.controlnet.forward
    
    def wrapped_forward(*args, **kwargs):
        print("\n=== ControlNet Forward Pass Debug ===")
        hidden_states = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        controlnet_cond = kwargs.get("controlnet_cond", args[4] if len(args) > 4 else None)
        
        print(f"hidden_states shape: {hidden_states.shape}")
        if controlnet_cond is not None:
            print(f"controlnet_cond shape: {controlnet_cond.shape}")
            print(f"controlnet_cond dtype: {controlnet_cond.dtype}")
            print(f"controlnet_cond range: {controlnet_cond.min().item():.4f} to {controlnet_cond.max().item():.4f}")
        
        try:
            result = original_forward(*args, **kwargs)
            print("✅ ControlNet forward pass successful!")
            return result
        except Exception as e:
            print(f"❌ ControlNet forward pass failed with error: {e}")
            raise e
    
    return wrapped_forward

def main(args):
    # Load models and create pipeline
    print(f"Loading pipeline from {args.model_path} with controlnet from {args.controlnet_path}")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set dtype based on args
    dtype = torch.float16 if args.fp16 else torch.float32
    print(f"Using dtype: {dtype}")
    
    # Load controlnet first
    print("Loading controlnet...")
    controlnet = SD3ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=dtype
    )
    
    # Load pipeline with loaded controlnet
    pipeline = DebugStableDiffusion3ControlNetPipeline.from_pretrained(
        args.model_path,
        controlnet=controlnet,  # Pass the loaded model instead of path
        torch_dtype=dtype,
        variant=args.variant,
    )
    
    # Add debugging to controlnet
    original_forward = pipeline.controlnet.forward
    pipeline.controlnet.forward = debug_controlnet_forward(pipeline, pipeline.controlnet.forward)
    
    # Print model structure information
    print("\n=== Model Architecture Debug ===")
    
    # Check ControlNet conditioning channels
    print("\n=== ControlNet Structure ===")
    controlnet = pipeline.controlnet
    
    # Print key layers with channel info
    for name, module in controlnet.named_modules():
        if any(x in name for x in ["conv_in", "controlnet_cond_embedding", "pos_embed"]):
            if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                print(f"Layer {name}: in_channels={module.in_channels}, out_channels={module.out_channels}")
    
    # Load control image
    control_image = load_image(args.control_image)
    print(f"Loaded control image: {control_image.size}")
    
    # Enable attention slicing for memory efficiency
    pipeline.enable_attention_slicing()
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Run inference with debug
    print(f"\nRunning inference with prompt: '{args.prompt}'")
    
    try:
        # Add tracing of the VAE encoding step
        original_vae_encode = pipeline.vae.encode
        
        def debug_vae_encode(*args, **kwargs):
            print("\n=== VAE Encode Debug ===")
            input_tensor = args[0]
            print(f"VAE encode input shape: {input_tensor.shape}")
            print(f"VAE encode input dtype: {input_tensor.dtype}")
            print(f"VAE encode input range: {input_tensor.min().item():.4f} to {input_tensor.max().item():.4f}")
            
            result = original_vae_encode(*args, **kwargs)
            
            # Debug the latent distribution
            print(f"VAE latent distribution mean shape: {result.latent_dist.mean.shape}")
            latent_sample = result.latent_dist.sample()
            print(f"VAE latent sample shape: {latent_sample.shape}")
            print(f"VAE latent sample range: {latent_sample.min().item():.4f} to {latent_sample.max().item():.4f}")
            print(f"VAE scaling factor: {pipeline.vae.config.scaling_factor}")
            
            return result
        
        # Replace with debug version temporarily
        pipeline.vae.encode = debug_vae_encode
        
        # Set seed for reproducibility
        generator = None
        if args.seed is not None:
            print(f"Using seed: {args.seed}")
            generator = torch.Generator(device=device).manual_seed(args.seed)
        
        # Run inference
        with torch.inference_mode():
            output = pipeline(
                prompt=args.prompt,
                control_image=control_image,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                debug=True,  # Enable debug prints
                generator=generator,  # Add generator here
            )
        
        # Restore original methods
        pipeline.vae.encode = original_vae_encode
        pipeline.controlnet.forward = original_forward
        
        # Save the result
        if output.images is not None:
            output_path = args.output_path
            output.images[0].save(output_path)
            print(f"✅ Successfully generated image at: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        
        # Add specific debugging for the error
        if "Expected input batch_size" in str(e):
            print("\n=== Batch Size Error Analysis ===")
            print("This often happens when classifier-free guidance is enabled but batch sizes don't match.")
            print("Check that controlnet_cond is properly duplicated for classifier-free guidance.")
        
        if "Expected hidden_states dimensions" in str(e) or "expected input" in str(e).lower():
            print("\n=== Shape Mismatch Analysis ===")
            print("This often happens when controlnet_cond doesn't match the expected shape.")
            print("The controlnet might expect different dimensions or channel count than provided.")
            
            # Try to inspect controlnet's expected shapes
            for name, module in pipeline.controlnet.named_modules():
                if "controlnet_cond_embedding" in name and hasattr(module, "in_channels"):
                    print(f"ControlNet conditioning module expects {module.in_channels} input channels")
                    
            print("\nPossible solutions:")
            print("1. Check if controlnet was trained with a different conditioning type (RGB, grayscale, depth, etc.)")
            print("2. Ensure control image is properly preprocessed to match training data")
            print("3. Try using a different ControlNet model")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple SD3 ControlNet debugging")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="stabilityai/stable-diffusion-3-medium",
        help="Path to SD3 model"
    )
    parser.add_argument(
        "--controlnet_path", 
        type=str, 
        required=True,
        help="Path to ControlNet checkpoint"
    )
    parser.add_argument(
        "--control_image", 
        type=str, 
        required=True, 
        help="Path to control image"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="a photo of an astronaut riding a horse on mars", 
        help="Prompt for generation"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="debug_output.png", 
        help="Output image path"
    )
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument("--variant", type=str, default=None, help="Model variant (e.g., fp16)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")  # Add seed argument
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
