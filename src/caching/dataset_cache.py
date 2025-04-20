#!/usr/bin/env python
import os
import functools
import logging
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import CLIPTokenizer, T5TokenizerFast
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from diffusers.utils import check_min_version
from src.utils.parse_args import parse_args
from src.utils.make_train_dataset import make_train_dataset
from src.utils.utils import encode_prompt, import_model_class_from_model_name_or_path, load_text_encoders

# Will error if the minimal version of diffusers is not installed
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

def compute_text_embeddings(batch, text_encoders, tokenizers, max_sequence_length, device=None):
    with torch.no_grad():
        prompt = batch["prompts"]
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length, device
        )
        
        # Check for invalid values before moving to CPU
        if torch.isinf(prompt_embeds).any() or torch.isnan(prompt_embeds).any():
            logger.error(f"NaN or Inf detected in prompt_embeds for batch starting with: {prompt[0][:50]}...")
            
        if torch.isinf(pooled_prompt_embeds).any() or torch.isnan(pooled_prompt_embeds).any():
            logger.error(f"NaN or Inf detected in pooled_prompt_embeds for batch starting with: {prompt[0][:50]}...")
        
        # Convert to CPU to save memory during caching
        prompt_embeds = prompt_embeds.cpu()
        pooled_prompt_embeds = pooled_prompt_embeds.cpu()
        
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

def get_stable_cache_key(args):
    from datasets.fingerprint import Hasher
    """Only include args that affect embeddings"""
    cache_relevant_args = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "max_sequence_length": args.max_sequence_length,
        "revision": args.revision,
        "variant": args.variant,
        # Add only args that affect embedding computation
    }
    # Creates same fingerprint for same embedding-relevant args
    return Hasher.hash(cache_relevant_args)

def main():
    args = parse_args()
    
    # Basic setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Configure NCCL timeouts and error handling
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # Use blocking mode for NCCL operations
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Enable async error handling
    os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minute timeout (in seconds)
    os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10485760"  # Enable NCCL tracing (10MB buffer)
    
    # Setup the accelerator for multi-GPU processing
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        log_with=None,
        project_dir=args.output_dir,
    )
    
    if accelerator.is_main_process:
        logger.info(f"Using {torch.cuda.device_count()} GPU(s) across {accelerator.num_processes} processes")
        logger.info(f"Current process (rank {accelerator.process_index}) device: {accelerator.device}")
    
    # Set seed for reproducibility
    from accelerate.utils import set_seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Make sure output directory exists
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.dataset_cache_dir:
            os.makedirs(args.dataset_cache_dir, exist_ok=True)
    
    # Set up tokenizers and encoders
    logger.info("Loading tokenizers and encoders...")
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )
    
    # Import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    
    # Load the text encoders
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
    )
    
    # Determine the weight dtype based on mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move models to device and dtype
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare models with accelerator
    text_encoder_one, text_encoder_two, text_encoder_three = accelerator.prepare(
        text_encoder_one, text_encoder_two, text_encoder_three
    )
    
    # Create dataset with error handling
    logger.info("Creating dataset...")
    try:
        with accelerator.main_process_first():
            train_dataset, preprocess_train = make_train_dataset(args, tokenizer_one, tokenizer_two, tokenizer_three, accelerator, logger)
            
        # Verify dataset creation was successful
        if train_dataset is None or len(train_dataset) == 0:
            raise RuntimeError("Dataset creation failed - empty or None dataset returned")
            
        logger.info(f"Successfully created dataset with {len(train_dataset)} examples")
        
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        # Clean shutdown of distributed process group
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        raise
    
    # Setup for text embedding computation
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    
    # Get the cache key
    fingerprint = get_stable_cache_key(args)
    logger.info(f"Using fingerprint: {fingerprint}")
    logger.info(f"Dataset size before processing: {len(train_dataset)}")
    
    # Split dataset across GPUs
    if accelerator.num_processes > 1:
        # Calculate shard size
        per_device_batch_size = max(1, args.dataset_preprocess_batch_size // accelerator.num_processes)
        shard_size = len(train_dataset) // accelerator.num_processes
        start_idx = accelerator.process_index * shard_size
        end_idx = start_idx + shard_size if accelerator.process_index < accelerator.num_processes - 1 else len(train_dataset)
        
        # Log shard information
        logger.info(f"Process {accelerator.process_index}: Processing shard {start_idx}:{end_idx} ({end_idx-start_idx} examples)")
        
        # Create a shard for this process
        process_train_dataset = train_dataset.select(range(start_idx, end_idx))
    else:
        # No sharding needed for single process
        process_train_dataset = train_dataset
        per_device_batch_size = args.dataset_preprocess_batch_size
    
    # Define process-specific cache file path
    cache_file = None
    final_cache_file = None
    if args.dataset_cache_dir:
        # Create a unique cache file for each process
        cache_file = os.path.join(args.dataset_cache_dir, f"cache_{fingerprint}_process_{accelerator.process_index}.arrow")
        final_cache_file = os.path.join(args.dataset_cache_dir, f"cache_{fingerprint}")
    
    # Create partial function for text embedding computation
    compute_embeddings_fn = functools.partial(
        compute_text_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        max_sequence_length=args.max_sequence_length,
        device=accelerator.device
    )
    
    # # print if the dataset having correct column names
    # logger.info(f"Dataset columns before renaming: {process_train_dataset.column_names}")
    # # First map just renames the columns permanently
    # process_train_dataset = process_train_dataset.map(
    #     preprocess_train,
    #     batched=True,
    #     batch_size=64, 
    #     num_proc=16
    # )

    # # print if the dataset having correct column names
    # logger.info(f"Dataset columns after renaming: {process_train_dataset.column_names}")
    
    # Process the dataset for this process shard
    process_train_dataset = process_train_dataset.map(
        compute_embeddings_fn,
        batched=True,
        batch_size=per_device_batch_size,
        load_from_cache_file=False,  # Force computation
        cache_file_name=cache_file,
        new_fingerprint=f"{fingerprint}_process_{accelerator.process_index}",
        desc=f"Computing text embeddings (process {accelerator.process_index})",
        num_proc=1  # Each GPU process handles its own shard
    )
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Clean up the accelerator's distributed environment since we don't need it anymore
    # for the merging phase - this will free up GPU resources
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # Only the main process merges the shards and cleans up
    if accelerator.is_main_process and args.dataset_cache_dir:
        logger.info("All processes finished embedding computation")
        logger.info("Merging cache files from all processes...")
        
        # Logic to merge the shards - this can now run without accelerator coordination
        from datasets import concatenate_datasets, Dataset
        import glob
        
        # Find all shard files using glob pattern matching
        shard_pattern = os.path.join(args.dataset_cache_dir, f"cache_{fingerprint}_process_*.arrow")
        shard_files = glob.glob(shard_pattern)
        
        logger.info(f"Found {len(shard_files)} shard files to merge")
        
        # Load all available shards
        all_shards = []
        for shard_file in shard_files:
            try:
                logger.info(f"Loading shard from {shard_file}")
                shard_dataset = Dataset.from_file(shard_file)
                all_shards.append(shard_dataset)
            except Exception as e:
                logger.warning(f"Failed to load shard {shard_file}: {e}")
        
        if all_shards:
            # Concatenate all shards
            merged_dataset = concatenate_datasets(all_shards)
            
            # Save the merged dataset
            merged_dataset.save_to_disk(final_cache_file)
            logger.info(f"Merged dataset saved to {final_cache_file}")
            
            # Clean up individual shard files
            for shard_file in shard_files:
                try:
                    os.remove(shard_file)
                    logger.info(f"Removed shard file {shard_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove shard file {shard_file}: {e}")
            
            # Write the cache path to a predictable file
            path_file = os.path.join(args.dataset_cache_dir, "latest_cache_path.txt")
            with open(path_file, "w") as f:
                f.write(final_cache_file)
            
            logger.info(f"Final cache saved to: {final_cache_file}")
            logger.info(f"Dataset size after processing: {len(merged_dataset)} examples")
        else:
            logger.error("No valid shards found to merge!")

    logger.info("Dataset preprocessing complete!")

if __name__ == "__main__":
    main()