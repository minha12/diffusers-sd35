import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm

# add ../ to sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.prompt_augmenter import augment_prompt

from scripts.labels import labels
from scripts.labels_shorten import labels_shorten

# Reverse the dictionary for label to name lookup
label_to_name = {v: k for k, v in labels.items()}

def clean_class_name(class_name):
    """Clean up class names by removing special characters and converting to lowercase."""
    # Replace underscores and commas with spaces
    cleaned = class_name.replace('_', ' ').replace(',', ' ')
    # Convert to lowercase
    cleaned = cleaned.lower()
    # Remove multiple spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

def create_prompt(class_percentages, use_short=False):
    """Create prompt string with either full or shortened class names."""
    class_descriptions = ", ".join([f"{class_name} {percentage:.2f}%" 
                                  for class_name, percentage in class_percentages])
    prompt = f"pathology image: {class_descriptions}"
    
    # Check if prompt is too short (less than 25 words)
    if len(prompt.split()) < 25:
        prompt = augment_prompt(prompt)
    
    return prompt

def process_mask(mask_path):
    # Read the mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"\nError: Could not load mask at {mask_path}")
        return None

    # Get unique labels and their counts in the mask
    unique_labels, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    # Check if mask only contains background and tissue unknown
    if set(unique_labels).issubset({0, 1}):
        return None
    
    # Check if mask is empty (more than 98% is class 0)
    empty_mask_threshold = 0.98
    if 0 in unique_labels:
        zero_idx = np.where(unique_labels == 0)[0][0]
        zero_percentage = counts[zero_idx] / total_pixels
        if zero_percentage > empty_mask_threshold:
            return None
    
    # Calculate percentages and create prompt
    class_percentages = []
    for label, count in zip(unique_labels, counts):
        # Skip tissue_unknown (label 0)
        if label in label_to_name:
            percentage = (count / total_pixels) * 100
            if percentage > 1:
                clean_name = clean_class_name(label_to_name[label])
                class_percentages.append((clean_name, percentage))

    # Sort by percentage in ascending order (smallest first)
    class_percentages.sort(key=lambda x: x[1])
    
    # Create prompt with full names first
    prompt = create_prompt(class_percentages)
    
    # If prompt is too long, recreate with shortened names
    if len(prompt.split()) > 55:
        class_percentages_short = []
        for label, count in zip(unique_labels, counts):
            if label in labels_shorten:
                percentage = (count / total_pixels) * 100
                if percentage > 1:
                    class_percentages_short.append((labels_shorten[label], percentage))
        
        class_percentages_short.sort(key=lambda x: x[1])
        prompt = create_prompt(class_percentages_short)
    
    return prompt

def main(use_augmentation=False):
    # Path to the masks directory
    mask_dir = Path("../pathology-datasets/DRSK/full_dataset/masks")
    output_file = Path("mask_prompts.json")

    # Initialize counters for summary
    total_files = 0
    processed_files = 0
    augmented_prompts = 0

    # Process all mask files
    mask_files = list(mask_dir.glob("*.jpg"))
    total_files = len(mask_files)
    
    # Open file for writing
    with open(output_file, 'w') as f:
        for mask_path in tqdm(mask_files, desc="Processing masks", unit="file"):
            prompt = process_mask(mask_path)
            # Only include results with non-empty prompts
            if prompt and prompt != "pathology image: ":
                # Use the mask file's stem name for source and target
                file_stem = mask_path.stem
                source_name = f"{file_stem}_mask.png"
                target_name = f"{file_stem}.jpg"
                
                # Augment prompt if specified and track augmentations
                original_length = len(prompt.split())
                prompt = augment_prompt(prompt, use_augmentation)
                if len(prompt.split()) > original_length:
                    augmented_prompts += 1
                
                # Write result directly to file
                result = {
                    "source": source_name,
                    "target": target_name,
                    "prompt": prompt
                }
                f.write(f"{json.dumps(result)}\n")
                processed_files += 1

    # Print detailed summary to stderr
    print("\nProcessing Summary:", file=sys.stderr)
    print(f"Total files processed: {total_files}", file=sys.stderr)
    print(f"Valid prompts generated: {processed_files}", file=sys.stderr)
    print(f"Filtered out: {total_files - processed_files}", file=sys.stderr)
    if use_augmentation:
        print(f"Prompts augmented: {augmented_prompts}", file=sys.stderr)
    print(f"\nResults saved to {output_file}", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', action='store_true', help='Enable prompt augmentation')
    args = parser.parse_args()
    main(use_augmentation=args.augment)
