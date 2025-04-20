import json
import shutil
import os
from pathlib import Path

# Define paths
SOURCE_DATASET_PATH = Path("~/datasets/drsk").expanduser()
TARGET_DATASET_PATH = Path("~/datasets/test").expanduser()
SAMPLES_TO_TAKE = 64

# Ensure target directories exist
os.makedirs(TARGET_DATASET_PATH / "source", exist_ok=True)
os.makedirs(TARGET_DATASET_PATH / "target", exist_ok=True)

# Read the original prompt.json
with open(SOURCE_DATASET_PATH / "prompt.json", 'r') as f:
    lines = f.readlines()

# Parse the JSON lines
samples = []
for line in lines:
    if line.strip():
        samples.append(json.loads(line))

# Take the first SAMPLES_TO_TAKE samples or all if fewer
selected_samples = samples[:SAMPLES_TO_TAKE]

print(f"Selected {len(selected_samples)} samples out of {len(samples)} total samples")

# Create a new prompt.json with selected samples
new_prompts = []
for sample in selected_samples:
    source = sample["source"]
    target = sample["target"]
    prompt = sample["prompt"]
    
    # Get the filenames
    source_file = Path(source).name
    target_file = Path(target).name
    
    # Copy the files
    shutil.copy2(SOURCE_DATASET_PATH / source, TARGET_DATASET_PATH / "source" / source_file)
    shutil.copy2(SOURCE_DATASET_PATH / target, TARGET_DATASET_PATH / "target" / target_file)
    
    # Update the path in the sample to be relative
    sample["source"] = f"source/{source_file}"
    sample["target"] = f"target/{target_file}"
    
    # Add to new prompts
    new_prompts.append(sample)
    
    print(f"Copied {source_file} and {target_file}")

# Write the new prompt.json
with open(TARGET_DATASET_PATH / "prompt.json", 'w') as f:
    for sample in new_prompts:
        f.write(json.dumps(sample) + "\n")

print(f"\nSuccessfully created test dataset with {len(new_prompts)} samples at {TARGET_DATASET_PATH}")
print(f"New prompt.json created at {TARGET_DATASET_PATH / 'prompt.json'}")
