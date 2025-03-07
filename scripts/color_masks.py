import os
from pathlib import Path
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm

# Define paths
input_dir = Path.home() / 'ControlNet/training/drsk/source'
output_dir = Path.home() / 'ControlNet/training/drsk/source-colored'

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Define color mapping (class to RGB)
colors = ['#FFFFFF', '#FFA500', '#00FFFF', '#FF0000', '#008000']
# Convert hex colors to RGB tuples
color_map = {
    i: tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
    for i, color in enumerate(colors)
}

def process_image(filename):
    """Process a single image file."""
    try:
        # Read the mask image
        img_path = input_dir / filename
        mask = np.array(Image.open(img_path))
        
        # Create an RGB image with the same size as mask
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # Fill colors based on mask values
        for class_idx, color in color_map.items():
            colored[mask == class_idx] = color
            
        # Save the colored image
        output_path = output_dir / filename
        Image.fromarray(colored).save(output_path)
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

def main():
    # Get list of all PNG files
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    total_files = len(image_files)
    
    print(f"Found {total_files} images to process")
    
    # Using ThreadPoolExecutor for parallel processing
    # Number of threads is set to number of CPU cores * 2 for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        # Create a progress bar
        list(tqdm(
            executor.map(process_image, image_files),
            total=total_files,
            desc="Processing images",
            unit="image"
        ))

if __name__ == "__main__":
    main()