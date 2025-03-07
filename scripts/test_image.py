import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_image(image_path):
    """
    Load and analyze an image, printing key information about it.
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return
    
    # Load the image
    try:
        img = Image.open(image_path)
        print(f"Successfully loaded image: {image_path}")
        
        # Basic image properties
        print(f"Image format: {img.format}")
        print(f"Image mode: {img.mode}")
        print(f"Image size (width × height): {img.size}")
        
        # Convert to numpy array for analysis
        img_array = np.array(img)
        
        # Analyze value ranges
        print(f"Array shape: {img_array.shape}")
        print(f"Data type: {img_array.dtype}")
        print(f"Min value: {np.min(img_array)}")
        print(f"Max value: {np.max(img_array)}")
        print(f"Mean value: {np.mean(img_array):.2f}")
        print(f"Standard deviation: {np.std(img_array):.2f}")
        
        # Count unique values (useful for masks)
        unique_values, counts = np.unique(img_array, return_counts=True)
        print(f"Unique values: {unique_values}")
        print(f"Value counts:")
        for val, count in zip(unique_values, counts):
            print(f"  {val}: {count} pixels ({count/(img_array.size)*100:.2f}%)")
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(img_array, cmap='viridis')
        plt.colorbar(label='Pixel Value')
        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.show()
        
    except Exception as e:
        print(f"Error processing the image: {e}")

if __name__ == "__main__":
    # Path to the image
    image_path = "validation_images/control_image_1.png"
    analyze_image(image_path)