import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_colored_mask(image_path, output_path=None):
    """
    Load a mask image and map specific pixel values to designated colors:
    0 -> white (#FFFFFF)
    63 -> green (#008000)
    191 -> red (#FF0000)
    255 -> orange (#FFA500)
    
    Saves the colored mask as a PNG with RGB mode and uint8 data type.
    """
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return
    
    try:
        # Load the image
        img = Image.open(image_path)
        print(f"Successfully loaded image: {image_path}")
        
        # Basic image properties of original image
        print(f"Original image format: {img.format}")
        print(f"Original image mode: {img.mode}")
        print(f"Original image size (width × height): {img.size}")
        
        # Convert to numpy array
        mask_array = np.array(img)
        
        # Print unique values
        unique_values = np.unique(mask_array)
        print(f"Unique values in mask: {unique_values}")
        
        # Create a new RGB image with the same dimensions
        width, height = img.size
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Map each pixel value to its designated color (RGB format)
        colored_mask[mask_array == 0] = [255, 255, 255]     # white
        colored_mask[mask_array == 63] = [0, 128, 0]        # green
        colored_mask[mask_array == 191] = [255, 0, 0]       # red
        colored_mask[mask_array == 255] = [255, 165, 0]     # orange
        
        # If there are any other values not in our mapping, color them black
        for value in unique_values:
            if value not in [0, 63, 191, 255]:
                print(f"Warning: Found unexpected value {value} in mask, coloring it black.")
                colored_mask[mask_array == value] = [0, 0, 0]  # black
                
        # Set output path if not provided
        if output_path is None:
            output_path = os.path.splitext(image_path)[0] + "_colored.png"
        
        # Convert to PIL image and save
        colored_img = Image.fromarray(colored_mask)
        colored_img.save(output_path, format="PNG")
        
        print(f"Colored mask saved to: {output_path}")
        
        # Verify the saved image has the requested characteristics
        saved_img = Image.open(output_path)
        saved_array = np.array(saved_img)
        
        print("\nVerifying saved colored mask:")
        print(f"Image format: {saved_img.format}")
        print(f"Image mode: {saved_img.mode}")
        print(f"Array shape: {saved_array.shape}")
        print(f"Data type: {saved_array.dtype}")
        print(f"Min value: {np.min(saved_array)}")
        print(f"Max value: {np.max(saved_array)}")
        
        # Display the original and colored masks
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(mask_array, cmap='gray')
        plt.colorbar(label='Original Mask Value')
        plt.title("Original Mask")
        
        plt.subplot(1, 2, 2)
        plt.imshow(colored_mask)
        plt.title("Colored Mask")
        
        plt.tight_layout()
        plt.show()
        
        return colored_mask
        
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

if __name__ == "__main__":
    # Path to the image
    image_path = "large-content/squamous_mask_example_5classes.png"
    output_path = "large-content/squamous_mask_colored.png"
    create_colored_mask(image_path, output_path)