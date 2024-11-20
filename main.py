import os
import numpy as np
from PIL import Image

def load_images_from_directory(directory):
    """
    Helper function to load handwritten images and their labels.

    Args:
        directory (str): Path to the directory containing image files.

    Returns:
        images (np.ndarray): Numpy array of image data (num_images, 28, 28).
        labels (np.ndarray): Numpy array of labels corresponding to the images.
    """
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Parse the label from the filename (e.g., '0-3-1.png' -> label = 0)
            label = int(filename.split('-')[0])
            
            # Load the image
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Resize to 28x28 if necessary (assuming MNIST compatibility)
            image = image.resize((28, 28))
            
            # Normalize pixel values (0 to 1)
            image_array = np.array(image) / 255.0
            
            # Append to list
            images.append(image_array)
            labels.append(label)
    
    # Convert lists to numpy arrays
    images = np.array(images).reshape(-1, 28, 28)
    labels = np.array(labels)
    
    return images, labels

if __name__ == "__main__":
    # Directory containing your handwritten images
    directory_path = input("Enter the directory path containing the images: ").strip()
    
    try:
        # Load images and labels
        images, labels = load_images_from_directory(directory_path)
        
        # Print results
        print(f"\nLoaded {len(labels)} images.")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
    except Exception as e:
        print(f"Error: {e}")
