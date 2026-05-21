import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob

def adjust_brightness(image, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_blur(image, ksize=(5, 5)):
    return cv2.GaussianBlur(image, ksize, 0)

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    
    # Salt noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

def augment_and_save_images(input_dir, output_dir):
    for split in ['train', 'val']:
        input_path = Path(input_dir) / split
        output_path = Path(output_dir) / split
        os.makedirs(output_path, exist_ok=True)
        
        for category in os.listdir(input_path):
            category_path = input_path / category
            output_category_path = output_path / category
            os.makedirs(output_category_path, exist_ok=True)
            
            image_paths = glob(str(category_path / "*.png"))
            for image_path in tqdm(image_paths, desc=f"Processing {split}/{category}"):
                image = cv2.imread(image_path)
                filename = Path(image_path).stem
                
                # Apply transformations
                bright_image = adjust_brightness(image)
                blur_image = add_blur(image)
                noisy_image = add_salt_and_pepper_noise(image)
                
                # Save augmented images
                cv2.imwrite(str(output_category_path / f"{filename}_bright.png"), bright_image)
                cv2.imwrite(str(output_category_path / f"{filename}_blur.png"), blur_image)
                cv2.imwrite(str(output_category_path / f"{filename}_noise.png"), noisy_image)

if __name__ == "__main__":
    input_directory = "../Orthophotos_patches_png_scale16_split_ratio/LR"
    output_directory = "../Orthophotos_patches_png_scale16_split_ratio/LR"
    augment_and_save_images(input_directory, output_directory)
    print("Data augmentation completed!")
