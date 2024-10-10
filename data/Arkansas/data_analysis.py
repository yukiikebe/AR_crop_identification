import os
import rasterio
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt  # Add this import
import yaml
import csv


dataset = "/data/datasets/satellite/raw_arkansas_2023/2023_all/"
output  = "./data/Arkansas/output/2023.png"

# Loading the YAML file
with open('./data/Arkansas/cdl.yaml', 'r') as file:
    data = yaml.safe_load(file)

crop_types = data['crop_type']
non_crop_types = data['non_crop_type']
num2class = data['num2class']

def visual_distribution(dataset, output):
    tiles = os.listdir(dataset)

    # Initialize a dictionary to store the counts of each unique value
    pixel_counts = defaultdict(int)

    for tile in tiles:
        date = os.listdir(os.path.join(dataset, tile))[0]
        cdl_image_path = os.path.join(dataset, tile, "cdl.tif")
        with rasterio.open(cdl_image_path) as src:
            cdl_image = src.read(1)  # Read the first band
        unique, counts = np.unique(cdl_image, return_counts=True)
        for value, count in zip(unique, counts):
            pixel_counts[value] += count

    # Map the counts to class names
    class_counts = {num2class.get(str(value), "Unknown"): count for value, count in pixel_counts.items()}

    # Sort the class counts in decreasing order
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))


    # Calculate the total number of pixels
    total_pixels = sum(sorted_class_counts.values())

    # Print the sorted class counts with distance between class name, number of pixels, and percentage
    max_length = max(len(crop) for crop in sorted_class_counts.keys()) + 5
    max_i_length = len(str(len(sorted_class_counts))) + 5  # Get the length of the largest index
    print(f"{'ID':<{max_i_length}} {'CROP_TYPE':<{max_length}} {'N_PIXEL':<15} {'PERCENTAGE'}")
    print("-" * (max_i_length + max_length + 25))
    for i, (crop, count) in enumerate(sorted_class_counts.items()):
        percentage = (count / total_pixels) * 100
        print(f"{i:<{max_i_length}} {crop:<{max_length}} {count:<15} {percentage:.7f} %")


    # Plot the distribution
    plt.figure(figsize=(15, 10))
    plt.bar(sorted_class_counts.keys(), sorted_class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Pixels')
    plt.title('Pixel Distribution by Class')
    plt.xticks(rotation=90)

    # Adjust layout to ensure labels are not cut off
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(output)

    # Optionally, you can close the plot to free up memory
    plt.close()



def check_tile_distribution(dataset_path, crop_types, non_crop_types):
    """
    Check the distribution of crop and non-crop pixels for each tile based on crop_type and non_crop_type.

    Args:
    dataset_path (str): Path to the dataset directory.
    crop_types (dict): Dictionary containing crop types.
    non_crop_types (dict): Dictionary containing non-crop types.

    Returns:
    dict: A dictionary where keys are tile indices and values are dictionaries with 'crop' and 'non-crop' counts.
    """
    distribution = {}
    crop_values = set(map(int, crop_types.keys()))
    non_crop_values = set(map(int, non_crop_types.keys()))
    
    tiles = os.listdir(dataset_path)

    for i, tile in enumerate(tiles):
        tile_path = os.path.join(dataset_path, tile)
        date = os.listdir(tile_path)[0]
        cdl_image_path = os.path.join(tile_path, "cdl.tif")
        
        with rasterio.open(cdl_image_path) as src:
            cdl_image = src.read(1)  # Read the first band
        
        crop_count = np.sum(np.isin(cdl_image, list(crop_values)))
        non_crop_count = np.sum(np.isin(cdl_image, list(non_crop_values)))
        
        total_pixels = cdl_image.size
        crop_count = (crop_count / total_pixels) * 100
        non_crop_count = (non_crop_count / total_pixels) * 100
        distribution[tile] = {
            'crop': crop_count,
            'non-crop': non_crop_count
        }
    
    return distribution


def save_distribution_as_csv(distribution, output_path):
    """
    Save the distribution of crop and non-crop pixels for each tile as a CSV file.

    Args:
    distribution (dict): A dictionary where keys are tile indices and values are dictionaries with 'crop' and 'non-crop' percentages.
    output_path (str): Path to save the output CSV file.
    """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['tile', 'crop', 'non-crop'])
        
        for tile, percentages in distribution.items():
            writer.writerow([tile, percentages['crop'], percentages['non-crop']])

# visual_distribution(dataset, output)
distribution = check_tile_distribution(dataset, crop_types, non_crop_types)

for tile, counts in distribution.items():
    print(f"Tile {tile} - Crop: {counts['crop']}, Non-Crop: {counts['non-crop']}")

output_csv_path = "tile_distribution.csv"
save_distribution_as_csv(distribution, output_csv_path)