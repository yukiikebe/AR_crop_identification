import os
import rasterio
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt  # Add this import
import yaml
import csv
from datetime import datetime

dataset = "/data/datasets/satellite/raw_arkansas_2023/2023_all/"
output  = "./data/Arkansas/output/"
output  = "./data/Arkansas/output/12months/"

sub_region = "17_10"
date = "2023-07-24"
# Loading the YAML file
with open('./data/Arkansas/cdl.yaml', 'r') as file:
    data = yaml.safe_load(file)

crop_types = data['crop_type']
non_crop_types = data['non_crop_type']
num2class = data['num2class']


def convert_to_RGB(blue_band_path, green_band_path, red_band_path):
    # Open the TIF files
    if not (os.path.isfile(blue_band_path) and os.path.isfile(green_band_path) and os.path.isfile(red_band_path)):
        print("One of the bands is missing")
        return

    with rasterio.open(blue_band_path) as blue_band:
        blue = blue_band.read(1)
    with rasterio.open(green_band_path) as green_band:
        green = green_band.read(1)
    with rasterio.open(red_band_path) as red_band:
        red = red_band.read(1)

    # Stack the bands
    stacked_data = np.stack((red, green, blue), axis=-1)
    return stacked_data


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

def analyze_tile_distribution(dataset, crop_types, non_crop_types):
    # visual_distribution(dataset, output)
    distribution = check_tile_distribution(dataset, crop_types, non_crop_types)

    print("Tile Distribution:", distribution )

    for tile, counts in distribution.items():
        print(f"Tile {tile} - Crop: {counts['crop']}, Non-Crop: {counts['non-crop']}")

    output_csv_path = "tile_distribution.csv"
    save_distribution_as_csv(distribution, output_csv_path)


    # Step 1: Extract crop percentages
    crop_percentages = [v['crop'] for v in distribution.values()]

    # Step 2: Round the crop percentages to the nearest integer
    rounded_crop_percentages = [round(p) for p in crop_percentages]

    # Step 3: Create a histogram with 100 bins
    plt.figure(figsize=(10, 6))
    plt.hist(rounded_crop_percentages, bins=100, color='skyblue', edgecolor='black')
    plt.xlabel('Crop Percentage')
    plt.ylabel('Frequency')
    plt.title('Distribution of Crop Percentages')
    plt.tight_layout()

    # Save the histogram as an image file
    plt.savefig('./data/Arkansas/output/crop_distribution_histogram.png')





def visual_crop(dataset,sub_region):


    
    # plt.imsave(output_path, stacked_data)
    cdl_image_path = os.path.join(dataset, sub_region, "cdl.tif")
    blue_band_path = os.path.join(dataset, sub_region,date,"TCI_B_"+date+".tif")
    green_band_path = os.path.join(dataset, sub_region,date, "TCI_G_"+date+".tif")
    red_band_path = os.path.join(dataset, sub_region,date, "TCI_R_"+date+".tif")
    output_path = os.path.join(output, sub_region)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    output_rgb = os.path.join(output, sub_region, "rgb.png")
    rgb_image = convert_to_RGB(blue_band_path, green_band_path, red_band_path)
    plt.imsave(output_rgb, rgb_image)

    # Load the CDL image
    with rasterio.open(cdl_image_path) as src:
        cdl_image = src.read(1)
    
    # Create a mask for crop labels
    crop_values = set(map(int, crop_types.keys()))


    for crop_value in crop_values:
        label_mask = (cdl_image == crop_value)
        
        # Create an RGBA image for the current crop label
        label_rgba_image = np.zeros((cdl_image.shape[0], cdl_image.shape[1], 4), dtype=np.uint8)
        label_rgba_image[..., :3] = rgb_image  # Copy RGB channels
        label_rgba_image[..., 3] = np.where(label_mask, 255, int(0.2 * 255))  # Set alpha channel
        
        # Save the RGBA image for the current crop label
        crop_name = crop_types[str(crop_value)]
        crop_name = crop_name.replace(" ", "_").replace("/", "_")
        output_label_rgba = os.path.join(output, sub_region, f"rgba_label_{crop_name}.png")


        plt.imsave(output_label_rgba, label_rgba_image)

# analyze_tile_distribution(dataset, crop_types, non_crop_types)

# visual_crop(dataset,sub_region)


def visual_crop_12_months(dataset, sub_region, output_dir):
    def get_12_dates_per_month(dates):
        # Filter out 'cdl.tif'
        dates = [date for date in dates if date != 'cdl.tif']
        
        # Group dates by month
        dates_by_month = defaultdict(list)
        for date in dates:
            try:
                parsed_date = datetime.strptime(date, '%Y-%m-%d')
                month_key = parsed_date.strftime('%Y-%m')
                dates_by_month[month_key].append(date)
            except ValueError:
                continue  # Skip invalid dates
        
        # Select one date per month
        selected_dates = []
        for month, month_dates in dates_by_month.items():
            selected_dates.append(month_dates[0])  # Select the first date of the month
        
        return selected_dates
    

    path_output_crop = os.path.join(output_dir, "Crop")
    if not os.path.exists(path_output_crop):
        os.makedirs(path_output_crop)
    
    path_output_non_crop = os.path.join(output_dir, "Non_Crop")
    if not os.path.exists(path_output_non_crop):
        os.makedirs(path_output_non_crop)
    
    path_output_rgb = os.path.join(output_dir, "RGB")
    if not os.path.exists(path_output_rgb):
        os.makedirs(path_output_rgb)


    print("Visualizing crop types for 12 months")
    print("dataset ", dataset)
    dataset_region = os.path.join(dataset, sub_region)
    list_date = os.listdir(dataset_region)
    print("list_date ", list_date)
    list_12_dates = get_12_dates_per_month(list_date)

    print("list_12_dates ", list_12_dates)

    crop_values = set(map(int, crop_types.keys()))
    non_crop_values = set(map(int, non_crop_types.keys()))




    # Load the CDL image
    cdl_image_path = os.path.join(dataset_region, "cdl.tif")
    with rasterio.open(cdl_image_path) as src:
        cdl_image = src.read(1)
    
    classes_id = np.unique(cdl_image)


    # print("cdl_image ", np.unique(cdl_image), len(np.unique(cdl_image)))
    # for idx in classes_id:
    #     crop_name = num2class[str(idx)]
    #     crop_name = crop_name.replace(" ", "_").replace("/", "_")
    #     if idx in crop_values:
    #         print("crop_name ", crop_name)
    #     else:
    #         print("non_crop_name ", crop_name)
    # exit()
    for date in list_12_dates:
        print("date ", date)
        blue_band_path = os.path.join(dataset_region, date, "TCI_B_" + date + ".tif")
        green_band_path = os.path.join(dataset_region, date, "TCI_G_" + date + ".tif")
        red_band_path = os.path.join(dataset_region, date, "TCI_R_" + date + ".tif")
        rgb_image = convert_to_RGB(blue_band_path, green_band_path, red_band_path)

        output_rgb = os.path.join(path_output_rgb, str(date)+".png")
        plt.imsave(output_rgb, rgb_image)

        for idx in classes_id:
            crop_name = num2class[str(idx)]
            crop_name = crop_name.replace(" ", "_").replace("/", "_")
            print("crop_name ", crop_name)
            if idx in crop_values:
                path_output = os.path.join(path_output_crop, crop_name)
                if not os.path.exists(path_output):
                    os.makedirs(path_output)
            else:
                path_output = os.path.join(path_output_non_crop, crop_name)
                if not os.path.exists(path_output):
                    os.makedirs(path_output)
            
            output_path = os.path.join(path_output, f"{date}.png")
            label_mask = (cdl_image == idx)
            label_rgba_image = np.zeros((cdl_image.shape[0], cdl_image.shape[1], 4), dtype=np.uint8)
            label_rgba_image[..., :3] = rgb_image
            label_rgba_image[..., 3] = np.where(label_mask, 255, int(0.2 * 255))
            plt.imsave(output_path, label_rgba_image)



    

visual_crop_12_months(dataset, sub_region, output)

# python data/Arkansas/data_analysis.py >/home/vuonghn/research/code/Agriculture/DeepSatModels/data/Arkansas/distribution.txt

# crop_name  Background
# crop_name  Corn
# crop_name  Cotton
# crop_name  Rice
# crop_name  Sorghum
# crop_name  Soybeans
# crop_name  Peanuts
# crop_name  Winter_Wheat
# crop_name  Dbl_Crop_WinWht_Soybeans
# crop_name  Millet
# crop_name  Other_Hay_Non_Alfalfa
# crop_name  Sweet_Potatoes
# crop_name  Herbs
# non_crop_name  Sod_Grass_Seed
# non_crop_name  Fallow_Idle_Cropland
# crop_name  Pecans
# non_crop_name  Aquaculture
# non_crop_name  Open_Water
# non_crop_name  Developed_Open_Space
# non_crop_name  Developed_Low_Intensity
# non_crop_name  Developed_Med_Intensity
# non_crop_name  Developed_High_Intensity
# non_crop_name  Barren
# non_crop_name  Deciduous_Forest
# non_crop_name  Evergreen_Forest
# non_crop_name  Mixed_Forest
# non_crop_name  Shrubland
# non_crop_name  Grassland_Pasture
# non_crop_name  Woody_Wetlands
# non_crop_name  Herbaceous_Wetlands
# crop_name  Triticale
# crop_name  Dbl_Crop_Soybeans_Oats