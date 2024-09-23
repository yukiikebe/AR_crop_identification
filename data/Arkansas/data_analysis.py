num2class = {
    "1": "Corn",
    "2": "Cotton",
    "3": "Rice",
    "4": "Sorghum",
    "5": "Soybeans",
    "6": "Sunflower",
    "10": "Peanuts",
    "11": "Tobacco",
    "12": "Sweet Corn",
    "13": "Pop or Orn Corn",
    "14": "Mint",
    "21": "Barley",
    "22": "Durum Wheat",
    "23": "Spring Wheat",
    "24": "Winter Wheat",
    "25": "Other Small Grains",
    "26": "Dbl Crop WinWht/Soybeans",
    "27": "Rye",
    "28": "Oats",
    "29": "Millet",
    "30": "Speltz",
    "31": "Canola",
    "32": "Flaxseed",
    "33": "Safflower",
    "34": "Rape Seed",
    "35": "Mustard",
    "36": "Alfalfa",
    "37": "Other Hay/Non Alfalfa",
    "38": "Camelina",
    "39": "Buckwheat",
    "41": "Sugarbeets",
    "42": "Dry Beans",
    "43": "Potatoes",
    "44": "Other Crops",
    "45": "Sugarcane",
    "46": "Sweet Potatoes",
    "47": "Misc Vegs & Fruits",
    "48": "Watermelons",
    "49": "Onions",
    "50": "Cucumbers",
    "51": "Chick Peas",
    "52": "Lentils",
    "53": "Peas",
    "54": "Tomatoes",
    "55": "Caneberries",
    "56": "Hops",
    "57": "Herbs",
    "58": "Clover/Wildflowers",
    "59": "Sod/Grass Seed",
    "60": "Switchgrass",
    "61": "Fallow/Idle Cropland",
    "62": "Pasture/Grass",
    "63": "Forest",
    "64": "Shrubland",
    "65": "Barren",
    "66": "Cherries",
    "67": "Peaches",
    "68": "Apples",
    "69": "Grapes",
    "70": "Christmas Trees",
    "71": "Other Tree Crops",
    "72": "Citrus",
    "74": "Pecans",
    "75": "Almonds",
    "76": "Walnuts",
    "77": "Pears",
    "81": "Clouds/No Data",
    "82": "Developed",
    "83": "Water",
    "87": "Wetlands",
    "88": "Nonag/Undefined",
    "92": "Aquaculture",
    "111": "Open Water",
    "112": "Perennial Ice/Snow",
    "121": "Developed/Open Space",
    "122": "Developed/Low Intensity",
    "123": "Developed/Med Intensity",
    "124": "Developed/High Intensity",
    "131": "Barren",
    "141": "Deciduous Forest",
    "142": "Evergreen Forest",
    "143": "Mixed Forest",
    "152": "Shrubland",
    "176": "Grassland/Pasture",
    "190": "Woody Wetlands",
    "195": "Herbaceous Wetlands",
    "204": "Pistachios",
    "205": "Triticale",
    "206": "Carrots",
    "207": "Asparagus",
    "208": "Garlic",
    "209": "Cantaloupes",
    "210": "Prunes",
    "211": "Olives",
    "212": "Oranges",
    "213": "Honeydew Melons",
    "214": "Broccoli",
    "215": "Avocados",
    "216": "Peppers",
    "217": "Pomegranates",
    "218": "Nectarines",
    "219": "Greens",
    "220": "Plums",
    "221": "Strawberries",
    "222": "Squash",
    "223": "Apricots",
    "224": "Vetch",
    "225": "Dbl Crop WinWht/Corn",
    "226": "Dbl Crop Oats/Corn",
    "227": "Lettuce",
    "228": "Dbl Crop Triticale/Corn",
    "229": "Pumpkins",
    "230": "Dbl Crop Lettuce/Durum Wht",
    "231": "Dbl Crop Lettuce/Cantaloupe",
    "232": "Dbl Crop Lettuce/Cotton",
    "233": "Dbl Crop Lettuce/Barley",
    "234": "Dbl Crop Durum Wht/Sorghum",
    "235": "Dbl Crop Barley/Sorghum",
    "236": "Dbl Crop WinWht/Sorghum",
    "237": "Dbl Crop Barley/Corn",
    "238": "Dbl Crop WinWht/Cotton",
    "239": "Dbl Crop Soybeans/Cotton",
    "240": "Dbl Crop Soybeans/Oats",
    "241": "Dbl Crop Corn/Soybeans",
    "242": "Blueberries",
    "243": "Cabbage",
    "244": "Cauliflower",
    "245": "Celery",
    "246": "Radishes",
    "247": "Turnips",
    "248": "Eggplants",
    "249": "Gourds",
    "250": "Cranberries",
    "254": "Dbl Crop Barley/Soybeans",
    "255":"other"
}

import os
import rasterio
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt  # Add this import

dataset = "/data/datasets/satellite/raw_arkansas_2023/2023_all/"
output  = "./data/Arkansas/output/2023.png"
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

