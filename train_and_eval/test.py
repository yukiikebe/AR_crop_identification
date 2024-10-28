# import torch

# # Create reverse mapping dictionary
Arkansas_classes = {0: {0: 'Background', 1: 'Corn', 2: 'Cotton', 3: 'Rice', 4: 'Sorghum', 5: 'Soybeans', 6: 'Sunflower', 10: 'Peanuts', 11: 'Tobacco', 12: 'Sweet Corn', 13: 'Pop or Orn Corn', 14: 'Mint', 21: 'Barley', 22: 'Durum Wheat', 23: 'Spring Wheat', 24: 'Winter Wheat', 25: 'Other Small Grains', 26: 'Dbl Crop WinWht/Soybeans', 27: 'Rye', 28: 'Oats', 29: 'Millet', 30: 'Speltz', 31: 'Canola', 32: 'Flaxseed', 33: 'Safflower', 34: 'Rape Seed', 35: 'Mustard', 36: 'Alfalfa', 37: 'Other Hay/Non Alfalfa', 38: 'Camelina', 39: 'Buckwheat', 41: 'Sugarbeets', 42: 'Dry Beans', 43: 'Potatoes', 44: 'Other Crops', 45: 'Sugarcane', 46: 'Sweet Potatoes', 47: 'Misc Vegs & Fruits', 48: 'Watermelons', 49: 'Onions', 50: 'Cucumbers', 51: 'Chick Peas', 52: 'Lentils', 53: 'Peas', 54: 'Tomatoes', 55: 'Caneberries', 56: 'Hops', 57: 'Herbs', 58: 'Clover/Wildflowers', 60: 'Switchgrass', 66: 'Cherries', 67: 'Peaches', 68: 'Apples', 69: 'Grapes', 70: 'Christmas Trees', 71: 'Other Tree Crops', 72: 'Citrus', 74: 'Pecans', 75: 'Almonds', 76: 'Walnuts', 77: 'Pears', 204: 'Pistachios', 205: 'Triticale', 206: 'Carrots', 207: 'Asparagus', 208: 'Garlic', 209: 'Cantaloupes', 210: 'Prunes', 211: 'Olives', 212: 'Oranges', 213: 'Honeydew Melons', 214: 'Broccoli', 215: 'Avocados', 216: 'Peppers', 217: 'Pomegranates', 218: 'Nectarines', 219: 'Greens', 220: 'Plums', 221: 'Strawberries', 222: 'Squash', 223: 'Apricots', 224: 'Vetch', 225: 'Dbl Crop WinWht/Corn', 226: 'Dbl Crop Oats/Corn', 227: 'Lettuce', 228: 'Dbl Crop Triticale/Corn', 229: 'Pumpkins', 230: 'Dbl Crop Lettuce/Durum Wht', 231: 'Dbl Crop Lettuce/Cantaloupe', 232: 'Dbl Crop Lettuce/Cotton', 233: 'Dbl Crop Lettuce/Barley', 234: 'Dbl Crop Durum Wht/Sorghum', 235: 'Dbl Crop Barley/Sorghum', 236: 'Dbl Crop WinWht/Sorghum', 237: 'Dbl Crop Barley/Corn', 238: 'Dbl Crop WinWht/Cotton', 239: 'Dbl Crop Soybeans/Cotton', 240: 'Dbl Crop Soybeans/Oats', 241: 'Dbl Crop Corn/Soybeans', 242: 'Blueberries', 243: 'Cabbage', 244: 'Cauliflower', 245: 'Celery', 246: 'Radishes', 247: 'Turnips', 248: 'Eggplants', 249: 'Gourds', 250: 'Cranberries', 254: 'Dbl Crop Barley/Soybeans'}, 1: {0: 'Background', 59: 'Sod/Grass Seed', 61: 'Fallow/Idle Cropland', 62: 'Pasture/Grass', 63: 'Forest', 64: 'Shrubland', 65: 'Barren', 81: 'Clouds/No Data', 82: 'Developed', 83: 'Water', 87: 'Wetlands', 88: 'Nonag/Undefined', 92: 'Aquaculture', 111: 'Open Water', 112: 'Perennial Ice/Snow', 121: 'Developed/Open Space', 122: 'Developed/Low Intensity', 123: 'Developed/Med Intensity', 124: 'Developed/High Intensity', 131: 'Barren', 141: 'Deciduous Forest', 142: 'Evergreen Forest', 143: 'Mixed Forest', 152: 'Shrubland', 176: 'Grassland/Pasture', 190: 'Woody Wetlands', 195: 'Herbaceous Wetlands', 255: 'Other'}}

# Extract only the IDs, removing the class names
class_mappings = {key: list(value.keys()) for key, value in Arkansas_classes.items()}

print(class_mappings)
# exit()
import torch

# # Define the class mappings
# class_mappings = {
#     0: {0, 1, 2, 3, 4, 5, 6},
#     1: {59, 61, 62, 63, 64, 65},
#     2: {221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 254},
#     3: {81, 82, 83, 87, 88, 92, 111, 112, 121, 122, 123, 124, 131, 141, 142, 143, 152, 176, 190, 195, 255}
# }

# Generate a label map
label_map = torch.randint(0, 256, (24, 24, 1))

# Create a new tensor for the mapped label map
mapped_label_map = torch.full_like(label_map, -1)  # Initialize with -1 to identify unmapped values

# Apply the class mappings
for new_class, original_classes in class_mappings.items():
    mask = torch.isin(label_map, torch.tensor(list(original_classes)))
    mapped_label_map[mask] = new_class

print(mapped_label_map)