
################ 1. Read data from csv file ################

# #read pickle file
# import pickle
# import numpy as np
# path = "/home/vuonghn/research/code/Agriculture/DeepSatModels/datasets/AR24/pickle24x24"
# import os
# list_file = os.listdir(path)
# for file in list_file:
#     file_path = os.path.join(path, file)
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     print("label shape ", data['labels'].shape, np.unique(data['labels']))



################### 2. Resample dates ###################

# import pandas as pd

# # Sample requirements
# sample_requirements = {
#     1: 1,  # January
#     2: 1,  # February
#     3: 1,  # March
#     4: 2,  # April
#     5: 2,  # May
#     6: 2,  # June
#     7: 2,  # July
#     8: 2,  # August
#     9: 2,  # September
#     10: 1, # October
#     11: 1, # November
#     12: 1  # December
# }

# def resample_dates(dates):
#     # Convert dates to DataFrame
#     df = pd.DataFrame(dates, columns=['date'])
#     df['date'] = pd.to_datetime(df['date'])
#     df['month'] = df['date'].dt.month
#     df['day'] = df['date'].dt.day
    
#     resampled_dates = []
    
#     # Group by month and sample
#     for month, group in df.groupby('month'):
#         sample_size = sample_requirements.get(month, 0)
#         if sample_size > 0:
#             group = group.sort_values(by='date')
#             if sample_size == 1:
#                 # Prefer the date in the middle of the month
#                 middle_date = group.iloc[(group['day'] - 15).abs().argsort()[:1]]
#                 resampled_dates.extend(middle_date['date'].dt.strftime('%Y-%m-%d').tolist())
#             elif sample_size == 2:
#                 # Prefer the beginning and ending of the month
#                 beginning_date = group.iloc[:1]
#                 ending_date = group.iloc[-1:]
#                 resampled_dates.extend(beginning_date['date'].dt.strftime('%Y-%m-%d').tolist())
#                 resampled_dates.extend(ending_date['date'].dt.strftime('%Y-%m-%d').tolist())
    
#     return resampled_dates

# # Example usage
# dates = [
#     '2023-02-12', '2023-11-04', '2023-05-03', '2023-10-30', '2023-11-01', 
#     '2023-06-27', '2023-09-25', '2023-07-27', '2023-09-27', '2023-03-04', 
#     '2023-10-07', '2023-09-10', '2023-06-17', '2023-04-08', '2023-09-30', 
#     '2023-10-25', '2023-09-15', '2023-08-18', '2023-10-17', '2023-01-08', 
#     '2023-01-23', '2023-04-18', '2023-04-10', '2023-03-26', '2023-06-04', 
#     '2023-03-19', '2023-03-14', '2023-04-23', '2023-06-09', '2023-05-20', 
#     '2023-05-28', '2023-08-31', '2023-06-29', '2023-08-21', '2023-05-25', 
#     '2023-10-12', '2023-03-29', '2023-08-28', '2023-01-05', '2023-12-11', 
#     '2023-12-19', '2023-07-17', '2023-11-11', '2023-06-24', '2023-01-15', 
#     '2023-01-10', '2023-05-18', '2023-02-27', '2023-12-26', '2023-12-06', 
#     '2023-08-11', '2023-06-07', '2023-05-30', '2023-09-17', '2023-10-22', 
#     '2023-10-02', '2023-10-20', '2023-02-17', '2023-07-02', '2023-12-14', 
#     '2023-10-10', '2023-09-07', '2023-01-20'
# ]

# # Sort dates
# # dates.sort()
# print(dates)
# resampled = resample_dates(dates)
# resampled.sort()
# print(resampled)




###########3. Check distriution of data ################


num2class = {
    "0": "Background",
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


data = "/data/datasets/satellite/raw_arkansas_2023/2023_all"
train_csv = "/home/vuonghn/research/code/Agriculture/fold/fold-paths_0/train_data.csv"
val_csv = "/home/vuonghn/research/code/Agriculture/fold/fold-paths_0/val_data.csv"

csv_files = {
    "train": train_csv,
    "val": val_csv
}

classes = {
    "train": set(),
    "val": set()
}

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import csv

# Read the train CSV file
# train_data = pd.read_csv(train_csv)

# Read the validation CSV file
# val_data = pd.read_csv(val_csv)

for split, csv_file in csv_files.items():
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            row = row[0].split("/")[-1]
            # print("row ", row)
            cdl_image_path = os.path.join(data, row, "cdl.tif")
            # print("cdl_image_path ", cdl_image_path)
            with rasterio.open(cdl_image_path) as src:
                cdl_image = src.read(1)
            unique_values = np.unique(cdl_image)
            classes[split].update(unique_values)
            # print("classes[split] ",classes[split])

            

            # img_path, label_path = row
            # print("img_path ", img_path)
            # print("label_path ", label_path)
            # with rasterio.open(img_path) as src:
            #     img = src.read(1)
            # with rasterio.open(label_path) as src:
            #     label = src.read(1)
            # print("img ", img.shape, np.unique(img))
            # print("label ", label.shape, np.unique(label))
            # break

classes["train"] = list(classes["train"])
classes["val"] = list(classes["val"])


crop = {
    "train": [],
    "val": []
}

for split, class_ids in classes.items():
    print(f"Split: {split}, Number of classes: {len(class_ids)}")
    for class_id in class_ids:
        if str(class_id) in num2class:
            print(num2class[str(class_id)])
            crop[split].append(num2class[str(class_id)])
    print("\n")


train_list = crop["train"]
val_list = crop["val"]


print("train_list ",train_list)
print("val_list ",val_list)

# Merge and keep unique values
merge_list = list(set(train_list + val_list))

# Print the total number of elements in merge_list
print(f"Total elements in merge_list: {len(merge_list)}")

# Print the merged list for debugging
print(f"merge_list: {merge_list}")

# Check if any value in train_list is not in merge_list
train_not_in_merge = [item for item in train_list if item not in merge_list]
print(f"Values in train_list not in merge_list: {train_not_in_merge}")

# Check if any value in val_list is not in merge_list
val_not_in_merge = [item for item in val_list if item not in merge_list]
print(f"Values in val_list not in merge_list: {val_not_in_merge}")

# Check if any value in val_list is not in train_list
val_not_in_train = [item for item in val_list if item not in train_list]
print(f"Values in val_list not in train_list: {val_not_in_train}")

# Additional debugging: Print lengths of original lists
print(f"Length of train_list: {len(train_list)}")
print(f"Length of val_list: {len(val_list)}")

# Additional debugging: Print elements in train_list but not in val_list
train_not_in_val = [item for item in train_list if item not in val_list]
print(f"Values in train_list not in val_list: {train_not_in_val}")

# Additional debugging: Print elements in val_list but not in train_list
val_not_in_train = [item for item in val_list if item not in train_list]
print(f"Values in val_list not in train_list: {val_not_in_train}")