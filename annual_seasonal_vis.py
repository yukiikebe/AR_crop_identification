import json
import numpy as np
from shapely.geometry import shape, Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image, ImageDraw
import os

SEASONAL_CLASSES = {
    1: 'Almonds',
    4: 'Alfalfa & Alfalfa Mixtures',
    5: 'Pistachios',
    6: 'Mixed Pasture',
    7: 'Walnuts',
    14: 'Native Pasture',
    23: 'Olives',
    24: 'Avocados',
    26: 'Strawberries',
    27: 'Prunes',
    28: 'Cherries',
    31: 'Flowers, Nursery & Christmas Tree Farms',
    33: 'Pomegranates',
    34: 'Bush Berries',
    38: 'Plums',
    39: 'Dates',
    40: 'Eucalyptus',
    43: 'Turf Farms',
    45: 'Pears',
    46: 'Apples',
    47: 'Pecans',
    48: 'Kiwis',
    49: 'Apricots',
    51: 'Induced High Water Table Native Pasture',
    53: 'Miscellaneous Subtropical Fruit'
}

ANNUAL_CLASSES = {
    2: 'Corn, Sorghum, or Sudan Grouped for Remote Sensing Only',
    9: 'Miscellaneous Grain and Hay',
    10: 'Rice',
    11: 'Wheat',
    12: 'Tomato (Processing and Market)',
    15: 'Cotton',
    16: 'Lettuce or Leafy Greens Grouped for Remote Sensing Only',
    17: 'Miscellaneous Truck Crops',
    19: 'Cole Crops (Mixture of 22-25)',
    20: 'Onions & Garlic',
    22: 'Melons, Squash, and Cucumbers (All Types)',
    25: 'Safflower',
    29: 'Carrots',
    30: 'Sunflowers',
    32: 'Potatoes',
    35: 'Sweet Potatoes',
    37: 'Sugar Beets',
    41: 'Beans (Dry)',
    42: 'Wild Rice',
    44: 'Peppers (Chili, Bell, etc.)',
    52: 'Miscellaneous Field Crops'
}

CDL_CLASSES = {
    1: 'Almonds',
    2: 'Corn,Sorghum or Sudan grouped for remote sensing only',
    3: 'Land not cropped the current or previous crop season',
    4: 'Alfalfa & alfalfa mixtures',
    5: 'Pistachios',
    6: 'Mixed pasture',
    7: 'Walnuts',
    8: 'Long term idle land',
    9: 'Miscellaneous grain and hay',
    10: 'Rice',
    11: 'Wheat',
    12: 'Tomato (processing and market)',
    13: 'Miscellaneous grasses',
    14: 'Native pasture',
    15: 'Cotton',
    16: 'Lettuce or Leafy Greens grouped for remote sensing only',
    17: 'Miscellaneous truck',
    18: 'Golf course – irrigated',
    19: 'Cole crops (mixture of 22-25)',
    20: 'Onions & garlic',
    21: 'Peaches and nectarines',
    22: 'Melons, squash, and cucumbers (all types)',
    23: 'Olives',
    24: 'Avocados',
    25: 'Safflower',
    26: 'Strawberries',
    27: 'Prunes',
    28: 'Cherries',
    29: 'Carrots',
    30: 'Sunflowers',
    31: 'Flowers, nursery & Christmas tree farms',
    32: 'Potatoes',
    33: 'Pomegranates',
    34: 'Bush berries',
    35: 'Sweet potatoes',
    36: 'Miscellaneous deciduous',
    37: 'Sugar beets',
    38: 'Plums',
    39: 'Dates',
    40: 'Eucalyptus',
    41: 'Beans (dry)',
    42: 'Wild Rice',
    43: 'Turf farms',
    44: 'Peppers (chili, bell, etc.)',
    45: 'Pears',
    46: 'Apples',
    47: 'Pecans',
    48: 'Kiwis',
    49: 'Apricots',
    50: 'Greenhouse',
    51: 'Induced high water table native pasture',
    52: 'Miscellaneous field',
    53: 'Miscellaneous subtropical fruit'
}

def load_geojson(file_path):
    """Load GeoJSON file and return the data"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_data_classes(data):
    """Analyze which classes are present in the data"""
    unique_labels = set()
    for feature in data['features']:
        label = feature['properties']['subclass_name_crop2']
        unique_labels.add(label)
    return unique_labels

def compute_affine_transform(roi_polygon, image_size):
    """Compute the transformation parameters from world to image coordinates"""
    minx, miny, maxx, maxy = roi_polygon.bounds
    transform_params = {
        'minx': minx,
        'miny': miny,
        'maxy': maxy,
        'x_range': maxx - minx,
        'y_range': maxy - miny,
        'image_width': image_size[0],
        'image_height': image_size[1]
    }
    return transform_params

def world_to_image_coords(x, y, transform_params):
    """Convert world coordinates to image coordinates"""
    ix = int((x - transform_params['minx']) / transform_params['x_range'] * 
             (transform_params['image_width'] - 1))
    iy = int((transform_params['maxy'] - y) / transform_params['y_range'] * 
             (transform_params['image_height'] - 1))
    return ix, iy

def rasterize_features(data, roi_polygon, class_dict, transform_params, image_size):
    """Rasterize GeoJSON features to an image"""
    mask_image = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask_image)
    
 
    name_to_value = {v: k for k, v in class_dict.items()}
    
    processed_features = 0
    for feature in data['features']:
        label = feature['properties']['subclass_name_crop2']
        if label in name_to_value:
            value = name_to_value[label]
            geometry = feature['geometry']
            feature_shape = shape(geometry)
            
            if feature_shape.intersects(roi_polygon):
                intersection = feature_shape.intersection(roi_polygon)
                if not intersection.is_empty:
                    processed_features += 1
                    
                    if intersection.geom_type == 'Polygon':
                        polygons = [intersection]
                    elif intersection.geom_type == 'MultiPolygon':
                        polygons = list(intersection.geoms)
                    else:
                        continue

                    for poly in polygons:
                        exterior_coords = [(coord[0], coord[1]) for coord in poly.exterior.coords]
                        exterior_pixels = [world_to_image_coords(x, y, transform_params) 
                                        for x, y in exterior_coords]
                        draw.polygon(exterior_pixels, fill=value)
    
    print(f"Processed {processed_features} features")
    return mask_image

def display_and_save_mask(mask_image, class_dict, cmap='tab20', 
                         filename='roi_mask.png', image_size=None):
    """Display and save the classified mask with legend"""
    mask = np.array(mask_image)
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values != 0]  # Remove background
    
    if len(unique_values) == 0:
        print("No classified pixels found in the mask!")
        return
    

    existing_classes = {v: class_dict[v] for v in unique_values if v in class_dict}
    n_classes = len(existing_classes)
    

    cmap = plt.get_cmap(cmap, n_classes)
    value_to_index = {val: idx for idx, val in enumerate(existing_classes.keys())}
    

    indexed_mask = np.zeros_like(mask)
    for val, idx in value_to_index.items():
        indexed_mask[mask == val] = idx
    

    plt.figure(figsize=(12, 8), dpi=300)  # Higher DPI for better quality
    im = plt.imshow(indexed_mask, cmap=cmap)
    plt.title(f'Crop Classification Map ({n_classes} classes)')
    

    cbar = plt.colorbar(im, ticks=range(n_classes))
    cbar_labels = [f"{v}: {existing_classes[v]}" for v in existing_classes.keys()]
    cbar.ax.set_yticklabels(cbar_labels, fontsize=8)
    cbar.set_label('Crop Classes', rotation=270, labelpad=15)
    

    print(f"\nFound {n_classes} classes in the mask:")
    for value, name in existing_classes.items():
        pixel_count = np.sum(mask == value)
        print(f"Class {value} ({name}): {pixel_count} pixels")
    

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Colored visualization saved as {filename}")
    
    plt.show()
    plt.close()

def main():

    geojson_file_path = '/Users/maral/Downloads/ca.geojson'
    

    ROI = [
        [-124.375251, 42.004483],
        [-114.132542, 42.004483],
        [-114.132542, 32.540499],
        [-124.375251, 32.540499]
    ]
    image_size = [3385, 2925]
    
 
    print("Loading GeoJSON data...")
    data = load_geojson(geojson_file_path)
    
 
    roi_polygon = Polygon(ROI)
    
    
    for class_name, class_dict in [
        ("CDL", CDL_CLASSES),
        ("Seasonal", SEASONAL_CLASSES),
        ("Annual", ANNUAL_CLASSES)
    ]:
        print(f"\nProcessing {class_name} classification...")
        transform_params = compute_affine_transform(roi_polygon, image_size)
        mask_image = rasterize_features(data, roi_polygon, class_dict, 
                                      transform_params, image_size)

        base_path = f'/Users/maral/Desktop/satelite_img/{class_name.lower()}'
        mask_image.save(f'{base_path}_raw_mask.tiff')
        

        display_and_save_mask(mask_image, class_dict, 
                            filename=f'{base_path}_colored_mask.png', 
                            image_size=image_size)

if __name__ == "__main__":
    main()
