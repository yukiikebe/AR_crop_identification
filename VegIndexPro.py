import glob
import pickle
import os
import numpy as np
import time


VEGETATION_INDICES =  {
    'NDVI': {
        'expr': '(N - R) / (N + R)',
        'help': 'Normalized Difference Vegetation Index shows the amount of green vegetation.',
        'range': (0.05, 95)
    },
    'NDYI': {
        'expr': '(G - B) / (G + B)',
        'help': 'Normalized difference yellowness index (NDYI), best model variability in relative yield potential in Canola.',
        'range': (-1, 1)
    },
    'NDRE': {
        'expr': '(N - Re) / (N + Re)',
        'help': 'Normalized Difference Red Edge Index shows the amount of green vegetation of permanent or later stage crops.',
        'range': (-300, 1)
    },
    'NDWI': {
        'expr': '(G - N) / (G + N)',
        'help': 'Normalized Difference Water Index shows the amount of water content in water bodies.',
        'range': (-1, 1)
    },
    'NDVI (Blue)': {
        'expr': '(N - B) / (N + B)',
        'help': 'Normalized Difference Vegetation Index shows the amount of green vegetation.',
        'range': (-1, 1)
    },
    'ENDVI':{
        'expr': '((N + G) - (2 * B)) / ((N + G) + (2 * B))',
        'help': 'Enhanced Normalized Difference Vegetation Index is like NDVI, but uses Blue and Green bands instead of only Red to isolate plant health.'
    },
    'vNDVI':{
        'expr': '0.5268*((R ** -0.1294) * (G ** 0.3389) * (B ** -0.3118))',
        'help': 'Visible NDVI is an un-normalized index for RGB sensors using constants derived from citrus, grape, and sugarcane crop data.'
    },    
    'VARI': {
        'expr': '(G - R) / (G + R - B)',
        'help': 'Visual Atmospheric Resistance Index shows the areas of vegetation.',
        'range': (-1, 1)
    },    
    'MPRI': {
        'expr': '(G - R) / (G + R)',
        'help': 'Modified Photochemical Reflectance Index',
        'range': (-1, 1)
    },
    'EXG': {
        'expr': '(2 * G) - (R + B)',
        'help': 'Excess Green Index (derived from only the RGB bands) emphasizes the greenness of leafy crops such as potatoes.'
    },
    'BAI': {
        'expr': '1.0 / (((0.1 - R) ** 2) + ((0.06 - N) ** 2))',
        'help': 'Burn Area Index highlights burned land in the red to near-infrared spectrum.'
    },
    'GLI': {
        'expr': '((G * 2) - R - B) / ((G * 2) + R + B)',
        'help': 'Green Leaf Index shows greens leaves and stems.',
        'range': (-1, 1)
    },
    'GNDVI':{
        'expr': '(N - G) / (N + G)',
        'help': 'Green Normalized Difference Vegetation Index is similar to NDVI, but measures the green spectrum instead of red.',
        'range': (-1, 1)
    },
    'GRVI':{
        'expr': 'N / G',
        'help': 'Green Ratio Vegetation Index is sensitive to photosynthetic rates in forests.'
    },
    'SAVI':{
        'expr': '(1.5 * (N - R)) / (N + R + 0.5)',
        'help': 'Soil Adjusted Vegetation Index is similar to NDVI but attempts to remove the effects of soil areas using an adjustment factor (0.5).'
    },
    'MNLI':{
        'expr': '((N ** 2 - R) * 1.5) / (N ** 2 + R + 0.5)',
        'help': 'Modified Non-Linear Index improves the Non-Linear Index algorithm to account for soil areas.'
    },
    'MSR': {
        'expr': '((N / R) - 1) / (np.sqrt(N / R) + 1)',
        'help': 'Modified Simple Ratio is an improvement of the Simple Ratio (SR) index to be more sensitive to vegetation.'
    },
    'RDVI': {
        'expr': '(N - R) / np.sqrt(N + R)',
        'help': 'Renormalized Difference Vegetation Index uses the difference between near-IR and red, plus NDVI to show areas of healthy vegetation.'
    },
    'TDVI': {
        'expr': '1.5 * ((N - R) / np.sqrt(N ** 2 + R + 0.5))',
        'help': 'Transformed Difference Vegetation Index highlights vegetation cover in urban environments.'
    },
    'OSAVI': {
        'expr': '(N - R) / (N + R + 0.16)',
        'help': 'Optimized Soil Adjusted Vegetation Index is based on SAVI, but tends to work better in areas with little vegetation where soil is visible.'
    },
    'LAI': {
        'expr': '3.618 * (2.5 * (N - R) / (N + 6*R - 7.5*B + 1)) * 0.118',
        'help': 'Leaf Area Index estimates foliage areas and predicts crop yields.',
        'range': (-1, 1)
    },
    'EVI': {
        'expr': '2.5 * (N - R) / (N + 6*R - 7.5*B + 1)',
        'help': 'Enhanced Vegetation Index is useful in areas where NDVI might saturate, by using blue wavelengths to correct soil signals.',
        'range': (-1, 1)
    },
    'ARVI': {
        'expr': '(N - (2 * R) + B) / (N + (2 * R) + B)',
        'help': 'Atmospherically Resistant Vegetation Index. Useful when working with imagery for regions with high atmospheric aerosol content.',
        'range': (-1, 1)
    },
}

def list_and_read_pickle_files(base_dir, output_base_dir="./output"):
    pickle_files = glob.glob(os.path.join(base_dir, "**", "*.pickle"), recursive=True)
    
    
    valid_labels = [2, 4, 6, 9, 10, 11, 12, 14, 17, 19, 20, 22, 25, 26, 30, 32, 34, 35, 37, 41, 42, 44, 52]
    
    for number, file_path in enumerate(pickle_files):
        print(f"{number} Processing: {file_path}")
        
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                labels = data["labels"]
                unique_values, counts = np.unique(labels, return_counts=True)
                if np.array_equal(unique_values, [1]):
                    with open('zero_tile_path_list1.txt', 'a') as f_out:
                        f_out.write(file_path + '\n')
                    continue

                labels = np.where(np.isin(labels, valid_labels), labels, 1)
                data['labels']=labels
                unique_values, counts = np.unique(labels, return_counts=True)
                if np.array_equal(unique_values, [1]):
                    with open('zero_tile_path_list_after_seasonal_filte1.txt', 'a') as f_out:
                        f_out.write(file_path + '\n')
                    continue

                # Extract bands
                img_data = data['img']
                B = img_data[0]  # Blue
                G = img_data[1]  # Green
                R = img_data[2]  # Red
                Re = img_data[3]  # Red Edge
                N = img_data[6]  # NIR

                print(B.shape, G.shape, R.shape,Re.shape)
                indices = {}
                for index_name, expr in VEGETATION_INDICES.items():
                    try:
                        index = eval(expr['expr'])
                        index = np.nan_to_num(index)  
                        data[index_name] = index
                        time.sleep(10)
                    except Exception as e:
                        print(f"Error calculating {index_name}: {str(e)}")
                        continue
                
                dir_name = os.path.dirname(file_path)
                base_name =os.path.basename(file_path)
                dir_name = dir_name.replace('new_out',"index_output")
                os.makedirs(dir_name, exist_ok=True)
                output_path = os.path.join(dir_name, base_name)
            

                
                with open(output_path, 'wb') as f_out:
                    pickle.dump(data, f_out)
                print(f"Saved processed file to {output_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

base_directory = "/mnt/data/mzarvani/new_out/out/pickle24x24/"
list_and_read_pickle_files(base_directory)
 
