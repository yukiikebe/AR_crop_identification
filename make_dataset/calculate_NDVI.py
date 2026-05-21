import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def read_band(dataset, band_number):
    return dataset.read(band_number).astype('float32')

def calculate_ndvi(file_path, output_folder):
    with rasterio.open(file_path) as dataset:
        if dataset.count > 3:
            R = read_band(dataset, 1) 
            N = read_band(dataset, 4) 
            print("Max", R.max(), N.max())
            print("Min", R.min(), N.min())
            ndvi = (N - R) / (N + R)
            print("minmax", ndvi.min(), ndvi.max())
            ndvi = np.clip(ndvi, -1, 1)
            print("ndvi", ndvi.max(), ndvi.min())
            print("dtype", ndvi.dtype) 
        elif dataset.count == 2:
            R = read_band(dataset, 1) 
            N = read_band(dataset, 2)
            print("Max", R.max(), N.max())
            print("Min", R.min(), N.min())
            ndvi = (N - R) / (N + R)
            print("minmax", ndvi.min(), ndvi.max())
            ndvi = np.clip(ndvi, -1, 1)
            print("ndvi", ndvi)
        elif dataset.count == 1:
            ndvi = read_band(dataset, 1)
            ndvi = np.clip(ndvi, -1, 1)
            print("ndvi", ndvi.max(), ndvi.min())
            print("dtype", ndvi.dtype) 
        
        output_tiff = os.path.join(output_folder, "NDVI.tif")
        with rasterio.open(
            output_tiff, 'w',
            driver='GTiff',
            height=ndvi.shape[0],
            width=ndvi.shape[1],
            count=1,
            dtype=np.float32,
            crs=dataset.crs,
            transform=dataset.transform,
        ) as dst:
            dst.write(ndvi, 1)
        print(f"NDVI GeoTIFF saved at: {output_tiff}")
        
    
        plt.figure(figsize=(10, 10))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI')
        # plt.title('NDVI from Drone Imagery')
        plt.axis('off')
        output_png = os.path.join(output_folder, "NDVI.png")
        plt.savefig(output_png, bbox_inches='tight', pad_inches=0.1)
        print(f"NDVI plot saved at: {output_png}")

def psnr_calculate(sr_ndvi_path, input_ndvi_path):
    sr_ndvi = rasterio.open(sr_ndvi_path).read(1)
    input_ndvi = rasterio.open(input_ndvi_path).read(1)
    epsilon = 1e-8
    sr_ndvi = np.clip(sr_ndvi, -1, 1)
    input_ndvi = np.clip(input_ndvi, -1, 1)
    psnr = 10 * np.log10(1 / (np.mean((sr_ndvi - input_ndvi) ** 2) + epsilon))
    return psnr

file_path = "../Orthophotos_patches_tiff_scale16_combine_20_NDVI_again/merged_farmland.tif" 
output_folder = "../Orthophotos_patches_tiff_scale16_combine_20_NDVI_again"
os.makedirs(output_folder, exist_ok=True)
calculate_ndvi(file_path, output_folder)

# sr_ndvi_path = "../Orthophotos_patches_tiff_scale16_combine_20_Multiband/NDVI.tif"
# input_ndvi_path = "../Orthophotos_patches_tiff_scale16_combine_20_Input/NDVI.tif"
# psnr = psnr_calculate(sr_ndvi_path, input_ndvi_path)
# print("PSNR:", psnr)

