import numpy as np
import ee
import geemap
import os
import multiprocessing
from tqdm import tqdm
import rasterio


ee.Authenticate()
ee.Initialize(project='satelite-430703')

# Configuration
roig = [
    [-94.7610, 36.6652],
    [-94.7610, 32.8376],
    [-89.5522, 36.6652],
    [-89.5522, 32.8376],
]
start_day = '2023-01-01'
end_day = '2023-12-31'
data_dir = '/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all/'


def save_rgb_image(red_path, green_path, blue_path, output_path):
    if not all([
        os.path.exists(red_path),
        os.path.exists(green_path),
        os.path.exists(blue_path),
    ]):
        return

    with rasterio.open(red_path) as red_file:
        red = red_file.read(1)
        profile = red_file.profile

    with rasterio.open(green_path) as green_file:
        green = green_file.read(1)

    with rasterio.open(blue_path) as blue_file:
        blue = blue_file.read(1)
    
    stacked_data = np.stack((red, green, blue))
    profile.update(count=3)

    with rasterio.open(output_path, 'w', **profile) as dst_file:
        dst_file.write(stacked_data)


def download_dataset(roig, start_day, end_day, save_dir):
    #print("Download.py at ", roig, start_day, end_day, save_dir)
    roi = ee.Geometry.Polygon(roig)
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(start_day, end_day) \
        .filterBounds(roi)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    collection_size = collection.size()
    #print("The size of Sentinel-2 Image Collection:", collection_size.getInfo())
    image_list = collection.toList(collection_size)
    n_images = collection_size.getInfo()
    is_first_date = True

    for i in range(n_images):
        image = ee.Image(image_list.get(i))
        date = image.date().format("YYYY-MM-dd").getInfo()
        cloud_percentage = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        #print(f"Image {i+1}/{n_images}: Date={date}, Cloudy Pixel={cloud_percentage}%")

        path_download = os.path.join(save_dir, str(date)) 
        if not os.path.exists(path_download):
            os.makedirs(path_download)
        #print("path_download ", path_download)


        # Save statellite image to local disk

        # B1 band with 60m resolution
        #image_B1 = image.select('B1')
        #output_path = os.path.join(path_download, f'B1_{date}.tif')
        #geemap.ee_export_image(image_B1, filename=output_path, scale=60, crs='EPSG:3857', region=roi)

        # B2 band with 10m resolution
        output_path = os.path.join(path_download, f'B2_{date}.tif')
        if not os.path.exists(output_path):
            image_B2 = image.select('B2')
            geemap.ee_export_image(image_B2, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # B3 band with 10m resolution
        output_path = os.path.join(path_download, f'B3_{date}.tif')
        if not os.path.exists(output_path):
            image_B3 = image.select('B3')
            geemap.ee_export_image(image_B3, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # B4 band with 10m resolution
        output_path = os.path.join(path_download, f'B4_{date}.tif')
        if not os.path.exists(output_path):
            image_B4 = image.select('B4')
            geemap.ee_export_image(image_B4, filename=output_path, scale=10, crs='EPSG:3857', region=roi)
        
        # Make and save RGB image
        output_path=os.path.join(path_download, f'10m_rgb_{date}.tif')
        if is_first_date and not os.path.exists(output_path):
            save_rgb_image(
                red_path=os.path.join(path_download, f'B4_{date}.tif'),
                green_path=os.path.join(path_download, f'B3_{date}.tif'),
                blue_path=os.path.join(path_download, f'B2_{date}.tif'),
                output_path=output_path,
            )
            is_first_date = False

        # B5 band with 20m resolution
        output_path = os.path.join(path_download, f'B5_{date}.tif')
        if not os.path.exists(output_path):
            image_B5 = image.select('B5')
            geemap.ee_export_image(image_B5, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B6 band with 20m resolution
        output_path = os.path.join(path_download, f'B6_{date}.tif')
        if not os.path.exists(output_path):
            image_B6 = image.select('B6')
            geemap.ee_export_image(image_B6, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B7 band with 20m resolution
        output_path = os.path.join(path_download, f'B7_{date}.tif')
        if not os.path.exists(output_path):
            image_B7 = image.select('B7')
            geemap.ee_export_image(image_B7, filename=output_path, scale=20, crs='EPSG:3857', region=roi)
        
        # B8 band with 10m resolution
        output_path = os.path.join(path_download, f'B8_{date}.tif')
        if not os.path.exists(output_path):
            image_B8 = image.select('B8')
            geemap.ee_export_image(image_B8, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # B8A band with 20m resolution
        output_path = os.path.join(path_download, f'B8A_{date}.tif')
        if not os.path.exists(output_path):
            image_B8A = image.select('B8A')
            geemap.ee_export_image(image_B8A, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B9 band with 60m resolution
        #image_B9 = image.select('B9')
        #output_path = os.path.join(path_download, f'B9_{date}.tif')
        #if not os.path.exists(output_path):
        #    geemap.ee_export_image(image_B9, filename=output_path, scale=60, crs='EPSG:3857', region=roi)

        # B11 band with 20m resolution
        output_path = os.path.join(path_download, f'B11_{date}.tif')
        if not os.path.exists(output_path):
            image_B11 = image.select('B11')
            geemap.ee_export_image(image_B11, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B12 band with 20m resolution
        output_path = os.path.join(path_download, f'B12_{date}.tif')
        if not os.path.exists(output_path):
            image_B12 = image.select('B12')
            geemap.ee_export_image(image_B12, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # SCL band with 20m resolution
        output_path = os.path.join(path_download, f'SCL_{date}.tif')
        if not os.path.exists(output_path):
            image_SCL = image.select('SCL')
            geemap.ee_export_image(image_SCL, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # TCI_R band with 10m resolution
        #image_TCI_R = image.select('TCI_R')
        #output_path = os.path.join(path_download, f'TCI_R_{date}.tif')
        #if not os.path.exists(output_path):
        #    geemap.ee_export_image(image_TCI_R, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # TCI_G band with 10m resolution
        #image_TCI_G = image.select('TCI_G')
        #output_path = os.path.join(path_download, f'TCI_G_{date}.tif')
        #if not os.path.exists(output_path):
        #    geemap.ee_export_image(image_TCI_G, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # TCI_B band with 10m resolution
        #image_TCI_B = image.select('TCI_B')
        #output_path = os.path.join(path_download, f'TCI_B_{date}.tif')
        #if not os.path.exists(output_path):
        #    geemap.ee_export_image(image_TCI_B, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # MSK_CLDPRB band with 20m resolution
        #image_MSK_CLDPRB = image.select('MSK_CLDPRB')
        #output_path = os.path.join(path_download, f'MSK_CLDPRB_{date}.tif')
        #geemap.ee_export_image(image_MSK_CLDPRB, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # QA10 band with 10m resolution
        #image_QA10 = image.select('QA10')
        #output_path = os.path.join(path_download, f'QA10_{date}.tif')
        #geemap.ee_export_image(image_QA10, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # QA20 band with 20m resolution
        #image_QA20 = image.select('QA20')
        #output_path = os.path.join(path_download, f'QA20_{date}.tif')
        #geemap.ee_export_image(image_QA20, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # QA60 band with 60m resolution
        #image_QA60 = image.select('QA60')
        #output_path = os.path.join(path_download, f'QA60_{date}.tif')
        #geemap.ee_export_image(image_QA60, filename=output_path, scale=60, crs='EPSG:3857', region=roi)

    #print("Download completed!")

if __name__ == "__main__":
    # Extracting the longitude and latitude of the corners
    lon_min = min(roig[0][0], roig[1][0])
    lon_max = max(roig[2][0], roig[3][0])
    lat_min = min(roig[1][1], roig[3][1])
    lat_max = max(roig[0][1], roig[2][1])

    # Create linspace for longitude and latitude
    lon_range = np.linspace(lon_min, lon_max, 21)  # 20 intervals, so 21 points
    lat_range = np.linspace(lat_min, lat_max, 21)
    print(lon_range)
    print(lat_range)

    # Function to get the four corners of a grid cell
    def get_grid_corners(i, j, lon_range, lat_range):
        lon1, lon2 = lon_range[i], lon_range[i+1]
        lat1, lat2 = lat_range[j], lat_range[j+1]
        return [
            [lon1, lat2],  # top-left
            [lon1, lat1],  # bottom-left
            [lon2, lat2],  # top-right
            [lon2, lat1],  # bottom-right
        ]

    # Iterate through each grid cell
    grids = []
    for i in range(lon_range.shape[0]-1):  # 10x10 grid
        for j in range(lat_range.shape[0]-1):
            grid_dir = os.path.join(data_dir, f'{i}_{j}')
            grid_corners = get_grid_corners(i, j, lon_range, lat_range)
            grids.append((grid_dir, grid_corners))

    def download_grid(grid_data):
        grid_dir, grid_corners = grid_data
        os.makedirs(grid_dir, exist_ok=True)
        download_dataset(grid_corners, start_day, end_day, grid_dir)


    with multiprocessing.Pool() as pool:
        with tqdm(total=len(grids)) as pbar:
            for _ in pool.imap_unordered(download_grid, grids):
                pbar.update()
