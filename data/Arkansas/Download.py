import ee
import geemap
import os
ee.Authenticate()
ee.Initialize(project='ee-vvuonghn')

# Configuration
roig = [
    [-93.70856254511031, 35.43604890036842],
    [-93.70856254511031, 35.57542037748203],
    [-93.96375646096627, 35.57542037748203],
    [-93.96375646096627, 35.43604890036842],
    [-93.70856254511031, 35.43604890036842]
]
start_day = '2023-01-01'
end_day = '2023-12-31'
save_dir = '/home/vuonghn/research/dataset/satellite/arkansas/satellite_images/'

def download_dataset(roig, start_day, end_day,save_dir):
    print("Downloading Sentinel-2 Image Collection...")
    roi = ee.Geometry.Polygon(roig)
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(start_day, end_day) \
        .filterBounds(roi)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    collection_size = collection.size()
    print("The size of Sentinel-2 Image Collection:", collection_size.getInfo())
    image_list = collection.toList(collection_size)
    n_images = collection_size.getInfo()



    for i in range(n_images):
        image = ee.Image(image_list.get(i))
        date = image.date().format("YYYY-MM-dd").getInfo()
        cloud_percentage = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        print(f"Image {i+1}/{n_images}: Date={date}, Cloudy Pixel={cloud_percentage}%")

        path_download = os.path.join(save_dir, str(date)) 
        if not os.path.exists(path_download):
            os.makedirs(path_download)
        print("path_download ", path_download)


        # Save statellite image to local disk

        # B1 band with 60m resolution
        image_B1 = image.select('B1')
        output_path = os.path.join(path_download, f'B1_{date}.tif')
        geemap.ee_export_image(image_B1, filename=output_path, scale=60, crs='EPSG:3857', region=roi)

        # B2 band with 10m resolution
        image_B2 = image.select('B2')
        output_path = os.path.join(path_download, f'B2_{date}.tif')
        geemap.ee_export_image(image_B2, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # B3 band with 10m resolution
        image_B3 = image.select('B3')
        output_path = os.path.join(path_download, f'B3_{date}.tif')
        geemap.ee_export_image(image_B3, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # B4 band with 10m resolution
        image_B4 = image.select('B4')
        output_path = os.path.join(path_download, f'B4_{date}.tif')
        geemap.ee_export_image(image_B4, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # B5 band with 20m resolution
        image_B5 = image.select('B5')
        output_path = os.path.join(path_download, f'B5_{date}.tif')
        geemap.ee_export_image(image_B5, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B6 band with 20m resolution
        image_B6 = image.select('B6')
        output_path = os.path.join(path_download, f'B6_{date}.tif')
        geemap.ee_export_image(image_B6, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B7 band with 20m resolution
        image_B7 = image.select('B7')
        output_path = os.path.join(path_download, f'B7_{date}.tif')
        geemap.ee_export_image(image_B7, filename=output_path, scale=20, crs='EPSG:3857', region=roi)
        
        # B8 band with 10m resolution
        image_B8 = image.select('B8')
        output_path = os.path.join(path_download, f'B8_{date}.tif')
        geemap.ee_export_image(image_B8, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # B8A band with 20m resolution
        image_B8A = image.select('B8A')
        output_path = os.path.join(path_download, f'B8A_{date}.tif')
        geemap.ee_export_image(image_B8A, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B9 band with 60m resolution
        image_B9 = image.select('B9')
        output_path = os.path.join(path_download, f'B9_{date}.tif')
        geemap.ee_export_image(image_B9, filename=output_path, scale=60, crs='EPSG:3857', region=roi)

        # B11 band with 20m resolution
        image_B11 = image.select('B11')
        output_path = os.path.join(path_download, f'B11_{date}.tif')
        geemap.ee_export_image(image_B11, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # B12 band with 20m resolution
        image_B12 = image.select('B12')
        output_path = os.path.join(path_download, f'B12_{date}.tif')
        geemap.ee_export_image(image_B12, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # SCL band with 20m resolution
        image_SCL = image.select('SCL')
        output_path = os.path.join(path_download, f'SCL_{date}.tif')
        geemap.ee_export_image(image_SCL, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # TCI_R band with 10m resolution
        image_TCI_R = image.select('TCI_R')
        output_path = os.path.join(path_download, f'TCI_R_{date}.tif')
        geemap.ee_export_image(image_TCI_R, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # TCI_G band with 10m resolution
        image_TCI_G = image.select('TCI_G')
        output_path = os.path.join(path_download, f'TCI_G_{date}.tif')
        geemap.ee_export_image(image_TCI_G, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # TCI_B band with 10m resolution
        image_TCI_B = image.select('TCI_B')
        output_path = os.path.join(path_download, f'TCI_B_{date}.tif')
        geemap.ee_export_image(image_TCI_B, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # MSK_CLDPRB band with 20m resolution
        image_MSK_CLDPRB = image.select('MSK_CLDPRB')
        output_path = os.path.join(path_download, f'MSK_CLDPRB_{date}.tif')
        geemap.ee_export_image(image_MSK_CLDPRB, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # QA10 band with 10m resolution
        image_QA10 = image.select('QA10')
        output_path = os.path.join(path_download, f'QA10_{date}.tif')
        geemap.ee_export_image(image_QA10, filename=output_path, scale=10, crs='EPSG:3857', region=roi)

        # QA20 band with 20m resolution
        image_QA20 = image.select('QA20')
        output_path = os.path.join(path_download, f'QA20_{date}.tif')
        geemap.ee_export_image(image_QA20, filename=output_path, scale=20, crs='EPSG:3857', region=roi)

        # QA60 band with 60m resolution
        image_QA60 = image.select('QA60')
        output_path = os.path.join(path_download, f'QA60_{date}.tif')
        geemap.ee_export_image(image_QA60, filename=output_path, scale=60, crs='EPSG:3857', region=roi)

    print("Download completed!")

if __name__ == "__main__":
    download_dataset(roig, start_day, end_day,save_dir)
