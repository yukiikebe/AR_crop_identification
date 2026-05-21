import rasterio
from rasterio.enums import Resampling
from PIL import Image
import numpy as np
import os
# img.max() --> find max value from whole tif images
def convert_tif_to_RGB(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        img = src.read()  # Read RGB bands
        # img = src.read([6,4,2])  # Read RGB bands 
        img = np.moveaxis(img, 0, -1)  # Move channels to last dimension
        # import pdb; pdb.set_trace()
        
        img = (img - img.min()) / (img.max() - img.min()) * 255  # Normalize to 0-255
        img = img.astype(np.uint8)
        Image.fromarray(img).save(png_path, "PNG")

def __main__():
    tif_file = "/home/yikebe/research/FastDiffSR/sentinel/RGB_image_2019-07-28.tif"
    # tif_file = "/home/yikebe/remotesensing_data/2024_08_29eb7eb5-5887-4f02-8133-83c85f0ce4cf/29eb7eb5-5887-4f02-8133-83c85f0ce4cf/PSScene/20240813_162753_05_2420_3B_AnalyticMS_SR_8b_clip.tif"
    # tif_file = "/home/thanyu/data/analytic_8b_sr_udm2/Lower/2020_03_cf5b8a65-366f-4b70-946f-44166d008371/cf5b8a65-366f-4b70-946f-44166d008371/PSScene/20200316_155442_43_2304_3B_AnalyticMS_SR_8b_clip.tif"
    png_file = "../debug/2019-07-28.png"

    if not os.path.exists(os.path.dirname(png_file)):
        os.makedirs(os.path.dirname(png_file))

    # convert_tif_to_png(tif_file, png_file)
    # planet_analytic_to_png(tif_file, png_file, bands=(6,4,2), max_dim=2048)
    convert_tif_to_RGB(tif_file, png_file)
    print(f"Converted {tif_file} to {png_file}")

if __name__ == "__main__":
    __main__()