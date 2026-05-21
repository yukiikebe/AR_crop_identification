import rasterio
from rasterio.enums import Resampling
from PIL import Image
import numpy as np
import os
from tqdm import tqdm 

# img.max() --> find max value from whole tif images
def convert_tif_to_RGB(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        img = src.read()  # Read RGB bands
        img = np.moveaxis(img, 0, -1)  # Move channels to last dimension
        # import pdb; pdb.set_trace()
        
        img = (img - img.min()) / (img.max() - img.min()) * 255  # Normalize to 0-255
        img = img.astype(np.uint8)
        Image.fromarray(img).save(png_path, "PNG")

def __main__():
    # tif_dir = "/home/thanyu/data/analytic_8b_sr_udm2/Lower/2020_03_cf5b8a65-366f-4b70-946f-44166d008371/cf5b8a65-366f-4b70-946f-44166d008371/PSScene/"
    # tif_dir = "/home/thanyu/data/analytic_8b_sr_udm2/Lower/2020_07_fa1621d8-017b-4bb1-9d9f-030250057dab/fa1621d8-017b-4bb1-9d9f-030250057dab/PSScene"
    tif_dir = "/home/yikebe/research/FastDiffSR/FastDiffSR/dataset/Test_Potsdam_64_256/lr_64"
    png_dir = "/home/yikebe/research/FastDiffSR/Potsdam_test_png_lr"

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # matches = [f for f in os.listdir(tif_dir)
    #            if f.lower().endswith(("sr_8b_clip.tif", "sr_8b_clip.tiff"))]

    # for tif_file in tqdm(sorted(matches[6:]), desc="Converting TIFFs"):
    for tif_file in tqdm(os.listdir(tif_dir), desc="Converting TIFFs"):
        if not tif_file.lower().endswith((".tif", ".tiff")):
            continue
        png_file = os.path.join(png_dir, os.path.splitext(tif_file)[0] + ".png")
        convert_tif_to_RGB(os.path.join(tif_dir, tif_file), png_file)
        print(f"Converted {tif_file} to {png_file}")

if __name__ == "__main__":
    __main__()