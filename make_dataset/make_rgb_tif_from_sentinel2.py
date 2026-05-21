import rasterio
import numpy as np
import os

#read tif whose name contains 'B2', 'B3', 'B4'
def read_rgb_tif(tif_dir):
    red, green, blue = None, None, None
    for path in tif_dir:
        if 'TCI_R' in path:
            with rasterio.open(path) as src:
                red = src.read(1)
                profile = src.profile.copy()
        elif 'TCI_G' in path:
            with rasterio.open(path) as src:
                green = src.read(1)
        elif 'TCI_B' in path:
            with rasterio.open(path) as src:
                blue = src.read(1)
    if red is not None and green is not None and blue is not None:
        arr_rgb = np.stack((red, green, blue), axis=-1)
    else:
        arr_rgb = None
        profile = None
    
    return arr_rgb, profile

def write_rgb_tif(output_path, rgb_image, profile=None):
    out_profile = profile if profile is not None else {}
    out_profile.update({
        'height': rgb_image.shape[0],
        'width': rgb_image.shape[1],
        'count': 3,
        'dtype': rgb_image.dtype,
        'driver': 'GTiff'
    })
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        for i in range(3):
            dst.write(rgb_image[:, :, i], i + 1)

def write_rgb_png(output_path, rgb_image):
    from PIL import Image
    img = Image.fromarray(rgb_image)
    img.save(output_path)
    
def main():
    # Example usage
    base_dir = '/home/yikebe/AR_sentinel2/2019_AR/'
    for sub_dir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, sub_dir) # 0_0, 0_1, ...
        if not os.path.isdir(subdir_path):
            continue
        for tif_dir in os.listdir(subdir_path): #2019-01-06, 2019-01-11, ...
            tif_dir_path = os.path.join(subdir_path, tif_dir)
            # print(f'Processing directory: {tif_dir_path}')
            if not os.path.isdir(tif_dir_path):
                continue
            tif_files = [os.path.join(tif_dir_path, f) for f in os.listdir(tif_dir_path) if f.endswith('.tif')]
            rgb_image, profile = read_rgb_tif(tif_files)
            if rgb_image is not None:
                out_dir = "/home/yikebe/research/FastDiffSR/sentinel_vr2/"
                out_dir = os.path.join(out_dir, sub_dir)
                out_dir = os.path.join(out_dir, tif_dir)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                output_path = os.path.join(out_dir, f'RGB_image_{tif_dir}.tif')
                write_rgb_tif(output_path, rgb_image, profile)
                # write_rgb_png(output_path.replace('.tif', '.png'), rgb_image)
                print(f'Written RGB image to: {output_path}')

if __name__ == "__main__":
    main()