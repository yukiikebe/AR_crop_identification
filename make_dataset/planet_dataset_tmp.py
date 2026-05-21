import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
from tqdm import tqdm
import rasterio
# import cv2
from rasterio.enums import Resampling
from tqdm import tqdm

# def adjust_brightness(image, alpha=1.0, beta=10):
#     return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# def add_blur(image, ksize=(3, 3)): # ksize shoould be odd number
#     return cv2.GaussianBlur(image, ksize, 0)

# def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
#     noisy_image = np.copy(image)
#     total_pixels = image.shape[0] * image.shape[1]
    
#     # Salt noise
#     num_salt = int(total_pixels * salt_prob)
#     salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
#     noisy_image[salt_coords[0], salt_coords[1]] = 255
    
#     # Pepper noise
#     num_pepper = int(total_pixels * pepper_prob)
#     pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
#     noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
#     return noisy_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def rescale_img_tiff(image, scale):
    """Rescale multi-band image while preserving range and dtype."""
    num_bands = image.shape[-1]
    resized = []
    
    for band in range(num_bands):
        resized[:, :, band] = resize(image[:, :, band], (resized.shape[0], resized.shape[1]), anti_aliasing=True, preserve_range=True)
    
    return resized

def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    #(th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    #img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    #info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        #img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            #img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            #img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug

def create_resized_patches(input_dir, hr_dir, lr_dir, patch_size=256, scale=4):
    """
    Create HR (256x256) and LR (56x56) patches from AID dataset.

    Args:
        input_dir (str): Path to original AID dataset.
        hr_dir (str): Output directory for HR patches.
        lr_dir (str): Output directory for LR patches.
        patch_size (int): Size of HR patches (default: 256).
        scale (int): Downscaling factor for LR patches (default: 4).
    """
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        if not os.path.isdir(category_path):
            continue

        output_hr_category_path = os.path.join(hr_dir, category)
        os.makedirs(output_hr_category_path, exist_ok=True)

        output_lr_category_path = os.path.join(lr_dir, category)
        os.makedirs(output_lr_category_path, exist_ok=True)

        for file_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            img_path = os.path.join(category_path, file_name)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Load the original image
                img = Image.open(img_path).convert('RGB')

                # Extract HR patch
                img_width, img_height = img.size
                if img_width < patch_size or img_height < patch_size:
                    continue  

                hr_patch, _ = get_patch(img, img, patch_size, scale)

                # Downscale HR patch to create LR patch
                lr_patch = rescale_img(hr_patch, scale=1/scale)

                # Save HR patch
                output_hr = os.path.join(output_hr_category_path, f"{file_name}")
                hr_patch.save(output_hr, format='PNG')

                # Save LR patch
                output_lr = os.path.join(output_lr_category_path, f"{file_name}")
                lr_patch.save(output_lr, format='PNG')

def create_resized_patches_from_tiff(input_dir, output_dir, patch_size=256, scale=4, overlap=0.5, black_threshold=50, max_black_ratio=0.5):
    """
    Create HR (256x256) and LR (56x56) patches from Orthophotos (Tiff images) dataset.
    It allows a 50% overlap between patches, but exclude patches if more than 50% of the patch is black.  

    Args:
        input_dir (str): Path to original Orthophotos (Tiff images) dataset.
        Option: 1 for tiff output, 2 for png output
        hr_dir (str): Output directory for HR patches.
        lr_dir (str): Output directory for LR patches.
        patch_size (int): Size of HR patches (default: 256).
        scale (int): Downscaling factor for LR patches (default: 4).
    """
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        
    # hr_dir = os.path.join(output_dir, "HR")
    # if not os.path.exists(hr_dir):
    #     os.makedirs(hr_dir)
    #     os.makedirs(os.path.join(hr_dir, "Farmland"))
    # lr_dir = os.path.join(output_dir, "LR")
    # if not os.path.exists(lr_dir):
    #     os.makedirs(lr_dir)
    #     os.makedirs(os.path.join(lr_dir, "Farmland"))
    
    # hr_dir_farmland = os.path.join(hr_dir, "Farmland")
    # lr_dir_farmland = os.path.join(lr_dir, "Farmland")
    
    hr_dir = os.path.join(output_dir, "HR")
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
    lr_dir = os.path.join(output_dir, "LR")
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    
    i = 0
    
    patch_count = 0
    
    for file_name in os.listdir(input_dir):
        str_i = str(i)
        hr_dir_farmland = os.path.join(hr_dir, str_i)
        if not os.path.exists(hr_dir_farmland):
            os.makedirs(hr_dir_farmland)
        lr_dir_farmland = os.path.join(lr_dir, str_i)
        if not os.path.exists(lr_dir_farmland):
            os.makedirs(lr_dir_farmland)
        i += 1
        img_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_name}")
        if file_name.lower().endswith(('.tif', '.tiff')):
            # Load the original image
            try:
                with rasterio.open(img_path) as src:
                    img_array = src.read([1, 2, 3])
                    img_array = np.moveaxis(img_array, 0, -1)  # Move channels to the last dimension
        
                    # Normalize the data to 0-255 if needed
                    if img_array.dtype != np.uint8:
                        img_min = img_array.min()
                        img_max = img_array.max()
                        img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            except Exception as e:
                error_msg = f"Error loading {file_name}: {e}"
                print(error_msg)
            
            print(f"Processing {file_name} with size {img_array.shape}")
            
            height, width, _ = img_array.shape
            step = int(patch_size * (1 - overlap))
            
            for y in range(0, height, step):
                for x in range(0, width, step):
                    # Extract the patch
                    # patch = img_array[y:y + patch_size, x:x + patch_size, :]
                    if y + patch_size > height or x + patch_size > width:
                        patch = np.zeros((patch_size, patch_size, 3), dtype=img_array.dtype)
                        patch[:min(height - y, patch_size), :min(width - x, patch_size), :] = img_array[y:min(y+patch_size, height), x:min(x+patch_size, width), :]
                    else:
                        patch = img_array[y:y + patch_size, x:x + patch_size, :]

                    if patch.shape[:2] != (patch_size, patch_size):
                        print(f"Error: Patch at ({x}, {y}) has wrong shape {patch.shape}")
                        continue
                    # Calculate the black pixel ratio
                    black_pixels = np.sum(np.all(patch <= black_threshold, axis=-1))
                    total_pixels = patch_size * patch_size
                    black_ratio = black_pixels / total_pixels

                    # Exclude patches with more than 50% black background
                    if black_ratio <= max_black_ratio:
                        patch_filename = f"patch_{patch_count:04d}.tif"
                        patch_path = os.path.join(hr_dir_farmland, patch_filename)

                        # Save the patch using rasterio
                        png_path = os.path.join(hr_dir_farmland, f"{patch_count:04d}.png")
                        patch_img = Image.fromarray(patch.astype('uint8'), 'RGB')  # Convert to PIL image
                        patch_img.save(png_path)
                        
                        lr_patch = rescale_img(patch_img, scale=1/scale)
                        # Save LR patch
                        output_lr = os.path.join(lr_dir_farmland, f"{patch_count:04d}.png")
                        lr_patch.save(output_lr, format='PNG')
                           
                        patch_count += 1

    print(f"Extracted {patch_count} patches to {output_dir}")

def create_resized_tiff_patches_from_tiff(input_dir, output_dir, patch_size=256, scale=4, overlap=0.5, black_threshold=50, max_black_ratio=0.5, sub_dir="AR", image_type="RGB"):
    """
    Create HR (256x256) and LR (56x56) patches from Orthophotos (Tiff images) dataset.
    It allows a 50% overlap between patches, but exclude patches if more than 50% of the patch is black.  

    Args:
        input_dir (str): Path to original Orthophotos (Tiff images) dataset.
        Option: 1 for tiff output, 2 for png output
        hr_dir (str): Output directory for HR patches.
        lr_dir (str): Output directory for LR patches.
        patch_size (int): Size of HR patches (default: 256).
        scale (int): Downscaling factor for LR patches (default: 4).
    """
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    hr_dir = os.path.join(output_dir, "HR")
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
        os.makedirs(os.path.join(hr_dir, sub_dir))
    lr_dir = os.path.join(output_dir, "LR")
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
        os.makedirs(os.path.join(lr_dir, sub_dir))
    
    hr_dir_farmland = os.path.join(hr_dir, sub_dir)
    lr_dir_farmland = os.path.join(lr_dir, sub_dir)
    
    patch_count = 0
    
    matches = [f for f in os.listdir(input_dir)
               if f.lower().endswith(("sr_8b_clip.tif", "sr_8b_clip.tiff"))]
    
    img_max = 0
    # Compute global min/max across all matching TIFF files
    # global_min = float('inf')
    # global_max = float('-inf')
    # for file_name in tqdm(sorted(matches), desc="Calculating global min/max"):
    #     img_path = os.path.join(input_dir, file_name)
    #     with rasterio.open(img_path) as src:
    #         if image_type == "RGB":
    #             band_indices = [6, 4, 2]
    #         else:
    #             band_indices = list(range(1, src.count + 1))
    #         for band in band_indices:
    #             stats = src.statistics(band)
    #             if stats.min < global_min:
    #                 global_min = stats.min
    #             if stats.max > global_max:
    #                 global_max = stats.max
    # img_min = global_min
    # img_max = global_max
    print(f"Found {len(matches)} SR_8b_clip.tif files")
    
    img_max = 26336
    img_min = 1

    for file_name in sorted(matches):
        img_path = os.path.join(input_dir, file_name)
        img_path = "/home/yikebe/remotesensing_data/2024_08_29eb7eb5-5887-4f02-8133-83c85f0ce4cf/29eb7eb5-5887-4f02-8133-83c85f0ce4cf/PSScene/20240813_162753_05_2420_3B_AnalyticMS_SR_8b_clip.tif"
        
        try:
            with rasterio.open(img_path) as src:
                if image_type == "RGB":
                    img_array = src.read([6, 4, 2])  # Read RGB bands
                else:
                    img_array = src.read()
                img_array = np.moveaxis(img_array, 0, -1)
                
                if img_array.dtype != np.uint8:
                    img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)

                num_bands = img_array.shape[-1]
                
                profile = src.profile  # Copy metadata
                profile.update(
                    driver="GTiff",
                    height=patch_size,
                    width=patch_size,
                    count=num_bands,
                    dtype=img_array.dtype
                )
        except Exception as e:
            error_msg = f"Error loading {file_name}: {e}"
            print(error_msg)
        
        height, width, _ = img_array.shape
        step = int(patch_size * (1 - overlap))
                            
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Extract the patch
                # patch = img_array[y:y + patch_size, x:x + patch_size, :]
                if y + patch_size > height or x + patch_size > width:
                    patch = np.zeros((patch_size, patch_size, num_bands), dtype=img_array.dtype)
                    patch[:min(height - y, patch_size), :min(width - x, patch_size), :] = img_array[y:min(y+patch_size, height), x:min(x+patch_size, width), :]
                else:
                    patch = img_array[y:y + patch_size, x:x + patch_size, :]

                if patch.shape[:2] != (patch_size, patch_size):
                    print(f"Error: Patch at ({x}, {y}) has wrong shape {patch.shape}")
                    continue
                
                # Calculate the black pixel ratio
                black_pixels = np.sum(np.all(patch <= black_threshold, axis=-1))
                total_pixels = patch_size * patch_size
                black_ratio = black_pixels / total_pixels
                # print("black_ratio:", black_ratio)

                # Exclude patches with more than 50% black background
                if black_ratio <= max_black_ratio:
                    patch_filename = f"{patch_count:04d}.tif"
                    patch_path = os.path.join(hr_dir_farmland, patch_filename)

                    # Save the patch using rasterio
                    with rasterio.open(
                        patch_path,
                        'w',
                        **profile
                    ) as dst:
                        if dst.shape[0] != patch_size or dst.shape[1] != patch_size:
                            print(f"Error: {patch_path} has wrong shape {dst.shape}")
                        for band in range(num_bands):
                            dst.write(patch[:, :, band], band + 1)
                            
                    # Save the patch as PNG
                    png_path = os.path.join(hr_dir_farmland, f"{patch_count:04d}.png")
                    # if patch.dtype != np.float32 and patch.dtype != np.float64:
                    #     # patch_rgb = patch[:, :, :3].astype('float32')  # Extract first three bands
                    #     # patch_rgb -= patch_rgb.min()  # Normalize to 0-255
                    #     # patch_rgb /= (patch_rgb.max() - patch_rgb.min() + 1e-5)  # Avoid division by zero
                    #     patch_rgb = patch[:, :, :3].astype('float32')
                    #     patch_rgb = np.clip(patch_rgb * 255, 0, 255).astype('uint8')
                    #     patch_rgb = (patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min() + 1e-5) * 255
                    #     patch_rgb = patch_rgb.astype('uint8')
                    # else:
                        # patch_rgb = (patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min() + 1e-5) * 255
                    print("path_rgb dtype:", patch.dtype)
                    print("patch min max:", patch.min(), patch.max())
                    patch_rgb = patch[:, :, :3].astype('uint8')  # Directly cast to uint8 if already float
                    patch_img = Image.fromarray(patch_rgb, 'RGB')  # Convert to PIL image
                    patch_img.save(png_path)
                                
                    patch_count += 1
        break
    print(f"Extracted {patch_count} patches to {output_dir}")

def merge_tiff_patches(input_dir, output_png_path, whole_image_size, patch_size=256):
    patch_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tif")])
    num_patches = len(patch_files)
    if whole_image_size[2] == 1:
        whole_image = np.zeros(whole_image_size, dtype=np.float32)
    else:
        whole_image = np.zeros(whole_image_size, dtype=np.uint16)
    if whole_image_size[2] > 3:
        whole_image_png = np.zeros((whole_image_size[0], whole_image_size[1], 3), dtype=np.uint8) 
        print("whole_image shape:", whole_image_png.shape)

    patch_idx = 0

    for y in range(0, whole_image_size[0], patch_size):
        for x in range(0, whole_image_size[1], patch_size):
            if patch_idx >= num_patches:
                break

            patch_path = os.path.join(input_dir, patch_files[patch_idx])

            with rasterio.open(patch_path) as patch:
                if whole_image_size[2] > 3:
                    patch_data = patch.read([1, 2, 3])  
                    patch_data = np.moveaxis(patch_data, 0, -1)
                pacth_data_tiff = patch.read()
                print("patch_data_tiff dtype:", pacth_data_tiff.dtype)
                patch_data_tiff = np.moveaxis(pacth_data_tiff, 0, -1)  

                if patch_data_tiff.shape[-1] == whole_image.shape[-1]:
                    whole_image[y:y + patch_size, x:x + patch_size, :] = patch_data_tiff
                else:
                    for band in range(patch_data_tiff.shape[-1]):
                        whole_image[y:y + patch_size, x:x + patch_size, band] = patch_data_tiff[:, :, band]
                if whole_image_size[2] > 3:
                    if patch_data.dtype != np.uint8:
                        img_min, img_max = patch_data.min(), patch_data.max()
                        if img_max > img_min:  
                            patch_data = ((patch_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            patch_data = np.zeros_like(patch_data, dtype=np.uint8)

                    whole_image_png[y:y + patch_size, x:x + patch_size, :] = patch_data  # PNG用

            patch_idx += 1

    #Save the merged image as TIFF
    output_tif_path = output_png_path.replace(".png", ".tif")
    print("Final image shape:", whole_image.shape)
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=whole_image.shape[0],
        width=whole_image.shape[1],
        count=whole_image.shape[2],
        dtype=whole_image.dtype
    ) as dst:
        for band in range(whole_image.shape[2]):
            dst.write(whole_image[:, :, band], band + 1)
            
    # Save the merged image as PNG
    if whole_image_size[2] > 3:
        whole_image_png = Image.fromarray(whole_image_png, 'RGB')
        whole_image_png.save(output_png_path)
        print("save png image")
        
def merge_png_patches(input_dir, output_png_path, whole_image_size, patch_size=256):
    patch_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    num_patches = len(patch_files)

    whole_image = np.zeros((whole_image_size[0], whole_image_size[1], 3), dtype=np.uint8)
    patch_idx = 0

    for y in range(0, whole_image_size[0], patch_size):
        for x in range(0, whole_image_size[1], patch_size):
            if patch_idx >= num_patches:
                break

            patch_path = os.path.join(input_dir, patch_files[patch_idx])
            patch = Image.open(patch_path).convert("RGB")
            patch_data = np.array(patch)

            if patch_data.shape[:2] != (patch_size, patch_size):
                print(f"Warning: Patch {patch_path} has unexpected size {patch_data.shape[:2]}")
                patch_data = np.array(patch.resize((patch_size, patch_size)))

            whole_image[y:y + patch_size, x:x + patch_size, :] = patch_data
            patch_idx += 1

    merged_image = Image.fromarray(whole_image, 'RGB')
    merged_image.save(output_png_path)
    print(f"Merged image saved to {output_png_path}")
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        name = self.hr_image_filenames[index]
        lr_name = name.replace('GT', 'LR')

        input = load_img(lr_name)

        input, target, = get_patch(input, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            input, target, _ = augment(input, target)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.hr_image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, HR_dir, LR_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in listdir(HR_dir) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        target = load_img(self.hr_image_filenames[index])
        name = self.hr_image_filenames[index]
        lr_name = name.replace('GT', 'LR')
        input = load_img(lr_name)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, name

    def __len__(self):
        return len(self.hr_image_filenames)

def main():
    # input_dir = "../AID/GT"
    # hr_dir = "../AID/HR"
    # if not os.path.exists(hr_dir):
    #     os.makedirs(hr_dir, exist_ok=True)
    # lr_dir = "../AID/LR"
    # if not os.path.exists(lr_dir):
    #     os.makedirs(lr_dir, exist_ok=True)
    # create_resized_patches(input_dir, hr_dir, lr_dir)
    
    # input_dir = "../Orthophotos"
    input_dir = "/home/yikebe/remotesensing_data/2024_08_29eb7eb5-5887-4f02-8133-83c85f0ce4cf/29eb7eb5-5887-4f02-8133-83c85f0ce4cf/PSScene"
    # input_dir = "../planet_dataset_tmp/HR/AR"
    
    output_dir = "../planet_dataset_tmp"
    # output_dir = "../planet_dataset_tmp_merge"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # create_resized_patches_from_tiff(input_dir, output_dir, scale=16) # make png images
    # create_resized_patches_from_tiff(input_dir, output_dir, scale=16, overlap=0)
    # create_resized_patches_from_tiff(input_dir, output_dir, scale=16, overlap=0, max_black_ratio=1.0)
    # create_resized_tiff_patches_from_tiff(input_dir, output_dir ,scale=16) # make tif images
    
    #no overlap and doesn't exclude black patches
    create_resized_tiff_patches_from_tiff(input_dir, output_dir, scale=4, overlap=0, black_threshold=50, max_black_ratio=0.5)
    
    #merge patches from tiff images
    # patch_size = 256
    # with rasterio.open("../Orthophotos/odm_orthophoto_20.tif") as dst:
    #     height, width, count = dst.height, dst.width, dst.count
    #     if height % patch_size != 0:
    #         height = (height // patch_size + 1) * patch_size
    #     if width % patch_size != 0:
    #         width = (width // patch_size + 1) * patch_size
    #     whole_size_image_size = (height, width, count)
    # input_tif_file = os.path.join(input_dir, "0000.tif")
    # with rasterio.open(input_tif_file) as src:
    #     count = src.count
    #     if whole_size_image_size[2] != count:
    #         whole_size_image_size = (whole_size_image_size[0], whole_size_image_size[1], count)
            
    # merge_tiff_patches(input_dir, output_dir + "/merged_farmland.png", whole_size_image_size, patch_size)
    
    #merge patches from png images
    # with rasterio.open("/home/yikebe/remotesensing_data/2024_08_29eb7eb5-5887-4f02-8133-83c85f0ce4cf/29eb7eb5-5887-4f02-8133-83c85f0ce4cf/PSScene/20240813_162753_05_2420_3B_AnalyticMS_SR_8b_clip.tif") as dst:
    #     height, width, count = dst.height, dst.width, dst.count
    #     if height % patch_size != 0:
    #         height = (height // patch_size + 1) * patch_size
    #     if width % patch_size != 0:
    #         width = (width // patch_size + 1) * patch_size
    #     whole_size_image_size = (height, width, 3)
    # print(whole_size_image_size)
    # merge_png_patches(input_dir, output_dir + "/merged_farmland.png", whole_size_image_size, patch_size)
    
main()