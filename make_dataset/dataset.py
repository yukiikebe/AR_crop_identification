import numpy as np
import os
import types
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
import cv2

try:
    from skimage.transform import resize
except ImportError:
    resize = None

try:
    import torch.utils.data as data
    import torch
except ImportError:
    # The CLI patch/merge utilities do not require torch, so keep the module usable
    # even when only the lightweight image-processing dependencies are installed.
    data = types.SimpleNamespace(Dataset=object)
    torch = None

def adjust_brightness(image, alpha=1.0, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_blur(image, ksize=(3, 3)): # ksize shoould be odd number
    return cv2.GaussianBlur(image, ksize, 0)

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    
    # Salt noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

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
    if resize is None:
        raise ImportError("scikit-image is required for rescale_img_tiff")

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
