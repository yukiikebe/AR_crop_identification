import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
from tqdm import tqdm
import torch
from glob import glob
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint
from data.Arkansas.dataloader_inference import get_dataloader as get_arkansas_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import rasterio
import json


color_map = np.array([
    [0, 0, 0],          #0
    [0, 0, 0],          #1
    [255, 210, 0],      #2
    [0, 168, 227],      #3
    [255, 158, 9],      #4
    [37, 111,  0],      #5
    [164, 111, 0],      #6
    [111, 111,  0],     #7
    [164, 241, 139],    #8
    [174, 255, 221],    #9
    [190, 190, 119],    #10
    [181, 111, 91],     #11
    [74, 111, 162],     #12
    [154, 154, 154],    #13
    [154, 154, 154],    #14
    [154, 154, 154],    #15
    [154, 154, 154],    #16
    [204, 190, 162],    #17
    [146, 204, 146],    #18
    [146, 204, 146],    #19
    [146, 204, 146],    #20
    [197, 213, 158],    #21
    [232, 255,190],     #22
    [125, 176, 176],    #23
    [125, 176, 176],    #24
    [255, 37, 37],      #25
    [111, 164, 0],      #26
    [111, 37, 0],       #27
    [255, 102, 102],    #28
    [0, 255, 255],      #29
    [255, 210, 0],      #30
    [255, 210, 0],      #31
    [255, 255, 0],      #32
    [111, 0, 72],       #33
    [125, 210, 255],    #34
    [159, 88, 136],     #35
    [255, 164, 225],    #36
    [0, 174, 74],       #37
    [37, 111, 0],       #38
    [232, 190, 255],    #39
    [164, 111, 0],      #40
    [215, 181, 107],    #41
    [0, 0, 153],        #42
    [241, 162, 119],    #43
    [255, 102, 102],    #44
    [255, 102, 102],    #45
    [111, 68, 136],     #46
    [172, 0, 123],      #47
    [255, 142, 170],    #48
    [213, 158, 187],    #49
    [83, 255, 0],       #50
    [255, 204, 102],    #51
    [164, 111, 0],      #52
    [255, 102, 102],    #53
    [221, 164, 9],      #54
    [0, 174, 74],       #55
    [225, 0, 123],      #56
    [176, 125, 255],    #57
    [227, 111, 37],     #58
    [111, 37, 0],       #59
    [255, 142, 170],    #60
])

def pad_images_to_same_size(img1, img2):
    """
    Pads the smaller of the two images (img1 and img2) so that they both have the same dimensions.

    Parameters:
        img1 (numpy.ndarray): The first image (height, width, channels).
        img2 (numpy.ndarray): The second image (height, width, channels).

    Returns:
        padded_img1 (numpy.ndarray): The padded version of img1.
        padded_img2 (numpy.ndarray): The padded version of img2.
    """
    # Determine the difference in dimensions
    height_diff = img1.shape[0] - img2.shape[0]
    width_diff = img1.shape[1] - img2.shape[1]

    # Pad img2 to match the dimensions of img1, or vice versa
    if height_diff > 0:
        img2 = np.pad(img2, ((0, height_diff), (0, 0), (0, 0)), mode='constant')
    elif height_diff < 0:
        img1 = np.pad(img1, ((0, -height_diff), (0, 0), (0, 0)), mode='constant')

    if width_diff > 0:
        img2 = np.pad(img2, ((0, 0), (0, width_diff), (0, 0)), mode='constant')
    elif width_diff < 0:
        img1 = np.pad(img1, ((0, 0), (0, -width_diff), (0, 0)), mode='constant')

    return img1, img2


def visualize_rgb(argmax_array, num_classes): 
    rgb_output = color_map[argmax_array]
    return rgb_output


def visualize_inference_results(output,output_vis, class_dict):
    # Get unique classes present in the image
    class_names = sorted(
        [(v['class_name'], v['remapped_id']) for v in class_dict.values()],
        key= lambda x: x[1]
    )
    class_names = [v[0] for v in class_names]

    num_classes = len(class_names)
    rgb_image = visualize_rgb(output, num_classes)
    unique_classes = np.unique(output)

    # Plotting the image and the color legend
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})

    # Plot the RGB image
    ax[0].imshow(rgb_image)
    ax[0].axis('off')
    ax[0].set_title('Visualized RGB Image')

    # Create the color legend
    ax[1].axis('off')
    for idx, class_name in enumerate(class_names):
        color = color_map[idx]
        ax[1].add_patch(plt.Rectangle((0, len(unique_classes) - idx - 1), 1, 1, color=np.array(color) / 255.0))
        ax[1].text(1.2, len(unique_classes) - idx - 0.5, f"{class_name}", va='center', ha='left', fontsize=14)

    ax[1].set_ylim(0, len(unique_classes))
    ax[1].set_xlim(0, 2)
    ax[1].set_title('Color Legend')

    plt.tight_layout()
    plt.show()

    # Save the combined image with the legend to a file
    fig.savefig(os.path.join(output_vis,"visualized_rgb_with_legend-TSViT_AR24.png"))
    print(f"Image with legend saved as visualized_rgb_with_legend.png")


def visualize_inference_ground_truth(output, ground_truth, output_path, class_dict):
    # Get unique classes present in the image

    print("pred ", np.unique(output))
    output[output > len(color_map)] = 0
    print("ground truth ", np.unique(ground_truth))
    
    class_names = []
    for v in class_dict.values():
        if v['remapped_id'] != 0 or v['class_name'] == 'Others':
            class_names.append((v['class_name'], v['remapped_id']))
    class_names = sorted(class_names, key= lambda x: x[1])
    class_names = [v[0] for v in class_names]

    num_classes = len(class_names)
    rgb_output = visualize_rgb(output, num_classes)

    rgb_output, ground_truth = pad_images_to_same_size(rgb_output, ground_truth)

    # Adjust the width_ratios to give more space to the legend
    fig, ax = plt.subplots(1, 2, figsize=(18, 12), gridspec_kw={'width_ratios': [3, 1]})

    # Create a new subplot for the RGB output and ground truth, stacked vertically
    ax[0].imshow(np.vstack([rgb_output, np.zeros((100, rgb_output.shape[1], 3),dtype=np.uint8) + 255, ground_truth]))
    ax[0].axis('off')
    ax[0].set_title('Predicted Output (top) and Ground Truth (bottom)')

    # Create the color legend
    ax[1].axis('off')

    for idx, class_name in enumerate(class_names):
        color = color_map[idx]
        ax[1].add_patch(plt.Rectangle((0, len(class_names) - idx - 1), 1.2, 1.2, color=np.array(color) / 255.0))
        ax[1].text(2, len(class_names) - idx - 0.5, f"{class_name}", va='center', ha='left', fontsize=12)

    ax[1].set_ylim(0, len(class_names))
    ax[1].set_xlim(0, 3)
    ax[1].set_title('Color Legend')

    plt.tight_layout()
    plt.show()

    # Save the combined images with the legend to a file
    fig.savefig(output_path)


def inference(net, dataloader, groundtruth, device):
    net.eval()

    h, w = groundtruth.shape[:2]
    pad_h = 24 - h % 24
    pad_w = 24 - w % 24

    all_outputs = np.zeros((h + pad_h, w + pad_w), dtype=np.uint8)
    with torch.no_grad():
        for sample, patch_ids in tqdm(dataloader):
            inputs = sample['inputs'].to(device)
            # print("inputs shape: ", inputs.shape)
            # exit()
            labels = sample['labels'].to(device)
            logits = net(inputs)
            logits = logits.permute(0, 2, 3, 1)
            _, outputs = torch.max(logits.data, -1)

            outputs = outputs.squeeze(0)
            # labels = labels.squeeze(-1).squeeze(0) 
            # print("Squeezed labels shape: ", labels_squeezed.shape)
            # print(output.shape)
            outputs = outputs.cpu().numpy() if outputs.is_cuda else outputs.numpy()
            # outputs[outputs == 20] = 19 # just test and fix bug
            # labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            for i in range(outputs.shape[0]):
                h_idx, w_idx = [int(s) for s in patch_ids[i].split('_')]
                all_outputs[h_idx:h_idx+24, w_idx:w_idx+24] = outputs[i]

    return all_outputs


def inference_AR(config_file, raw_dir, input_dir, sub_region, output_dir):
    output_path = os.path.join(output_dir, sub_region + '.png')
    if os.path.isfile(output_path):
        return
    config = read_yaml(config_file)
    device_ids = config['DEVICE']['device_id']
    device = get_device(device_ids, allow_cpu=False)
    config['local_device_ids'] = device_ids
    model_config = config['MODEL']
    eval_config  = config['DATASETS']['eval']
    eval_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    eval_config['base_dir'] = os.path.join(input_dir, sub_region)
    eval_config['paths'] = list(glob(os.path.join(eval_config['base_dir'], '*.pickle')))
    eval_config['batch_size'] = 32

    class_dict = json.load(open(config['DATASETS']['classnames'], 'r'))
    config['MODEL']['num_classes'] = len(class_dict)

    # print("eval_config['paths'] ",eval_config['paths'])
    # exit()

    eval_dataloader = get_arkansas_dataloader(
            paths=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=False),
            batch_size=eval_config['batch_size'], shuffle=False, return_paths=True,
            num_workers=eval_config['num_workers'])

    ground_truth_path = os.path.join(raw_dir, sub_region, 'cdl.tif')
    ground_truth = plt.imread(ground_truth_path)[..., :3]

    net = get_model(config, device)
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)

    crop_types_inference = inference(net, eval_dataloader, ground_truth, device)

    visualize_inference_ground_truth(crop_types_inference, ground_truth, output_path, class_dict)


if __name__ == '__main__':
    config_file = 'configs/Arkansas/TSViT_AR23_infer.yaml'
    raw_dir = '/home/khoavo/Desktop/workplace/satelite/raw_arkansas/2023_all/'
    input_dir = '/home/khoavo/Desktop/workplace/satelite/AR23_all/pickle24x24/'
    sub_region = '12_12'
    output_dir = './output/'
    inference_AR(config_file, raw_dir, input_dir, sub_region, output_dir)
