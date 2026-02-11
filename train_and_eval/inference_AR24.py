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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
import json
import yaml

# Load the classes for mapping the labels
with open('configs/Arkansas/arkansas_data.yaml', 'r') as file:
    arkansas_data = yaml.safe_load(file)
arkansas_classes = arkansas_data['classes']
colormaps = arkansas_data['colormaps']

class_mappings = {key: list(value.keys()) for key, value in arkansas_classes.items()}
colormaps = [value for _, value in colormaps.items()]
colormaps = np.array(colormaps)

classes = [k for k, v in class_mappings.items()]

def normalize_classes(labels):
    normalized_labels = np.zeros_like(labels)
    for new_class, original_classes in class_mappings.items():
        mask = np.isin(labels, np.array(list(original_classes)))
        normalized_labels[mask] = new_class
    return normalized_labels


def crop_to_min_size(arr1, arr2):
    min_height = min(arr1.shape[0], arr2.shape[0])
    min_width = min(arr1.shape[1], arr2.shape[1])
    cropped_arr1 = arr1[:min_height, :min_width]
    cropped_arr2 = arr2[:min_height, :min_width]
    return cropped_arr1, cropped_arr2


def calculate_IoU(pred, target):
    num_classes = len(np.unique(target))
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou = np.sum(intersection) / np.sum(union)
    iou_per_class = {}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        class_intersection = np.logical_and(pred_mask, target_mask)
        class_union = np.logical_or(pred_mask, target_mask)
        if np.sum(class_union) == 0:
            iou_per_class[cls] = 0.0
        else:
            iou_per_class[cls] = np.sum(class_intersection) / np.sum(class_union)
    return iou, iou_per_class

def pad_images_to_same_size(img1, img2):
    height_diff = img1.shape[0] - img2.shape[0]
    width_diff = img1.shape[1] - img2.shape[1]
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
    return colormaps[argmax_array]

def visualize_ground_truth(ground_truth, output_path, class_names):
    import pdb; pdb.set_trace()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})
    ax[0].imshow(ground_truth)
    ax[0].axis('off')
    ax[0].set_title('Ground Truth')

    for idx, class_name in enumerate(class_names):
        color = colormaps[idx]
        ax[1].add_patch(plt.Rectangle((0, len(class_names) - idx - 1), 1.2, 1.2, color=np.array(color) / 255.0))
        ax[1].text(2, len(class_names) - idx - 0.5, class_name, va='center', ha='left', fontsize=12)
    ax[1].set_ylim(0, len(class_names))
    ax[1].set_xlim(0, 3)
    ax[1].axis('off')
    ax[1].set_title('Color Legend')

    plt.tight_layout()
    plt.show()
    fig.savefig(output_path)

def visualize_inference(output, output_path, class_names):
    output[output > len(colormaps)] = 0
    num_classes = len(class_names)
    rgb_output = visualize_rgb(output, num_classes)
    fig, ax = plt.subplots(1, 2, figsize=(9, 12), gridspec_kw={'width_ratios': [4, 1]})
    ax[0].imshow(rgb_output)
    ax[0].axis('off')
    ax[0].set_title('Predicted Output')
    ax[1].axis('off')
    for idx, class_name in enumerate(class_names):
        color = colormaps[idx]
        ax[1].add_patch(plt.Rectangle((0, len(class_names) - idx - 1), 1.2, 1.2, color=np.array(color) / 255.0))
        ax[1].text(2, len(class_names) - idx - 0.5, f"{class_name}", va='center', ha='left', fontsize=12)
    ax[1].set_ylim(0, len(class_names))
    ax[1].set_xlim(0, 3)
    ax[1].set_title('Color Legend')
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path)

def visualize_inference_ground_truth(output, ground_truth, output_path, class_names):
    output[output > len(colormaps)] = 0
    num_classes = len(class_names)
    rgb_output = visualize_rgb(output, num_classes)
    rgb_output, ground_truth = pad_images_to_same_size(rgb_output, ground_truth)
    fig, ax = plt.subplots(1, 2, figsize=(18, 12), gridspec_kw={'width_ratios': [3, 1]})
    ax[0].imshow(np.vstack([rgb_output, np.zeros((100, rgb_output.shape[1], 3), dtype=np.uint8) + 255, ground_truth]))
    ax[0].axis('off')
    ax[0].set_title('Predicted Output (top) and Ground Truth (bottom)')
    ax[1].axis('off')
    for idx, class_name in enumerate(class_names):
        color = colormaps[idx]
        ax[1].add_patch(plt.Rectangle((0, len(class_names) - idx - 1), 1.2, 1.2, color=np.array(color) / 255.0))
        ax[1].text(2, len(class_names) - idx - 0.5, f"{class_name}", va='center', ha='left', fontsize=12)
    ax[1].set_ylim(0, len(class_names))
    ax[1].set_xlim(0, 3)
    ax[1].set_title('Color Legend')
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path)

def inference(net, dataloader, groundtruth, device):
    predicted_all = []
    labels_all = []
    net.eval()
    h, w = groundtruth.shape[:2]
    pad_h = 24 - h % 24
    pad_w = 24 - w % 24
    all_outputs = np.zeros((h + pad_h, w + pad_w), dtype=np.uint8)
    with torch.no_grad():
        for sample, patch_ids in tqdm(dataloader):
            inputs = sample['inputs'].to(device)
            labels = sample['labels'].to(device).squeeze(-1)
            labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            logits = net(inputs).permute(0, 2, 3, 1)
            _, outputs = torch.max(logits.data, -1)
            outputs = outputs.squeeze(0).cpu().numpy() if outputs.is_cuda else outputs.numpy()
            for i in range(outputs.shape[0]):
                h_idx, w_idx = [int(s) for s in patch_ids[i].split('_')]
                all_outputs[h_idx:h_idx+24, w_idx:w_idx+24] = outputs[i]
                labels_data_loader = labels[i].copy()
                predicted_sample = outputs[i].copy()
                if predicted_sample.shape == labels_data_loader.shape:
                    predicted_all.append(predicted_sample.flatten())
                    labels_all.append(labels_data_loader.flatten())
    pred = np.concatenate(predicted_all)
    lab = np.concatenate(labels_all)
    iou_per_class_patches = calculate_IoU(pred, lab)
    print("IoU per class (patches):", iou_per_class_patches)
    crop_types_inference = all_outputs[:h, :w].flatten()
    cdl_image_2classes = groundtruth.flatten()
    iou_per_class_whole = calculate_IoU(crop_types_inference, cdl_image_2classes)
    print("IoU per class (whole):", iou_per_class_whole)
    return all_outputs

def inference_AR(config_file, raw_dir, input_dir, sub_region, output_dir, show_gt=True, show_gt_only=False):
    output_path = os.path.join(output_dir, sub_region + '.png')
    if os.path.isfile(output_path):
        print("Output file already exists, skipping inference for %s" % sub_region)
        # return
    ground_truth_path = os.path.join(raw_dir, sub_region, 'cdl.tif')
    with rasterio.open(ground_truth_path) as src:
        cdl_image = src.read(1)
    #cdl_image_2classes = normalize_classes(cdl_image)
    #cdl_image_color_map = colormaps[cdl_image_2classes]
    cdl_image_color_map = cdl_image
    if show_gt_only:
        visualize_ground_truth(cdl_image_color_map, output_path, classes)
        return

    config = read_yaml(config_file)
    device_ids = config['DEVICE']['device_id']
    device = get_device(device_ids, allow_cpu=False)
    config['local_device_ids'] = device_ids
    model_config = config['MODEL']
    eval_config = config['DATASETS']['eval']
    eval_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    eval_config['base_dir'] = os.path.join(input_dir, sub_region)
    eval_config['paths'] = list(glob(os.path.join(eval_config['base_dir'], '*.pickle')))
    eval_config['batch_size'] = 256
    config['MODEL']['num_classes'] = len(classes)
    eval_dataloader = get_arkansas_dataloader(
        paths=eval_config['paths'], root_dir=eval_config['base_dir'],
        transform=PASTIS_segmentation_transform(model_config, is_training=False),
        batch_size=eval_config['batch_size'], shuffle=False, return_paths=True,
        num_workers=eval_config['num_workers']
    )
    net = get_model(config, device)
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False, device=device)
    crop_types_inference = inference(net, eval_dataloader, cdl_image_2classes, device)
    crop_types_inference, cdl_image_2classes = crop_to_min_size(crop_types_inference, cdl_image_2classes)
    IoU, IoU_classes = calculate_IoU(crop_types_inference, cdl_image_2classes)
    print("IoU:", IoU)
    print("IoU_classes:", IoU_classes)
    if show_gt:
        visualize_inference_ground_truth(crop_types_inference, cdl_image_color_map, output_path, classes)
    else:
        visualize_inference(crop_types_inference, output_path, classes)

if __name__ == '__main__':
    config_file = 'configs/Arkansas/TSViT_AR23_infer.yaml'
    raw_dir = '/home/vuonghn/research/code/Agriculture/arkansas/2023_all'
    input_dir = '/home/vuonghn/research/code/Agriculture/arkansas/AR23_preprocessed/pickle24x24'
    sub_region = '17_10'
    output_dir = './output/'
    inference_AR(config_file, raw_dir, input_dir, sub_region, output_dir, show_gt=True)
