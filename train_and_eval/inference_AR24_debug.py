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
import yaml
from collections import defaultdict



# Load the YAML file
with open('configs/Arkansas/cdl.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Extract keys from crop_type and non_crop_type and convert them to integers
crop_labels = list(map(int, data['crop_type'].keys()))
non_crop_labels = list(map(int, data['non_crop_type'].keys()))


# print("crop_labels ", crop_labels)
# print("non_crop_labels ", non_crop_labels)
# exit()



color_map = np.array([
    [0, 0, 0],          #0
    [255, 0, 0],          #1
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

def crop_to_min_size(arr1, arr2):
    """
    Crop two arrays to the minimum dimensions of the two arrays.

    Parameters:
        arr1 (numpy.ndarray): The first array.
        arr2 (numpy.ndarray): The second array.

    Returns:
        cropped_arr1 (numpy.ndarray): The cropped first array.
        cropped_arr2 (numpy.ndarray): The cropped second array.
    """
    min_height = min(arr1.shape[0], arr2.shape[0])
    min_width = min(arr1.shape[1], arr2.shape[1])
    
    cropped_arr1 = arr1[:min_height, :min_width]
    cropped_arr2 = arr2[:min_height, :min_width]
    
    return cropped_arr1, cropped_arr2



def convert_to_crop_non_crop_labels(labels):
    """
    Convert the labels to crop/non-crop labels
    """
    # Define crop_labels (assuming crop_labels is defined elsewhere in your code)
    crop_labels_np = np.array(crop_labels)  # Replace with actual crop labels
    
    # Perform the conversion
    crop_mask = np.isin(labels, crop_labels_np)
    labels = np.zeros_like(labels)  # Set all values to 0
    labels[crop_mask] = 1  # Set crop labels to 1
    
    return labels.astype(np.uint8)

def calculate_IoU(pred, target):
    """
    Calculate the Intersection over Union (IoU) for a predicted and target image.
    The IoU is calculated as the area of intersection divided by the area of union.

    Parameters:
        pred (numpy.ndarray): The predicted image (flattened).
        target (numpy.ndarray): The target image (flattened).

    Returns:
        iou (float): The overall IoU score.
        iou_per_class (dict): A dictionary with class labels as keys and IoU scores as values.
    """

    num_classes = len(np.unique(target))
    
    # Calculate the overall intersection and union
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    
    # Calculate the overall IoU
    iou = np.sum(intersection) / np.sum(union)
    
    # Calculate IoU for each class
    iou_per_class = {}
    for cls in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        # Calculate the intersection and union for the current class
        class_intersection = np.logical_and(pred_mask, target_mask)
        class_union = np.logical_or(pred_mask, target_mask)

        # Calculate the IoU for the current class
        if np.sum(class_union) == 0:
            iou_per_class[cls] = 0.0
        else:
            iou_per_class[cls] = np.sum(class_intersection) / np.sum(class_union)
    
    return iou, iou_per_class

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

def visualize_inference(output, output_path, class_dict):
	# Get unique classes present in the image

	print("pred ", np.unique(output))
	output[output > len(color_map)] = 0
	
	class_names = []
	for v in class_dict.values():
		if v['remapped_id'] != 0 or v['class_name'] == 'Others':
			class_names.append((v['class_name'], v['remapped_id']))
	class_names = sorted(class_names, key= lambda x: x[1])
	class_names = [v[0] for v in class_names]

	num_classes = len(class_names)
	rgb_output = visualize_rgb(output, num_classes)

	# Adjust the width_ratios to give more space to the legend
	fig, ax = plt.subplots(1, 2, figsize=(9, 12), gridspec_kw={'width_ratios': [4, 1]})

	# Create a new subplot for the RGB output and ground truth, stacked vertically
	ax[0].imshow(rgb_output)
	ax[0].axis('off')
	ax[0].set_title('Predicted Output')

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


def visualize_inference_ground_truth(output, ground_truth, output_path, class_dict):
    # Get unique classes present in the image

    # print("pred ", np.unique(output))
    output[output > len(color_map)] = 0
    # print("ground truth ", np.unique(ground_truth))
    
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


# def inference(net, dataloader, groundtruth, device):
#     net.eval()

#     h, w = groundtruth.shape[:2]
#     pad_h = 24 - h % 24
#     pad_w = 24 - w % 24

#     all_outputs = np.zeros((h + pad_h, w + pad_w), dtype=np.uint8)
#     with torch.no_grad():
#         for sample, patch_ids in tqdm(dataloader):
#             inputs = sample['inputs'].to(device)
#             # print("inputs shape: ", inputs.shape)
#             # exit()
#             labels = sample['labels'].to(device)
#             logits = net(inputs)
#             logits = logits.permute(0, 2, 3, 1)
#             _, outputs = torch.max(logits.data, -1)

#             outputs = outputs.squeeze(0)
#             # labels = labels.squeeze(-1).squeeze(0) 
#             # print("Squeezed labels shape: ", labels_squeezed.shape)
#             # print(output.shape)
#             outputs = outputs.cpu().numpy() if outputs.is_cuda else outputs.numpy()
#             # outputs[outputs == 20] = 19 # just test and fix bug
#             # labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
#             for i in range(outputs.shape[0]):
#                 h_idx, w_idx = [int(s) for s in patch_ids[i].split('_')]
#                 all_outputs[h_idx:h_idx+24, w_idx:w_idx+24] = outputs[i]

#     return all_outputs


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
            # print("patch_ids ", patch_ids)

            inputs = sample['inputs'].to(device)
            # print("inputs shape: ", inputs.shape)
            # exit()
            labels = sample['labels'].to(device)
            labels = labels.squeeze(-1)
            labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            # labels  = convert_to_crop_non_crop_labels(labels)

            # print("infer: labels ", patch_ids ,labels)


            logits = net(inputs)
            logits = logits.permute(0, 2, 3, 1)
            _, outputs = torch.max(logits.data, -1)

            # predicted_sample = outputs.view(-1).cpu().numpy()
            # labels_sample = labels.view(-1).cpu().numpy()
            # if predicted_sample.shape != labels_sample.shape:
            #     raise ValueError(f"Shape mismatch: predicted_sample shape {predicted_sample.shape} and labels_sample shape {labels_sample.shape} must be the same.")

            # predicted_all.append(predicted_sample)
            # labels_all.append(labels_sample)

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
                labels_sample_cdl = groundtruth[h_idx:h_idx+24, w_idx:w_idx+24]
                # print("labels_sample_cdl ",patch_ids, labels_sample_cdl)
                # exit()  

                labels_data_loader = labels[i].copy()
                # print("labels_data_loader ",labels_data_loader)
                # print("labels_sample_cdl ",labels_sample_cdl)
                # exit()
                predicted_sample = outputs[i].copy()
                # Check shapes before flattening
                # if predicted_sample.shape != labels_sample_cdl.shape:
                #     print("h_idx, w_idx ", h_idx, w_idx, groundtruth.shape)
                    
                #     # raise ValueError(f"Shape mismatch: predicted_sample s
                #     # hape {predicted_sample.shape} and labels_sample_cdl shape {labels_sample_cdl.shape} must be the same.")
                # else:
                #     predicted_all.append(predicted_sample.flatten())
                #     labels_all.append(labels_sample_cdl.flatten())
                if predicted_sample.shape == labels_data_loader.shape:
                    predicted_all.append(predicted_sample.flatten())
                    labels_all.append(labels_data_loader.flatten())
    pred = np.concatenate(predicted_all)
    lab = np.concatenate(labels_all)
    # print("pred ", np.unique(pred), pred.shape)
    # print("lab ", np.unique(lab), lab.shape)      
    # Calculate IoU for each patch
    iou_per_class_patches = calculate_IoU(np.concatenate(predicted_all), np.concatenate(labels_all))
    print("IoU per class (patches):", iou_per_class_patches)
    # Calculate IoU for the whole segmentation
    crop_types_inference = all_outputs[:h, :w].flatten()
    cdl_image_2classes = groundtruth.flatten()
    iou_per_class_whole = calculate_IoU(crop_types_inference, cdl_image_2classes)
    print("IoU per class (whole):", iou_per_class_whole)
    # IoU per class (patches): (0.691369360079951, {0: 0.7371550957565407, 1: 0.691369360079951})
    # IoU per class (whole): (0.6908273032450578, {0: 0.7385602856340818, 1: 0.6908273032450578})
    return all_outputs


def inference_AR(config_file, raw_dir, input_dir, sub_region, output_dir, show_gt=True):
    output_path = os.path.join(output_dir, sub_region + '.png')
    if os.path.isfile(output_path):
        print("Output file already exists, skipping inference for %s" % sub_region)
        # return
    
    ground_truth_path = os.path.join(raw_dir, sub_region, 'cdl.tif')
    with rasterio.open(ground_truth_path) as src:
        cdl_image = src.read(1)

    
    # print("ground_truth_path: ", ground_truth_path)
    # print("cdl_image: ", np.unique(cdl_image))
    

    cdl_image_2classes = convert_to_crop_non_crop_labels(cdl_image)

    # print("cdl_image_2classes: ", np.unique(cdl_image_2classes))
    # ground_truth = plt.imread(ground_truth_path)[..., :3]
    # Save the ground truth as an image
    # ground_truth_output_path = os.path.join(output_dir, sub_region + '_ground_truth.png')
    # plt.imsave(ground_truth_output_path, ground_truth)

    # print("ground_truth: ", np.unique(ground_truth))

    # print("cdl_image_2classes: ", np.unique(cdl_image_2classes), cdl_image_2classes.shape)
    # print("ground_truth: ", np.unique(ground_truth), ground_truth.shape)
    cdl_image_color_map = color_map[cdl_image_2classes]
    # cdl_image_2classes = cdl_image
    # print("cdl_vis: ", np.unique(cdl_image_color_map), cdl_image_color_map.shape)
    # exit()
    config = read_yaml(config_file)
    device_ids = config['DEVICE']['device_id']
    # print("device_ids: ", device_ids)
    device = get_device(device_ids, allow_cpu=False)
    
    config['local_device_ids'] = device_ids
    model_config = config['MODEL']
    eval_config  = config['DATASETS']['eval']
    eval_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    eval_config['base_dir'] = os.path.join(input_dir, sub_region)
    eval_config['paths'] = list(glob(os.path.join(eval_config['base_dir'], '*.pickle')))
    eval_config['batch_size'] = 256

    class_dict = json.load(open(config['DATASETS']['classnames'], 'r'))

    config['MODEL']['num_classes'] =  len(class_dict)

    # print("eval_config['paths'] ",eval_config['paths'])
    # exit()

    eval_dataloader = get_arkansas_dataloader(
            paths=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=False),
            batch_size=eval_config['batch_size'], shuffle=False, return_paths=True,
            num_workers=eval_config['num_workers'])

 
    net = get_model(config, device)
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
 
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False, device=device)

    crop_types_inference = inference(net, eval_dataloader, cdl_image_2classes, device)

    crop_types_inference,cdl_image_2classes =  crop_to_min_size(crop_types_inference, cdl_image_2classes)
    IoU, IoU_classes = calculate_IoU(crop_types_inference, cdl_image_2classes)
    print("IoU: ", IoU)
    print("IoU_classes: ", IoU_classes)


    if show_gt:
        visualize_inference_ground_truth(crop_types_inference, cdl_image_color_map, output_path, class_dict)
    else:
        visualize_inference(crop_types_inference, output_path, class_dict)



if __name__ == '__main__':
    config_file = 'configs/Arkansas/TSViT_AR23_infer.yaml'
    raw_dir = '/home/vuonghn/research/dataset/arkansas/2023_all' # path for cdl
    input_dir = '/home/vuonghn/research/dataset/arkansas/AR23_preprocessed/pickle24x24'
    sub_region = '17_10'
    output_dir = './output/'
    inference_AR(config_file, raw_dir, input_dir, sub_region, output_dir, show_gt=True)
