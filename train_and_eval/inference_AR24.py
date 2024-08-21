import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
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


CLASS_NAMES = ['Corn', 'Cotton', 'Rice', 'Sorghum', 'Soybeans', 'Winter Wheat', 
               'Dbl Crop WinWht/Soybeans', 'Other Hay/Non Alfalfa', 'Sod/Grass Seed', 
               'Fallow/Idle Cropland', 'Grapes', 'Pecans', 'Open Water', 'Developed/Open Space', 
               'Developed/Low Intensity', 'Developed/Med Intensity', 'Developed/High Intensity', 
               'Barren', 'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Shrubland', 'Grassland/Pasture', 
               'Woody Wetlands', 'Herbaceous Wetlands', 'Dbl Crop Corn/Soybeans', 'other']

color_map = np.array([
    [ 31, 119, 180],
    [174, 199, 232],
    [255, 127,  14],
    [255, 187, 120],
    [ 44, 160,  44],
    [152, 223, 138],
    [214,  39,  40],
    [255, 152, 150],
    [148, 103, 189],
    [197, 176, 213],
    [140,  86,  75],
    [196, 156, 148],
    [227, 119, 194],
    [247, 182, 210],
    [127, 127, 127],
    [199, 199, 199],
    [188, 189,  34],
    [219, 219, 141],
    [ 23, 190, 207],
    [158, 218, 229],
    [ 31, 119, 180] , # Repeat starts here
    [174, 199, 232],
    [255, 127,  14],
    [255, 187, 120],
    [ 44, 160,  44],
    [152, 223, 138],
    [214,  39,  40],
])

def visualize_rgb(argmax_array, num_classes): 
    rgb_output = color_map[argmax_array]
    # Get the color codes
    color_codes = {i: tuple(color_map[i]) for i in range(num_classes)}
    return rgb_output, color_codes

def visualize_inference_results(output,output_vis):
    # Get unique classes present in the image
    num_classes = len(CLASS_NAMES)
    rgb_image, color_codes = visualize_rgb(output, num_classes)
    unique_classes = np.unique(output)

    # Plotting the image and the color legend
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})

    # Plot the RGB image
    ax[0].imshow(rgb_image)
    ax[0].axis('off')
    ax[0].set_title('Visualized RGB Image')

    # Create the color legend
    ax[1].axis('off')
    for idx, class_idx in enumerate(unique_classes):
        color = color_map[class_idx]
        ax[1].add_patch(plt.Rectangle((0, len(unique_classes) - idx - 1), 1, 1, color=np.array(color) / 255.0))
        ax[1].text(1.2, len(unique_classes) - idx - 0.5, f"{CLASS_NAMES[class_idx]}", va='center', ha='left', fontsize=14)

    ax[1].set_ylim(0, len(unique_classes))
    ax[1].set_xlim(0, 2)
    ax[1].set_title('Color Legend')

    plt.tight_layout()
    plt.show()

    # Save the combined image with the legend to a file
    fig.savefig(os.path.join(output_vis,"visualized_rgb_with_legend-TSViT_AR24.png"))
    print(f"Image with legend saved as visualized_rgb_with_legend.png")


def visualize_inference_ground_truth(output, ground_truth,output_vis, class_dict):
    # Get unique classes present in the image

    print("pred ", np.unique(output))
    print("ground truth ", np.unique(ground_truth))

    class_names = sorted(
        [(v['cdl_name'], v['remapped_id']) for v in class_dict.values()],
        key= lambda x: x[1]
    )
    class_names = [v[0] for v in class_names]

    num_classes = len(class_names)
    rgb_output, color_codes_output = visualize_rgb(output, num_classes)
    rgb_ground_truth, color_codes_ground_truth = visualize_rgb(ground_truth, num_classes)

    unique_classes = np.unique(np.concatenate([output, ground_truth]))

    # Plotting the images and the color legend
    fig, ax = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={'width_ratios': [4, 1]})

    # Plot the RGB output
    ax[0, 0].imshow(rgb_output)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Predicted Output')

    # Plot the RGB ground truth
    ax[1, 0].imshow(rgb_ground_truth)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Ground Truth')

    # Create the color legend for both
    ax[0, 1].axis('off')
    ax[1, 1].axis('off')
    for idx, class_name in enumerate(class_names):
        color = color_map[idx]
        ax[1, 1].add_patch(plt.Rectangle((0, len(unique_classes) - idx - 1), 1.2, 1.2, color=np.array(color) / 255.0))  # Increase the width and height of the rectangle
        ax[1, 1].text(2, len(unique_classes) - idx - 0.5, f"{class_name}", va='center', ha='left', fontsize=12)  # Increase the fontsize and adjust the position

    ax[1, 1].set_ylim(0, len(unique_classes))
    ax[1, 1].set_xlim(0, 3)  # Adjust the x limit to accommodate the larger text
    ax[1, 1].set_title('Color Legend')

    plt.tight_layout()
    plt.show()

    # Save the combined images with the legend to a file
    fig.savefig(output_vis+"/output_ground_truth_vis.png")
    print("Images with legend saved as output_ground_truth_vis.png")


def inference(net, dataloader,device,output_vis):
    net.eval()
    all_outputs = np.zeros((1920, 2856), dtype=np.uint8)
    with torch.no_grad():
        for sample, patch_ids in dataloader:
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
                # print(h_idx, w_idx)
                all_outputs[h_idx:h_idx+24, w_idx:w_idx+24] = outputs[i]
    # if np.any(all_outputs == 19):
    #     print("Value 19 is present in all_outputs")
    # else:
    #     print("Value 19 is not present in all_outputs") # this case
    all_outputs = all_outputs[6:-6, 7:-7]

    return all_outputs


def inference_AR(config_file, input_dir, output_vis, ground_truth_path=None):
    # print("Inference Arkansas input ", config_file,input_dir, output_vis, ground_truth_path)
    
    # exit()
    config = read_yaml(config_file)
    device_ids = config['DEVICE']['device_id']
    device = get_device(device_ids, allow_cpu=False)
    config['local_device_ids'] = device_ids
    model_config = config['MODEL']
    eval_config  = config['DATASETS']['eval']
    eval_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    eval_config['base_dir'] = input_dir
    eval_config['paths'] = list(glob(os.path.join(eval_config['base_dir'], '*.pickle')))

    class_dict = json.load(open(config['DATASETS']['classnames'], 'r'))
    config['MODEL']['num_classes'] = len(class_dict)

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
        load_from_checkpoint(net, checkpoint, partial_restore=False)
        
    crop_types_inference = inference(net, eval_dataloader, device, output_vis)
    #visualize_inference_results(crop_types_inference, output_vis)

    if ground_truth_path:
        from data.Arkansas.pre_processing import read_cdl_image
        ground_truth, classes = read_cdl_image(ground_truth_path)
        print("Visualizing compare with ground truth")
        # ground_truth[ground_truth == 19] = 20 # just test and fix bug

        visualize_inference_ground_truth(crop_types_inference, ground_truth, output_vis, class_dict)


if __name__ == '__main__':
    config_file = 'configs/Arkansas/TSViT_AR23.yaml'
    input_dir = '/home/khoavo/Desktop/workplace/satelite/AR23/pickle24x24'
    cdl_image_dir = "/home/khoavo/Desktop/workplace/satelite/cdl_arkansas/"
    output_vis = './output/'
    ground_truth = '/home/khoavo/Desktop/workplace/satelite/cdl_arkansas/rgb_27classes_cdl.tif'
    inference_AR(config_file, input_dir, output_vis, ground_truth)
