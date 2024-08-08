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

CLASS_NAMES = ['Corn', 'Cotton', 'Rice', 'Sorghum', 'Soybeans', 'Winter Wheat', 
               'Dbl Crop WinWht/Soybeans', 'Other Hay/Non Alfalfa', 'Sod/Grass Seed', 
               'Fallow/Idle Cropland', 'Grapes', 'Pecans', 'Open Water', 'Developed/Open Space', 
               'Developed/Low Intensity', 'Developed/Med Intensity', 'Developed/High Intensity', 
               'Barren', 'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Shrubland', 'Grassland/Pasture', 
               'Woody Wetlands', 'Herbaceous Wetlands', 'Dbl Crop Corn/Soybeans', 'other']

color_map = np.array([
    [160, 160, 160],  # Light Gray - Corn
    [10, 128, 10],  # Dark Green - Cotton
    [10, 10, 128],  # Dark Blue - Rice
    [128, 128, 50],  # Olive variant - Sorghum
    [50, 128, 128],  # Dark Cyan variant - Soybeans
    [128, 50, 128],  # Purple variant - Winter Wheat
    [192, 192, 192],  # Silver - Dbl Crop WinWht/Soybeans
    [128, 0, 0],  # Maroon - Other Hay/Non Alfalfa
    [128, 128, 50],  # Olive variant - Sod/Grass Seed
    [0, 128, 0],  # Dark Green - Fallow/Idle Cropland
    [128, 50, 128],  # Purple variant - Grapes
    [0, 128, 128],  # Teal - Pecans
    [0, 0, 128],  # Navy - Open Water
    [255, 165, 0],  # Orange - Developed/Open Space
    [255, 192, 203],  # Pink - Developed/Low Intensity
    [165, 42, 42],  # Brown - Developed/Med Intensity
    [255, 105, 180],  # Hot Pink - Developed/High Intensity 
    [0, 255, 127],  # Spring Green - Barren
    [70, 130, 180],  # Steel Blue - Deciduous Forest
    [255, 105, 180],  # Hot Pink - Evergreen Forest X (ACTTUALLY) 19
    [255, 69, 0],  # Red-Orange - Mixed Forest (X) PREDICT 20
    [0, 191, 255],  # Deep Sky Blue - Shrubland 
    [135, 206, 250],  # Light Sky Blue - Grassland/Pasture
    [219, 112, 147],  # Pale Violet Red - Woody Wetlands
    [138, 43, 226],  # Blue Violet - Herbaceous Wetlands
    [75, 0, 130],  # Indigo - Dbl Crop Corn/Soybeans
    [144, 238, 144]  # Light Green - other
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

def visualize_inference_ground_truth(output, ground_truth,output_vis):
    # Get unique classes present in the image

    print("pred ", np.unique(output))
    print("ground truth ", np.unique(ground_truth))


    num_classes = len(CLASS_NAMES)
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
    for idx, class_idx in enumerate(unique_classes):
        color = color_map[class_idx]
        ax[1, 1].add_patch(plt.Rectangle((0, len(unique_classes) - idx - 1), 1.2, 1.2, color=np.array(color) / 255.0))  # Increase the width and height of the rectangle
        ax[1, 1].text(2, len(unique_classes) - idx - 0.5, f"{CLASS_NAMES[class_idx]}", va='center', ha='left', fontsize=12)  # Increase the fontsize and adjust the position

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
def inference_AR(config_file,input_dir, output_vis, ground_truth_path=None):
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
        
    crop_types_inference = inference(net, eval_dataloader,device,output_vis)
    visualize_inference_results(crop_types_inference,output_vis)

    if ground_truth_path:
        print("Visualizing compare with ground truth")
        with rasterio.open(ground_truth_path) as src:
            ground_truth = src.read(1)  # Read the first band
        # ground_truth[ground_truth == 19] = 20 # just test and fix bug

        visualize_inference_ground_truth(crop_types_inference, ground_truth,output_vis)


if __name__ == '__main__':
    config_file = 'configs/Arkansas/TSViT_AR24.yaml'
    input_dir = '/home/vuonghn/research/dataset/satellite/arkansas/preprocessed_data/preprocessed_data_aKhoa'
    output_vis = './output/'
    ground_truth = '/home/vuonghn/research/dataset/satellite/arkansas/org_maral/cdl/rgb_27classes_cdl.tif'
    inference_AR(config_file, input_dir, output_vis, ground_truth)


