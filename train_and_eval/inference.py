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
from data.Arkansas.dataloader import get_dataloader as get_arkansas_dataloader

from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
import pickle

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap


# device_ids = [0]

CLASS_NAMES = [
    'Background', 'Meadow', 'Soft winter wheat', 'Corn', 'Winter barley', 'Winter rapeseed',
    'Spring barley', 'Sunflower', 'Grapevine', 'Beet', 'Winter triticale', 'Winter durum wheat',
    'Fruits, vegetables, flowers', 'Potatoes', 'Leguminous fodder', 'Soybeans', 'Orchard', 
    'Mixed cereal', 'Sorghum', 'Void label',
]

class SingleSampleDataset(Dataset):
    """Dataset for a single satellite image sample."""

    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (string): Path to the single sample file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_path = file_path
        self.transform = transform

    def __len__(self):
        # This dataset always contains only one sample
        return 1

    def __getitem__(self, idx):
        # Load the sample
        with open(self.file_path, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')
        
        # Apply transform if any
        # print("before transform" , sample['img'].shape)
        if self.transform:
            sample = self.transform(sample)
        # print("after transform" , sample['inputs'].shape)
        return sample


def visualize_rgb(argmax_array, num_classes): 
    cm = matplotlib.cm.get_cmap('tab20')
    def_colors = cm.colors
    cus_colors = ['k'] + [def_colors[i] for i in range(1,20)]+['w']
    cmap = ListedColormap(colors=cus_colors, name='agri', N=21)

    # Convert class colors to a NumPy array for vectorized operations
    color_map = np.array([cmap(i) for i in range(num_classes)], dtype=np.float32)[:, :3] * 255.0
    color_map = color_map.astype(np.uint8)

    # Use advanced indexing to map class indices to RGB colors
    rgb_output = color_map[argmax_array]

    # Get the color codes
    color_codes = {i: tuple(color_map[i]) for i in range(num_classes)}

    return rgb_output, color_codes


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
            # labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            for i in range(outputs.shape[0]):
                h_idx, w_idx = [int(s) for s in patch_ids[i].split('_')]
                # print(h_idx, w_idx)
                all_outputs[h_idx:h_idx+24, w_idx:w_idx+24] = outputs[i]
    
    all_outputs = all_outputs[6:-6, 7:-7]
    num_classes = 19
    rgb_image, color_codes = visualize_rgb(all_outputs, num_classes)

    # Get unique classes present in the image
    unique_classes = np.unique(all_outputs)

    # Plotting the image and the color legend
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})

    # Plot the RGB image
    ax[0].imshow(rgb_image)
    ax[0].axis('off')
    ax[0].set_title('Visualized RGB Image')

    # Create the color legend
    ax[1].axis('off')
    for idx, class_idx in enumerate(unique_classes):
        color = color_codes[class_idx]
        ax[1].add_patch(plt.Rectangle((0, len(unique_classes) - idx - 1), 1, 1, color=np.array(color) / 255.0))
        ax[1].text(1.2, len(unique_classes) - idx - 0.5, f"{CLASS_NAMES[class_idx]}", va='center', ha='left', fontsize=12)

    ax[1].set_ylim(0, len(unique_classes))
    ax[1].set_xlim(0, 2)
    ax[1].set_title('Color Legend')

    plt.tight_layout()
    plt.show()

    # Save the combined image with the legend to a file
    fig.savefig(output_vis)
    print(f"Image with legend saved as visualized_rgb_with_legend.png")

def inference_AR(config_file,input_dir, output_vis):
    print("Inference Arkansas")
    
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
    inference(net, eval_dataloader,device,output_vis)


if __name__ == '__main__':
    config_file = 'configs/Arkansas/TSViT_inference.yaml'
    input_dir = '/home/vuonghn/research/dataset/satellite/arkansas/preprocessed_data/preprocessed_data_aKhoa'
    output_vis = './output/visualized_rgb_with_legend.png'
    inference_AR(config_file, input_dir, output_vis)


