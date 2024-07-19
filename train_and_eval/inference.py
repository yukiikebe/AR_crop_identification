import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input

from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
import pickle

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

def visualize_inference(output_np, labels_np):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(output_np)
    ax[0].set_title("Output")
    ax[1].imshow(labels_np)
    ax[1].set_title("Labels")
    
    # Save the figure
    plt.savefig('inference_visualization.png', bbox_inches='tight')

def inference(net, dataloader):
    net.eval()
    with torch.no_grad():
        for sample in dataloader:
            inputs = sample['inputs'].to(device)
            labels = sample['labels'].to(device)
            # print("Original labels shape: ", labels.shape)
            logits = net(inputs)
            logits = logits.permute(0, 2, 3, 1)
            _, output = torch.max(logits.data, -1)

    output = output.squeeze(0)
    labels = labels.squeeze(-1).squeeze(0) 
    # print("Squeezed labels shape: ", labels_squeezed.shape)
    # print(output.shape)
    output_np = output.cpu().numpy() if output.is_cuda else output.numpy()
    labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

    visualize_inference(output_np, labels_np)
    return output_np,labels_np


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str,
                        help='gpu ids to use')
    args = parser.parse_args()
    config_file = args.config
    config = read_yaml(config_file)
    model_config = config['MODEL']

    print("Loading config file: ", config_file)

    device_ids = [int(d) for d in args.device.split(',')]
    device = get_device(device_ids, allow_cpu=False)

    #load data from a sample
    transform = PASTIS_segmentation_transform(model_config, is_training=False)

    # Example usage
    file_path = '/home/vuonghn/research/code/Agriculture/DeepSatModels/datasets/PASTIS24/pickle24x24/30615_9.pickle'
    dataset = SingleSampleDataset(file_path,transform)
    dataloader = DataLoader(dataset, batch_size=1)

    net = get_model(config, device)
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)
    net.to(device)
    output,label = inference(net, dataloader)

    # print("output ", output)


