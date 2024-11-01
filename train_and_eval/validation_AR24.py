import sys
import os
sys.path.insert(0, os.getcwd())

import json
from tqdm import tqdm
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models import get_model
from utils.lr_scheduler import build_scheduler
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders, get_loss_data_input
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries

def evaluate_model(net, dataloaders, config, device, lin_cls=False):
    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        predicted_all, labels_all, losses_all = [], [], []
        net.eval()
        
        with torch.no_grad():
            for step, sample in enumerate(tqdm(evalloader)):
                labels = sample['labels'].to(device).squeeze(-1)
                logits = net(sample['inputs'].to(device)).permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)

                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth

                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())

        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(
            predicted=predicted_classes, labels=target_classes, n_classes=num_classes
        )

        un_labels, class_loss = get_per_class_loss(losses, target_classes)
        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        
        print("Mean (micro) Evaluation metrics:")
        print(f"Loss: {losses.mean():.7f}, IOU: {micro_IOU:.4f}/{macro_IOU:.4f}, "
              f"Accuracy: {micro_acc:.4f}/{macro_acc:.4f}, Precision: {micro_precision:.4f}/{macro_precision:.4f}, "
              f"Recall: {micro_recall:.4f}/{macro_recall:.4f}, F1: {micro_F1:.4f}/{macro_F1:.4f}")
        
        return un_labels, {
            "macro": {
                "Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU
            },
            "micro": {
                "Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU
            },
            "class": {
                "Loss": class_loss, "Accuracy": eval_metrics['class'][0],
                "Precision": eval_metrics['class'][1], "Recall": eval_metrics['class'][2],
                "F1": eval_metrics['class'][3], "IOU": eval_metrics['class'][4]
            }
        }

    # Initialize settings from configuration
    lr = float(config['SOLVER']['lr_base'])
    local_device_ids = config['local_device_ids']
    loss_input_fn = get_loss_data_input(config)
    loss_fn = {
        'all': get_loss(config, device, reduction=None),
        'mean': get_loss(config, device, reduction="mean")
    }

    if config['CHECKPOINT']["load_from_checkpoint"]:
        load_from_checkpoint(net, config['CHECKPOINT']["load_from_checkpoint"], device=device)
    
    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)
    
    eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
    print("IoU per class:", eval_metrics[1]['class']['IOU'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str, help='gpu ids to use')
    parser.add_argument('--lin', action='store_true', help='train linear classifier only')
    args = parser.parse_args()
    
    config = read_yaml(args.config)
    device_ids = [int(d) for d in args.device.split(',')]
    device = get_device(device_ids, allow_cpu=False)
    config['local_device_ids'] = device_ids

    with open(config['DATASETS']['class_config'], 'r') as file:
        arkansas_data = yaml.safe_load(file)
    config['MODEL']['num_classes'] = len(arkansas_data['classes'])

    dataloaders = get_dataloaders(config)
    net = get_model(config, device)
    evaluate_model(net, dataloaders, config, device)

# python train_and_eval/validation_AR24.py --config configs/Arkansas/TSViT_AR23_infer.yaml