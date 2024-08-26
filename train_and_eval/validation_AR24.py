import sys
import os
import json
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input

CLASS_NAMES = ['Corn', 'Cotton', 'Rice', 'Sorghum', 'Soybeans', 'Winter Wheat', 
               'Dbl Crop WinWht/Soybeans', 'Other Hay/Non Alfalfa', 'Sod/Grass Seed', 
               'Fallow/Idle Cropland', 'Grapes', 'Pecans', 'Open Water', 'Developed/Open Space', 
               'Developed/Low Intensity', 'Developed/Med Intensity', 'Developed/High Intensity', 
               'Barren', 'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Shrubland', 'Grassland/Pasture', 
               'Woody Wetlands', 'Herbaceous Wetlands', 'Dbl Crop Corn/Soybeans', 'other']

def evaluate_model(net, dataloaders, config, device, lin_cls=False):

    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(tqdm(evalloader)):
                # print("evaluate: sample ",sample.keys())
                
                # sample_inputs = sample['inputs'].to(device) # torch.Size([24, 60, 24, 24, 11])
                # print("evaluate: sample ",len(sample))
                # print("evaluate: sample inputs ",sample[0][])
                # exit()
                logits = net(sample['inputs'].to(device))
                # print("evaluate: logits ",logits.shape)
                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1) # torch.Size([24, 24, 24])
                # print("evaluate: predicted ",predicted.shape)
                ground_truth = loss_input_fn(sample, device)
                # print("evaluate: ground_truth ",ground_truth.shape)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth
                # print("evaluate: mask ",mask.shape)
                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())

        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f,\nunique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print(
            "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)

    print("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_input_fn = get_loss_data_input(config)
    
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)

    class_IOU = eval_metrics[1]['class']['IOU']
    macro_IOU = eval_metrics[1]['macro']['IOU']

    max_length = max(len(crop) for crop in CLASS_NAMES)+5
    max_i_length = len(str(len(CLASS_NAMES)))+5  # Get the length of the largest index
    print(f"{'ID':<{max_i_length}} {'CROP_TYPE':<{max_length}} {'IoU'}")
    print("-" * 70)
    

    for i, (crop, score) in enumerate(zip(CLASS_NAMES, class_IOU)):
        print(f"{i:<{max_i_length}} {crop:<{max_length}} {score}")
    print("-" * 70)
    print(f"{'mean IOU':<{max_length}} {np.mean(class_IOU)}")
    print(f"{'macro_IOU':<{max_length}} {np.mean(macro_IOU)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str,
                         help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                         help='train linear classifier only')

    args = parser.parse_args()
    config_file = args.config

    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    num_classes = len(json.load(open(config['DATASETS']['classnames'], 'r')))
    config['MODEL']['num_classes'] = num_classes

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    evaluate_model(net, dataloaders, config, device)


# python train_and_eval/validation_AR24.py --config configs/Arkansas/TSViT_AR24.yaml