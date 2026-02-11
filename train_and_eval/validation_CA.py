import sys
import os
sys.path.insert(0, os.getcwd())

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import pickle as pkl
from tqdm import tqdm
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint
from data import get_dataloaders, get_loss_data_input
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss, get_accuracy_topk, get_accuracy_per_class
from metrics.loss_functions import get_loss


# ─────────── Canonical class ordering + names ────────────
ALL_CLASSES = [
    12, 13,  3,  5, 51, 14, 37, 15, 16, 44,
    27, 38, 20, 33, 31, 40, 49, 17, 29, 46,
    54, 41, 42
]

# Original CDL id → human‑readable name
ORIG_CLASS_NAMES = {
    1:"Unknown", 12:"Corn/Sorghum", 13:"Alfalfa", 3:"Mixed pasture",
    5:"Misc grain", 51:"Rice", 14:"Wheat", 37:"Tomato", 15:"Misc grasses",
    16:"Native pasture", 44:"Cotton", 27:"Leafy greens", 38:"Onions",
    20:"Melons", 33:"Safflower", 31:"Strawberries", 40:"Carrots",
    49:"Sunflower", 17:"Potatoes", 29:"Bush berries", 46:"Sweet potatoes",
    54:"Sugar beets", 41:"Dry beans", 42:"Peppers",
}

# Compact index (0 = Other) → name
COMPACT_NAMES = ["Other"] + [ORIG_CLASS_NAMES[cid] for cid in ALL_CLASSES]

NUM_CLASSES = 24


def evaluate_model(net, dataloaders, config, device, lin_cls=False):
    def evaluate(net, evalloader, loss_fn, config):
        predicted_all, labels_all, losses_all = [], [], []
        logits_all = []
        net.eval()
        
        with torch.no_grad():
            for step, sample in enumerate(tqdm(evalloader)):
                logits = net(sample['inputs'].to(device)).permute(0, 2, 3, 1)
                logits = logits[..., :NUM_CLASSES]  # Ensure logits match num_classes
                _, predicted = torch.max(logits.data, -1)

                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth

                if mask is not None:
                    logits_all.append(logits.reshape(-1, NUM_CLASSES)[mask.view(-1)].cpu().numpy())
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    logits_all.append(logits.cpu().numpy().reshape(-1, NUM_CLASSES))
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())

        logits_all = np.concatenate(logits_all, axis=0)
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all).astype(np.int64)
        losses = np.concatenate(losses_all)

        cm = confusion_matrix(target_classes, predicted_classes, labels=np.arange(NUM_CLASSES))
        eval_metrics = get_classification_metrics(
            predicted=predicted_classes, labels=target_classes, n_classes=NUM_CLASSES
        )

        topk_accuracies = get_accuracy_topk(logits_all, target_classes, top_k=5)
        for k, v in topk_accuracies.items():
            print(f"{k} Accuracy: {v:.4f}")

        import pdb; pdb.set_trace()
        topk_percls_accuracies = get_accuracy_per_class(logits_all, target_classes, top_k=5)
        for cls_name, accs in topk_percls_accuracies.items():
            print('-' * 50)
            print(f"{cls_name} Accuracy: ")
            for k, v in accs.items():
                print(f"{k} Accuracy: {v:.4f}")
        with open('topk_per_cls_accuracies.pkl', 'wb') as f:
            pkl.dump(topk_percls_accuracies, f)

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
        }, cm

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
    
    _, eval_metrics, cm = evaluate(net, dataloaders['eval'], loss_fn, config)
    print("IoU per class:", eval_metrics['class']['IOU'])
    print("Acc per class:", eval_metrics['class']['Accuracy'])

    # ── Confusion Matrix plot ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize rows so each shows % distribution of a ground‑truth class
    cm_norm = np.zeros_like(cm, dtype=np.float64)
    cm_norm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=cm_norm, where=cm.sum(axis=1,keepdims=True)!=0)

    im = ax.imshow(cm_norm, cmap='YlGnBu')

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground‑truth class")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(COMPACT_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(COMPACT_NAMES, fontsize=6)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            pct = cm_norm[i, j]
            if pct < 0.01: continue
            color = "white" if pct > 0.3 else "black"
            ax.text(j, i, f"{pct*100:4.1f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row‑normalized share")
    plt.tight_layout()
    plt.savefig("confusion_matrix_CA.png", dpi=300)
    plt.close(fig)
    print("📊 confusion matrix saved → confusion_matrix_CA.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str, help='gpu ids to use')
    args = parser.parse_args()

    config = read_yaml(args.config)
    device_ids = [int(d) for d in args.device.split(',')]
    device = get_device(device_ids, allow_cpu=False)
    config['local_device_ids'] = device_ids

    with open(config['DATASETS']['class_config'], 'r') as file:
        arkansas_data = yaml.safe_load(file)
    config['MODEL']['num_classes'] = 26

    dataloaders = get_dataloaders(config)
    net = get_model(config, device)
    evaluate_model(net, dataloaders, config, device)

# python train_and_eval/validation_AR24.py --config configs/Arkansas/TSViT_AR23_infer.yaml