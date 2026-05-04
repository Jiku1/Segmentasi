import torch
import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(pred, target, num_classes=4):

    pred = torch.argmax(pred, dim=1)

    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    # =========================
    # IoU SIMPLE
    # =========================
    iou = np.mean(pred == target)

    # =========================
    # DICE APPROX
    # =========================
    dice = 2 * np.sum(pred == target) / (len(pred) + len(target))

    # =========================
    # F1 SCORE 
    # =========================
    f1 = f1_score(target, pred, average="macro")

    return {
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(f1)
    }