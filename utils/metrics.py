import torch

# =========================
# IoU
# =========================
def compute_iou(preds, masks, num_classes=4, ignore_bg=True):

    preds = preds.view(-1)
    masks = masks.view(-1)

    start = 1 if ignore_bg else 0
    ious = []

    for cls in range(start, num_classes):

        pred_c = preds == cls
        mask_c = masks == cls

        inter = (pred_c & mask_c).sum().item()
        union = (pred_c | mask_c).sum().item()

        if union > 0:
            ious.append(inter / union)

    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


# =========================
# F1 SCORE
# =========================
def compute_f1(preds, masks, num_classes=4, ignore_bg=True):

    preds = preds.view(-1)
    masks = masks.view(-1)

    start = 1 if ignore_bg else 0
    f1s = []

    for cls in range(start, num_classes):

        pred_c = preds == cls
        mask_c = masks == cls

        tp = (pred_c & mask_c).sum().item()
        fp = (pred_c & ~mask_c).sum().item()
        fn = (~pred_c & mask_c).sum().item()

        denom = (2 * tp + fp + fn)

        if denom > 0:
            f1s.append((2 * tp) / (denom + 1e-6))

    return sum(f1s) / len(f1s) if len(f1s) > 0 else 0.0


# =========================
# DICE SCORE
# =========================
def compute_dice(preds, masks, num_classes=4, ignore_bg=True):

    preds = preds.view(-1)
    masks = masks.view(-1)

    start = 1 if ignore_bg else 0
    dices = []

    for cls in range(start, num_classes):

        pred_c = preds == cls
        mask_c = masks == cls

        inter = (pred_c & mask_c).sum().item()
        total = pred_c.sum().item() + mask_c.sum().item()

        if total > 0:
            dices.append((2 * inter) / (total + 1e-6))

    return sum(dices) / len(dices) if len(dices) > 0 else 0.0