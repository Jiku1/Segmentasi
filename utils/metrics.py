import cv2
import numpy as np
import torch


# =====================================================
# SAFE FLATTEN
# =====================================================
def flatten_tensor(x):

    return x.contiguous().reshape(-1)


# =====================================================
# SAFE DIVISION
# =====================================================
def safe_divide(a, b):

    return a / (b + 1e-6)


# =====================================================
# GENERIC PER-CLASS METRIC
# =====================================================
def compute_per_class_metric(
    preds,
    masks,
    metric_type="iou",
    num_classes=4,
    ignore_bg=True
):

    preds = flatten_tensor(preds)
    masks = flatten_tensor(masks)

    start = 1 if ignore_bg else 0

    scores = []

    for cls in range(start, num_classes):

        pred_c = preds == cls
        mask_c = masks == cls

        tp = (pred_c & mask_c).sum().item()
        fp = (pred_c & ~mask_c).sum().item()
        fn = (~pred_c & mask_c).sum().item()

        # ---------------------------------------------
        # IoU
        # ---------------------------------------------
        if metric_type == "iou":

            union = tp + fp + fn

            if union > 0:

                score = safe_divide(tp, union)

                scores.append(score)

        # ---------------------------------------------
        # F1 / DICE
        # ---------------------------------------------
        elif metric_type in ["f1", "dice"]:

            denom = (2 * tp) + fp + fn

            if denom > 0:

                score = safe_divide(
                    2 * tp,
                    denom
                )

                scores.append(score)

    return (
        sum(scores) / len(scores)
        if len(scores) > 0
        else 0.0
    )


# =====================================================
# IoU
# =====================================================
def compute_iou(
    preds,
    masks,
    num_classes=4,
    ignore_bg=True
):

    return compute_per_class_metric(
        preds=preds,
        masks=masks,
        metric_type="iou",
        num_classes=num_classes,
        ignore_bg=ignore_bg
    )


# =====================================================
# F1 SCORE
# =====================================================
def compute_f1(
    preds,
    masks,
    num_classes=4,
    ignore_bg=True
):

    return compute_per_class_metric(
        preds=preds,
        masks=masks,
        metric_type="f1",
        num_classes=num_classes,
        ignore_bg=ignore_bg
    )


# =====================================================
# DICE SCORE
# =====================================================
def compute_dice(
    preds,
    masks,
    num_classes=4,
    ignore_bg=True
):

    return compute_per_class_metric(
        preds=preds,
        masks=masks,
        metric_type="dice",
        num_classes=num_classes,
        ignore_bg=ignore_bg
    )


# =====================================================
# BOUNDARY EXTRACTION (Perbaikan Akurasi Multi-Kelas)
# =====================================================
def mask_to_boundary(mask):
    """
    Mengekstrak garis tepi (kontur) objek menggunakan Morfologi Gradient.
    Sangat akurat untuk kelas objek kecil/tipis pada segmentasi korosi.
    """
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    
    # Menggunakan MORPH_GRADIENT untuk mendapatkan outline tepi setebal 1 piksel
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    
    return boundary



# =====================================================
# BOUNDARY F1 SCORE
# =====================================================
def compute_boundary_f1(
    preds,
    masks,
    num_classes=4,
    ignore_bg=True
):

    preds = preds.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    batch_scores = []

    for pred, gt in zip(preds, masks):

        class_scores = []

        start = 1 if ignore_bg else 0

        for cls in range(start, num_classes):

            pred_bin = (pred == cls).astype(np.uint8)
            gt_bin = (gt == cls).astype(np.uint8)

            pred_boundary = mask_to_boundary(pred_bin)
            gt_boundary = mask_to_boundary(gt_bin)

            tp = np.logical_and(
                pred_boundary,
                gt_boundary
            ).sum()

            fp = np.logical_and(
                pred_boundary,
                np.logical_not(gt_boundary)
            ).sum()

            fn = np.logical_and(
                np.logical_not(pred_boundary),
                gt_boundary
            ).sum()

            denom = (2 * tp) + fp + fn

            if denom > 0:

                bf1 = safe_divide(
                    2 * tp,
                    denom
                )

                class_scores.append(bf1)

        if len(class_scores) > 0:

            batch_scores.append(
                np.mean(class_scores)
            )

    return (
        float(np.mean(batch_scores))
        if len(batch_scores) > 0
        else 0.0
    )


# =====================================================
# DEGRADATION SLOPE
# =====================================================
def compute_degradation_slope(scores):

    """
    scores example:
    [0.85, 0.82, 0.78, 0.71, 0.65]
    """

    if len(scores) < 2:
        return 0.0

    x = np.arange(len(scores))

    slope = np.polyfit(
        x,
        scores,
        1
    )[0]

    return float(slope)