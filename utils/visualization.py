import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.nn.functional as F


# =========================
# COLOR MAP
# =========================
COLORS = {
    0: [0, 0, 0],        # background
    1: [0, 255, 0],      # Fair
    2: [255, 255, 0],    # Poor
    3: [255, 0, 0],      # Severe
}


# =========================
# MASK -> RGB
# =========================
def decode_mask(mask):

    h, w = mask.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLORS.items():
        rgb[mask == cls] = color

    return rgb


# =========================
# SAFE OVERLAY
# =========================
def overlay_mask(image, mask_rgb, alpha=0.5):

    # ensure same shape
    if image.shape[:2] != mask_rgb.shape[:2]:

        mask_rgb = cv2.resize(
            mask_rgb,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    image = image.astype(np.uint8)
    mask_rgb = mask_rgb.astype(np.uint8)

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        mask_rgb,
        alpha,
        0
    )

    return overlay


# =========================
# EXTRACT LOGITS
# =========================
def extract_logits(outputs):

    if hasattr(outputs, "logits"):
        return outputs.logits

    return outputs


# =========================
# VISUALIZE PREDICTION
# =========================
def visualize_prediction(
    model,
    dataset,
    device,
    save_dir="outputs/qualitative"
):

    os.makedirs(save_dir, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model.eval()

    with torch.no_grad():

        for idx, (img, mask) in enumerate(loader):

            img = img.to(device)

            outputs = model(img)
            logits = extract_logits(outputs)

            # =========================
            # UPSAMPLE SEGFORMER OUTPUT
            # =========================
            logits = F.interpolate(
                logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(logits, dim=1)

            pred = preds.squeeze().cpu().numpy()
            gt = mask.squeeze().cpu().numpy()

            # =========================
            # IMAGE DENORMALIZATION
            # =========================
            image = img.squeeze().permute(1, 2, 0).cpu().numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            image = (image * std) + mean
            image = np.clip(image, 0, 1)

            image = (image * 255).astype(np.uint8)

            # =========================
            # RGB MASKS
            # =========================
            gt_rgb = decode_mask(gt)
            pred_rgb = decode_mask(pred)

            overlay = overlay_mask(image, pred_rgb)

            # =========================
            # PLOT
            # =========================
            fig, ax = plt.subplots(1, 4, figsize=(18, 5))

            ax[0].imshow(image)
            ax[0].set_title("Original")
            ax[0].axis("off")

            ax[1].imshow(gt_rgb)
            ax[1].set_title("Ground Truth")
            ax[1].axis("off")

            ax[2].imshow(pred_rgb)
            ax[2].set_title("Prediction")
            ax[2].axis("off")

            ax[3].imshow(overlay)
            ax[3].set_title("Overlay")
            ax[3].axis("off")

            plt.tight_layout()

            plt.savefig(
                f"{save_dir}/sample_{idx}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            if idx >= 10:
                break


# =========================
# VISUALIZE DEGRADATION
# =========================
def visualize_degradation(
    dataset,
    save_path="outputs/degradation.png"
):

    img, _ = dataset[0]

    image = img.permute(1, 2, 0).numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    plt.figure(figsize=(6, 6))

    plt.imshow(image)
    plt.title(dataset.degrade)
    plt.axis("off")

    os.makedirs("outputs", exist_ok=True)

    plt.savefig(save_path)
    plt.close()


# =========================
# METRIC PLOT
# =========================
def plot_metrics(
    csv_path,
    save_path="outputs/metric_plot.png"
):

    import pandas as pd

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))

    for model_name in df["model"].unique():

        sub = df[df["model"] == model_name]

        plt.plot(
            sub["degradation"],
            sub["IoU_mean"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Degradation")
    plt.ylabel("Mean IoU")
    plt.title("Robustness Analysis")

    plt.legend()
    plt.grid(True)

    os.makedirs("outputs", exist_ok=True)

    plt.savefig(save_path)
    plt.close()