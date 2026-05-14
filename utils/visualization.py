import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# =========================
# COLOR MAP
# =========================
COLORS = {
    0: [0, 0, 0],          # background
    1: [0, 255, 0],        # Fair
    2: [255, 255, 0],      # Poor
    3: [255, 0, 0],        # Severe
}


# =========================
# MASK → RGB
# =========================
def decode_mask(mask):

    h, w = mask.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLORS.items():
        rgb[mask == cls] = color

    return rgb


# =========================
# OVERLAY
# =========================
def overlay_mask(image, mask_rgb, alpha=0.5):

    image = image.astype(np.uint8)

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        mask_rgb,
        alpha,
        0
    )

    return overlay


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

            # =========================
            # SEGFORMER SAFE
            # =========================
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            preds = torch.argmax(outputs, dim=1)

            pred = preds.squeeze().cpu().numpy()
            gt = mask.squeeze().cpu().numpy()

            # =========================
            # DENORMALIZATION
            # =========================
            image = img.squeeze().permute(1, 2, 0).cpu().numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            image = (image * std) + mean
            image = np.clip(image, 0, 1)

            image = (image * 255).astype(np.uint8)

            # =========================
            # COLOR MASKS
            # =========================
            gt_rgb = decode_mask(gt)
            pred_rgb = decode_mask(pred)

            overlay = overlay_mask(image, pred_rgb)

            # =========================
            # FIGURE
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


# =========================
# QUALITATIVE COMPARISON
# =========================
def qualitative_comparison(
    original,
    gt,
    pred_unet,
    pred_segformer,
    save_path="outputs/qualitative_comparison.png"
):

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(gt)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred_unet)
    ax[2].set_title("UNet")
    ax[2].axis("off")

    ax[3].imshow(pred_segformer)
    ax[3].set_title("SegFormer")
    ax[3].axis("off")

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    plt.savefig(save_path)

    plt.close()