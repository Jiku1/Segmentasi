import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


# =========================
# COLOR MAP (FIXED)
# =========================
def color_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # background
    colored[mask == 0] = [0, 0, 0]

    # corrosion classes
    colored[mask == 1] = [0, 255, 0]      # green
    colored[mask == 2] = [255, 255, 0]    # yellow
    colored[mask == 3] = [255, 0, 0]      # red

    return colored


# =========================
# VISUALIZATION FUNCTION
# =========================
def visualize_publish(model, dataset, save_path, device):
    os.makedirs(save_path, exist_ok=True)

    model.eval()

    for i in range(3):

        img, mask = dataset[i]

        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))

            if isinstance(out, dict):
                out = out.logits

        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
        gt = mask.numpy()

        # =========================
        # IMAGE DENORMALIZATION 
        # =========================
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        gt_color = color_mask(gt)
        pred_color = color_mask(pred)

        # =========================
        # PLOT
        # =========================
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(img_np)
        ax[0].set_title("Input Image")

        ax[1].imshow(gt_color)
        ax[1].set_title("Ground Truth")

        ax[2].imshow(pred_color)
        ax[2].set_title("Prediction")

        for a in ax:
            a.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"{save_path}/viz_{i}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()