import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import DataLoader


# =========================
# COLOR MAP
# =========================
COLORS = {
    0: [0, 0, 0],        # background (Hitam)
    1: [0, 255, 0],      # Fair / Ringan (Hijau)
    2: [255, 255, 0],    # Poor / Sedang (Kuning)
    3: [255, 0, 0],      # Severe / Berat (Merah)
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


# =====================================================
# VISUALIZE PREDICTION (Hasil Kualitatif Gambar untuk Laporan)
# =====================================================
def visualize_prediction(
    model,
    model_name,
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

            logits = F.interpolate(
                logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(logits, dim=1)

            pred = preds.squeeze().cpu().numpy()
            gt = mask.squeeze().cpu().numpy()

            # Image Denormalization
            image = img.squeeze().permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            image = (image * std) + mean
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

            # RGB Masks
            gt_rgb = decode_mask(gt)
            pred_rgb = decode_mask(pred)
            overlay = overlay_mask(image, pred_rgb)

            # Plot Berdampingan untuk Lampiran Bab IV
            fig, ax = plt.subplots(1, 4, figsize=(18, 5))

            ax[0].imshow(image)
            ax[0].set_title("Original")
            ax[0].axis("off")

            ax[1].imshow(gt_rgb)
            ax[1].set_title("Ground Truth")
            ax[1].axis("off")

            ax[2].imshow(pred_rgb)
            ax[2].set_title(f"Prediction ({model_name})")
            ax[2].axis("off")

            ax[3].imshow(overlay)
            ax[3].set_title("Overlay")
            ax[3].axis("off")

            plt.tight_layout()
            
            # Disimpan dengan nama model agar tidak saling menimpa
            plt.savefig(
                f"{save_dir}/{model_name}_sample_{idx}.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

            if idx >= 10:  # Membatasi hanya 10 gambar sampel agar storage RunPod hemat
                break


# =====================================================
# METRIC PLOT (Otomatis Dipanggil di Akhir run_experiments.py Anda)
# =====================================================
def plot_metrics(
    csv_path,
    save_dir="outputs"
):
    """
    Fungsi ini otomatis membaca hasil K-Fold dan menggambar grafik tren 
    penurunan Dice Coefficient & Boundary F1 lengkap dengan pita simpangan baku (Std Dev).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"[WARNING] Berkas {csv_path} belum terbentuk. Grafik dilewati.")
        return

    df = pd.read_csv(csv_path)

    # 1. GRAFIK TREN PENURUNAN DICE COEFFICIENT (Bab 3.3.1 & 3.3.4)
    plt.figure(figsize=(10, 5))
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        plt.errorbar(
            sub["degradation"], 
            sub["Dice_mean"], 
            yerr=sub["Dice_std"], 
            marker='o', 
            linewidth=2,
            capsize=4,
            label=f"{model_name}"
        )
    plt.xlabel("Jenis Degradasi Citra Korosi", fontsize=10)
    plt.ylabel("Mean Dice Coefficient", fontsize=10)
    plt.title("Analisis Ketahanan Model: Penurunan Dice Coefficient terhadap Degradasi", fontsize=11, pad=15)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dice_comparison_robustness.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. GRAFIK TREN PENURUNAN BOUNDARY F1-SCORE (Bab 3.3.2 & 3.3.4)
    plt.figure(figsize=(10, 5))
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        plt.errorbar(
            sub["degradation"], 
            sub["Boundary_mean"], 
            yerr=sub["Boundary_std"], 
            marker='s', 
            linewidth=2,
            capsize=4,
            label=f"{model_name}"
        )
    plt.xlabel("Jenis Degradasi Citra Korosi", fontsize=10)
    plt.ylabel("Mean Boundary F1-Score", fontsize=10)
    plt.title("Analisis Sensitivitas Batas Tepi: Penurunan Boundary F1 terhadap Degradasi", fontsize=11, pad=15)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/boundary_comparison_robustness.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[SUKSES] 2 Grafik Analisis Kuantitatif (*Dice & Boundary*) Berhasil Diperbarui di Folder outputs/")
