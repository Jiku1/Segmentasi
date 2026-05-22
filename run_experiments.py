import os
import random
import gc  # Tambahan wajib untuk pembersihan RAM
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader

from configs.config import CONFIG
from datasets.corrosion import CorrosionDataset
from models.unet import UNet
from models.segformer import get_segformer
from engine.loop import run_training_experiment
# Import fungsi visualisasi
from utils.visualization import plot_metrics, visualize_prediction 



# =====================================================
# ENV
# =====================================================
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# =====================================================
# FLAGS
# =====================================================
RUN_UNET = True       # Kembalikan ke True jika Anda ingin menguji kedua model sekaligus
RUN_SEGFORMER = True

NUM_FOLDS = 5
SEEDS = [42]

DEGRADATIONS = [
    "none", "noise", "blur",
    "compression", "gamma", "shadow", "glare"
]


# =====================================================
# DEVICE
# =====================================================
device = torch.device(CONFIG["device"]) 
print(f"\nDevice: {device}")


# =====================================================
# SAFE DIR
# =====================================================
os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# =====================================================
# SEED
# =====================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================
# DATA LOADER
# =====================================================
def load_all_data():

    imgs, masks = [], []
    splits = ["train", "validation", "test"]

    print("\n================ DATA LOADING ================")

    for split in splits:

        img_dir = f"data/{split}/images"
        mask_dir = f"data/{split}/masks"

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"[SKIP] Missing folder: {split}")
            continue

        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))

        img_paths = [os.path.join(img_dir, f) for f in img_files]
        mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

        imgs.extend(img_paths)
        masks.extend(mask_paths)

    print("\nTOTAL DATASET:", len(imgs))

    if len(imgs) == 0:
        raise RuntimeError("Dataset kosong. Cek folder data/")

    return imgs, masks


# =====================================================
# MODEL FACTORY
# =====================================================
def get_models():
    models = {}

    if RUN_UNET:
        models["unet"] = UNet

    if RUN_SEGFORMER:
        models["segformer"] = get_segformer

    return models


# =====================================================
# MAIN
# =====================================================
def main():

    all_imgs, all_masks = load_all_data()

    trainval_imgs, test_imgs, trainval_masks, test_masks = train_test_split(
        all_imgs,
        all_masks,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTrainVal: {len(trainval_imgs)} | Test: {len(test_imgs)}")

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    results = []

    models = get_models()

    for model_name, model_fn in models.items():

        print(f"\n================ MODEL: {model_name} ================")

        for deg in DEGRADATIONS:

            print(f"\n---- Degradation: {deg} ----")

            fold_dice_scores = []
            fold_boundary_scores = []

            for fold, (tr_idx, val_idx) in enumerate(kfold.split(trainval_imgs)):

                print(f"\nFold {fold+1}/{NUM_FOLDS}")

                tr_imgs = [trainval_imgs[i] for i in tr_idx]
                tr_masks = [trainval_masks[i] for i in tr_idx]

                val_imgs = [trainval_imgs[i] for i in val_idx]
                val_masks = [trainval_masks[i] for i in val_idx]

                seed_dice = []
                seed_boundary = []

                for seed in SEEDS:

                    set_seed(seed)

                    train_ds = CorrosionDataset(tr_imgs, tr_masks, degrade="none")
                    val_ds = CorrosionDataset(val_imgs, val_masks, degrade=deg)

                    train_loader = DataLoader(
                        train_ds,
                        batch_size=CONFIG["batch_size"],
                        shuffle=True,
                        num_workers=CONFIG["num_workers"],
                        pin_memory=CONFIG["pin_memory"]
                    )

                    val_loader = DataLoader(
                        val_ds,
                        batch_size=CONFIG["batch_size"],
                        shuffle=False,
                        num_workers=CONFIG["num_workers"],
                        pin_memory=CONFIG["pin_memory"]
                    )

                    model = model_fn(num_classes=4).to(device)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
                    criterion = torch.nn.CrossEntropyLoss()

                    save_path = f"checkpoints/{model_name}_{deg}_f{fold}_s{seed}.pth"

                    # Eksekusi Loop Training
                    output = run_training_experiment(
                        model,
                        train_loader,
                        val_loader,
                        optimizer,
                        criterion,
                        device,
                        CONFIG["epochs"],
                        save_path,
                        patience=CONFIG["patience"]
                    )

                    # AMBIL METRIK DICE DAN BOUNDARY F1
                    # (Menggunakan data evaluasi epoch terbaik dari 'output')
                    # Jika output loop Anda mengembalikan history, ambil entri terakhir/terbaiknya
                    best_epoch_idx = np.argmax(output["history"]["iou"])
                    seed_dice.append(output["history"]["dice"][best_epoch_idx])
                    seed_boundary.append(output["history"]["boundary_f1"][best_epoch_idx])

                    # =====================================================
                    # CETAK GAMBAR KUALITATIF
                    # =====================================================
                    # Menghasilkan 10 sampel gambar hasil visualisasi segmentasi model ini
                    visualize_prediction(model, model_name, val_ds, device)

                    # ULTRA CLEANUP: Menghapus seluruh objek komputasi agar RunPod bebas OOM
                    del model, optimizer, criterion, train_loader, val_loader, train_ds, val_ds
                    torch.cuda.empty_cache()
                    gc.collect()

                fold_dice_scores.append(np.mean(seed_dice))
                fold_boundary_scores.append(np.mean(seed_boundary))

            mean_dice = np.mean(fold_dice_scores)
            std_dice = np.std(fold_dice_scores)
            mean_boundary = np.mean(fold_boundary_scores)
            std_boundary = np.std(fold_boundary_scores)

            # Masukkan ke array dengan susunan kolom terbaru
            results.append([model_name, deg, mean_dice, std_dice, mean_boundary, std_boundary])

            print(f"\nDice: {mean_dice:.4f} ± {std_dice:.4f} | Boundary F1: {mean_boundary:.4f} ± {std_boundary:.4f}")
            
            # Backup data CSV secara real-time di setiap akhir siklus degradasi
            pd.DataFrame(results, columns=["model", "degradation", "Dice_mean", "Dice_std", "Boundary_mean", "Boundary_std"]).to_csv("results_kfold.csv", index=False)

    # Simpan hasil akhir komparatif lengkap
    df = pd.DataFrame(results, columns=["model", "degradation", "Dice_mean", "Dice_std", "Boundary_mean", "Boundary_std"])
    df.to_csv("results_kfold.csv", index=False)

    # Panggil fungsi plot untuk menggambar 2 kurva tren robustness otomatis
    plot_metrics("results_kfold.csv")

    print("\nEXPERIMENT DONE")
    print(df)


if __name__ == "__main__":
    main()
