import os
import glob
import random
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
from utils.visualization import plot_metrics
from utils.device import get_device


# =====================================================
# ENV
# =====================================================
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# =====================================================
# CONFIG
# =====================================================
RUN_UNET = False
RUN_SEGFORMER = True

NUM_FOLDS = 5
SEEDS = [42, 123, 999]

DEGRADATIONS = [
    "none", "noise", "blur",
    "compression", "gamma", "shadow", "glare"
]


# =====================================================
# DEVICE
# =====================================================
device, _ = get_device()
print(f"\nDevice: {device}")


# =====================================================
# SAFE MKDIR
# =====================================================
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


safe_mkdir(CONFIG["save_dir"])
safe_mkdir("outputs")
safe_mkdir("checkpoints")


# =====================================================
# SEED CONTROL
# =====================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# LOAD DATA (ROBUST VERSION)
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

        img_files = sorted(glob.glob(f"{img_dir}/*"))
        mask_files = sorted(glob.glob(f"{mask_dir}/*"))

        print(f"\n[{split}] images: {len(img_files)} masks: {len(mask_files)}")

        def key(p):
            return os.path.splitext(os.path.basename(p))[0]

        img_map = {key(p): p for p in img_files}
        mask_map = {key(p): p for p in mask_files}

        keys = sorted(set(img_map.keys()) & set(mask_map.keys()))

        print(f"matched pairs: {len(keys)}")

        for k in keys:
            imgs.append(img_map[k])
            masks.append(mask_map[k])

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
# MAIN PIPELINE
# =====================================================
def main():

    all_imgs, all_masks = load_all_data()

    # =================================================
    # 80/20 SPLIT (FIXED TEST SET)
    # =================================================
    trainval_imgs, test_imgs, trainval_masks, test_masks = train_test_split(
        all_imgs,
        all_masks,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTrainVal: {len(trainval_imgs)} | Test: {len(test_imgs)}")

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    results = []

    # =================================================
    # MODEL LOOP
    # =================================================
    for model_name, model_fn in get_models().items():

        print(f"\n================ MODEL: {model_name} ================")

        for deg in DEGRADATIONS:

            print(f"\n---- Degradation: {deg} ----")

            fold_scores = []

            # =============================================
            # K-FOLD ONLY ON TRAINVAL
            # =============================================
            for fold, (tr_idx, val_idx) in enumerate(kfold.split(trainval_imgs)):

                print(f"\nFold {fold+1}/{NUM_FOLDS}")

                tr_imgs = [trainval_imgs[i] for i in tr_idx]
                tr_masks = [trainval_masks[i] for i in tr_idx]

                val_imgs = [trainval_imgs[i] for i in val_idx]
                val_masks = [trainval_masks[i] for i in val_idx]

                seed_scores = []

                # =========================================
                # SEED LOOP
                # =========================================
                for seed in SEEDS:

                    set_seed(seed)

                    train_ds = CorrosionDataset(tr_imgs, tr_masks, degrade="none")
                    val_ds = CorrosionDataset(val_imgs, val_masks, degrade=deg)

                    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

                    model = model_fn(num_classes=4).to(device)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    criterion = torch.nn.CrossEntropyLoss()

                    save_path = f"checkpoints/{model_name}_{deg}_f{fold}_s{seed}.pth"

                    output = run_training_experiment(
                        model,
                        train_loader,
                        val_loader,
                        optimizer,
                        criterion,
                        device,
                        CONFIG["epochs"],
                        save_path
                    )

                    # =========================
                    # SAFE READ
                    # =========================
                    best_iou = output.get("best_iou", None)

                    if best_iou is None:
                        best_iou = max(output["history"]["iou"])

                    seed_scores.append(best_iou)

                fold_scores.append(np.mean(seed_scores))

            mean_iou = np.mean(fold_scores)
            std_iou = np.std(fold_scores)

            results.append([model_name, deg, mean_iou, std_iou])

            print(f"\nIoU: {mean_iou:.4f} ± {std_iou:.4f}")

    # =================================================
    # SAVE RESULT
    # =================================================
    df = pd.DataFrame(results, columns=["model", "degradation", "IoU_mean", "IoU_std"])

    csv_path = "results_kfold.csv"
    df.to_csv(csv_path, index=False)

    plot_metrics(csv_path)

    print("\nEXPERIMENT DONE")
    print(df)


if __name__ == "__main__":
    main()