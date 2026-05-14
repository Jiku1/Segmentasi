import os
import glob
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from configs.config import CONFIG
from datasets.corrosion import CorrosionDataset
from models.unet import UNet
from models.segformer import get_segformer
from engine.loop import run_training_experiment


# =========================
# EXPERIMENT SWITCH
# =========================
RUN_UNET = False
RUN_SEGFORMER = True
RUN_BOTH = False


# =========================
# DEVICE
# =========================
device = CONFIG["device"]
print(f"\nDevice: {device}")


# =========================
# MODEL SELECTOR
# =========================
def get_models():

    models = {}

    if RUN_BOTH or RUN_UNET:
        models["unet"] = UNet

    if RUN_BOTH or RUN_SEGFORMER:
        models["segformer"] = get_segformer

    return models


models = get_models()


# =========================
# SETTINGS FOLD & SEEDS
# =========================
NUM_FOLDS = 3
SEEDS = [42]

#[42, 123, 999]


# =========================
# DEGRADATION SCENARIOS
# =========================
degradations = [
    "none",
    "noise",
    "blur",
    "compression",
    "gamma",
    "shadow",
    "glare"
]


# =========================
# OUTPUT
# =========================
os.makedirs(CONFIG["save_dir"], exist_ok=True)


# =========================
# LOAD ALL DATASET
# =========================
def load_all_data():

    img_paths = []
    mask_paths = []

    splits = ["train", "validation", "test"]

    for split in splits:

        imgs = sorted(glob.glob(f"data/{split}/images/*"))
        masks = sorted(glob.glob(f"data/{split}/masks/*"))

        img_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in imgs
        }

        mask_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in masks
        }

        keys = sorted(set(img_map.keys()) & set(mask_map.keys()))

        for k in keys:
            img_paths.append(img_map[k])
            mask_paths.append(mask_map[k])

    print(f"\nTOTAL DATASET: {len(img_paths)} samples")

    return img_paths, mask_paths


# =========================
# LOAD ALL DATA
# =========================
all_imgs, all_masks = load_all_data()


# =========================
# K-FOLD
# =========================
kfold = KFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=42
)


# =========================
# RESULTS
# =========================
results = []


# =========================
# MAIN EXPERIMENT LOOP
# =========================
for model_name, model_fn in models.items():

    for deg in degradations:

        print("\n========================")
        print(f"MODEL: {model_name}")
        print(f"DEGRADATION: {deg}")
        print("========================")

        fold_scores = []

        # =========================
        # 5-FOLD CV
        # =========================
        for fold_idx, (train_idx, test_idx) in enumerate(
            kfold.split(all_imgs)
        ):

            print(f"\n----- FOLD {fold_idx + 1}/{NUM_FOLDS} -----")

            train_imgs = [all_imgs[i] for i in train_idx]
            train_masks = [all_masks[i] for i in train_idx]

            test_imgs = [all_imgs[i] for i in test_idx]
            test_masks = [all_masks[i] for i in test_idx]

            seed_scores = []

            # =========================
            # MULTI-SEED
            # =========================
            for seed in SEEDS:

                print(f"Seed: {seed}")

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # =========================
                # DATASET
                # =========================
                train_ds = CorrosionDataset(
                    train_imgs,
                    train_masks,
                    degrade="none"
                )

                # degradation ONLY on test
                test_ds = CorrosionDataset(
                    test_imgs,
                    test_masks,
                    degrade=deg
                )

                train_loader = DataLoader(
                    train_ds,
                    batch_size=CONFIG["batch_size"],
                    shuffle=True,
                    num_workers=CONFIG["num_workers"],
                    pin_memory=CONFIG["pin_memory"]
                )

                test_loader = DataLoader(
                    test_ds,
                    batch_size=CONFIG["batch_size"],
                    shuffle=False,
                    num_workers=CONFIG["num_workers"],
                    pin_memory=CONFIG["pin_memory"]
                )

                # =========================
                # MODEL RESET
                # =========================
                model = model_fn(num_classes=4).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=CONFIG["lr"]
                )

                criterion = torch.nn.CrossEntropyLoss()

                save_path = (
                    f"{CONFIG['save_dir']}/"
                    f"{model_name}_{deg}_fold{fold_idx+1}_seed{seed}.pth"
                )

                # =========================
                # TRAIN + EVALUATE
                # =========================
                best_iou = run_training_experiment(
                    model=model,
                    train_loader=train_loader,
                    val_loader=test_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    epochs=CONFIG["epochs"],
                    save_path=save_path
                )

                seed_scores.append(best_iou)

            # =========================
            # FOLD SCORE
            # =========================
            fold_mean = np.mean(seed_scores)
            fold_scores.append(fold_mean)

            print(f"Fold IoU: {fold_mean:.4f}")

        # =========================
        # FINAL STATS
        # =========================
        mean_iou = np.mean(fold_scores)
        std_iou = np.std(fold_scores)

        results.append([
            model_name,
            deg,
            mean_iou,
            std_iou
        ])

        print(
            f"\nFINAL RESULT | "
            f"IoU: {mean_iou:.4f} ± {std_iou:.4f}"
        )


# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(
    results,
    columns=[
        "model",
        "degradation",
        "IoU_mean",
        "IoU_std"
    ]
)

csv_path = f"{CONFIG['save_dir']}/kfold_results.csv"
df.to_csv(csv_path, index=False)


# =========================
# DONE
# =========================
print("\n========================")
print("EXPERIMENT FINISHED")
print("========================")
print(df)
print(f"\nSaved to: {csv_path}")