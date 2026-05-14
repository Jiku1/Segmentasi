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

from utils.visualization import (
    visualize_prediction,
    visualize_degradation,
    plot_metrics
)


# =========================
# EXPERIMENT SWITCH
# =========================
RUN_UNET = True
RUN_SEGFORMER = True
RUN_BOTH = True


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
# SETTINGS
# =========================
NUM_FOLDS = 5

SEEDS = [
    42,
    123,
    999
]


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
# OUTPUT DIR
# =========================
os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================
# LOAD ALL DATASET
# =========================
def load_all_data():

    img_paths = []
    mask_paths = []

    splits = [
        "train",
        "validation",
        "test"
    ]

    for split in splits:

        imgs = sorted(
            glob.glob(f"data/{split}/images/*")
        )

        masks = sorted(
            glob.glob(f"data/{split}/masks/*")
        )

        img_map = {
            os.path.splitext(
                os.path.basename(p)
            )[0]: p
            for p in imgs
        }

        mask_map = {
            os.path.splitext(
                os.path.basename(p)
            )[0]: p
            for p in masks
        }

        keys = sorted(
            set(img_map.keys()) &
            set(mask_map.keys())
        )

        for k in keys:

            img_paths.append(img_map[k])
            mask_paths.append(mask_map[k])

    print("\n========================")
    print(f"TOTAL DATASET: {len(img_paths)}")
    print("========================")

    return img_paths, mask_paths


# =========================
# LOAD DATA
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
# MAIN LOOP
# =========================
for model_name, model_fn in models.items():

    for deg in degradations:

        print("\n========================")
        print(f"MODEL: {model_name}")
        print(f"DEGRADATION: {deg}")
        print("========================")

        fold_scores = []

        # =========================
        # K-FOLD CV
        # =========================
        for fold_idx, (train_idx, test_idx) in enumerate(
            kfold.split(all_imgs)
        ):

            print(
                f"\n----- "
                f"FOLD {fold_idx + 1}/{NUM_FOLDS} "
                f"-----"
            )

            train_imgs = [
                all_imgs[i]
                for i in train_idx
            ]

            train_masks = [
                all_masks[i]
                for i in train_idx
            ]

            test_imgs = [
                all_imgs[i]
                for i in test_idx
            ]

            test_masks = [
                all_masks[i]
                for i in test_idx
            ]

            seed_scores = []

            # =========================
            # MULTI-SEED
            # =========================
            for seed in SEEDS:

                print(f"\nSeed: {seed}")

                # =========================
                # FIX RANDOMNESS
                # =========================
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

                # =========================
                # DATALOADER
                # =========================
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
                model = model_fn(
                    num_classes=4
                ).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=CONFIG["lr"]
                )

                criterion = torch.nn.CrossEntropyLoss()

                save_path = (
                    f"{CONFIG['save_dir']}/"
                    f"{model_name}_"
                    f"{deg}_"
                    f"fold{fold_idx+1}_"
                    f"seed{seed}.pth"
                )

                # =========================
                # TRAIN
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
                # VISUALIZATION
                # =========================
                visualize_prediction(
                    model=model,
                    dataset=test_ds,
                    device=device,
                    save_dir=(
                        f"outputs/"
                        f"{model_name}_"
                        f"{deg}_"
                        f"fold{fold_idx+1}_"
                        f"seed{seed}"
                    )
                )

                visualize_degradation(
                    dataset=test_ds,
                    save_path=(
                        f"outputs/"
                        f"{model_name}_"
                        f"{deg}_"
                        f"fold{fold_idx+1}_"
                        f"seed{seed}_"
                        f"degradation.png"
                    )
                )

            # =========================
            # FOLD SCORE
            # =========================
            fold_mean = np.mean(seed_scores)

            fold_scores.append(fold_mean)

            print(
                f"\nFold Result "
                f"| IoU: {fold_mean:.4f}"
            )

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

        print("\n========================")
        print("FINAL RESULT")
        print("========================")

        print(
            f"IoU: "
            f"{mean_iou:.4f} "
            f"± "
            f"{std_iou:.4f}"
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

csv_path = (
    f"{CONFIG['save_dir']}/"
    f"kfold_results.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# =========================
# METRIC PLOT
# =========================
plot_metrics(csv_path)


# =========================
# DONE
# =========================
print("\n========================")
print("EXPERIMENT FINISHED")
print("========================")

print(df)

print(f"\nSaved to: {csv_path}")