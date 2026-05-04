import os
import numpy as np
import pandas as pd
import torch
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

print("\n========================")
print("EXPERIMENT MODE")
print("========================")
print(f"UNET: {RUN_UNET}")
print(f"SEGFORMER: {RUN_SEGFORMER}")
print(f"BOTH: {RUN_BOTH}")
print("========================\n")


os.makedirs(CONFIG["save_dir"], exist_ok=True)


# =========================
# DATA LOADER UTILITY
# =========================
def build_dataset(img_dir, mask_dir):
    import glob

    imgs = sorted(glob.glob(f"{img_dir}/*"))
    masks = sorted(glob.glob(f"{mask_dir}/*"))

    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}

    keys = sorted(set(img_map.keys()) & set(mask_map.keys()))

    paired_imgs = [img_map[k] for k in keys]
    paired_masks = [mask_map[k] for k in keys]

    print(f"{img_dir} → {len(paired_imgs)} samples")

    return paired_imgs, paired_masks


# =========================
# LOAD DATA
# =========================
train_imgs, train_masks = build_dataset("data/train/images", "data/train/masks")
val_imgs, val_masks     = build_dataset("data/validation/images", "data/validation/masks")


# =========================
# MODEL SELECTOR (ON/OFF SYSTEM)
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
# DEGRADATIONS
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
# RESULTS
# =========================
results = []


# =========================
# EXPERIMENT LOOP
# =========================
for model_name, model_fn in models.items():

    for deg in degradations:

        print(f"\n=== {model_name} | {deg} ===")

        seed_scores = []

        for seed in [42]:

            torch.manual_seed(seed)
            np.random.seed(seed)

            # =========================
            # DATASET
            # =========================
            train_ds = CorrosionDataset(train_imgs, train_masks, degrade=deg)
            val_ds   = CorrosionDataset(val_imgs, val_masks, degrade="none")

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

            # =========================
            # MODEL (FIX SEGFORMER HERE)
            # =========================
            model = model_fn(num_classes=4).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
            criterion = torch.nn.CrossEntropyLoss()

            save_path = f"{CONFIG['save_dir']}/{model_name}_{deg}.pth"

            # =========================
            # TRAIN
            # =========================
            best_iou = run_training_experiment(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=CONFIG["epochs"],
                save_path=save_path
            )

            seed_scores.append(best_iou)

        # =========================
        # STATS 
        # =========================
        mean_iou = np.mean(seed_scores)
        std_iou = np.std(seed_scores)

        results.append([model_name, deg, mean_iou, std_iou])

        print(f"Final IoU: {mean_iou:.4f} ± {std_iou:.4f}")


# =========================
# SAVE RESULT
# =========================
df = pd.DataFrame(
    results,
    columns=["model", "degradation", "IoU_mean", "IoU_std"]
)

output_path = f"{CONFIG['save_dir']}/thesis_results.csv"
df.to_csv(output_path, index=False)

print("\n========================")
print("DONE")
print("========================")
print(df)
print(f"\nSaved to: {output_path}")