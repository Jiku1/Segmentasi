import os
import random
import gc
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader

from configs.config import CONFIG
from datasets.corrosion import CorrosionDataset
from models.unet import UNet
from models.segformer import get_segformer
from engine.loop import run_training_experiment, evaluate  # Menggunakan evaluate bawaan loop.py
from utils.visualization import visualize_prediction 

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

RUN_UNET = True       
RUN_SEGFORMER = True
NUM_FOLDS = 5
SEEDS = [42]

# Daftar degradasi sesuai BAB III
DEGRADATIONS = ["none", "noise", "blur", "compression", "gamma", "shadow", "glare"]

device = torch.device(CONFIG["device"]) 
print(f"\n[SISTEM] Berjalan pada Device: {device}")

os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_all_data():
    imgs, masks = [], []
    splits = ["train", "validation", "test"]
    print("\n================ DATA LOADING ================")
    for split in splits:
        img_dir = f"data/{split}/images"
        mask_dir = f"data/{split}/masks"
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue
        imgs.extend([os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))])
        masks.extend([os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir))])
    print("TOTAL DATASET:", len(imgs))
    return imgs, masks

def get_models():
    models = {}
    if RUN_UNET: models["unet"] = UNet
    if RUN_SEGFORMER: models["segformer"] = get_segformer
    return models

def main():
    all_imgs, all_masks = load_all_data()
    trainval_imgs, test_imgs, trainval_masks, test_masks = train_test_split(
        all_imgs, all_masks, test_size=0.2, random_state=42
    )
    print(f"TrainVal: {len(trainval_imgs)} | Test: {len(test_imgs)}")
    
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    models = get_models()
    
    # Array penampung hasil mentah per fold
    raw_results = []

    for model_name, model_fn in models.items():
        print(f"\n=============================================")
        print(f"   START EXPERIMENT FOR MODEL: {model_name.upper()}   ")
        print(f"=============================================")

        # -----------------------------------------------------------------
        # TAHAP 1: TRAINING 5-FOLD HANYA PADA KONDISI BERSIH (NONE)
        # -----------------------------------------------------------------
        for fold, (tr_idx, val_idx) in enumerate(kfold.split(trainval_imgs)):
            print(f"\n>>> [TRAINING] {model_name.upper()} - Fold {fold+1}/{NUM_FOLDS}")
            
            tr_imgs = [trainval_imgs[i] for i in tr_idx]
            tr_masks = [trainval_masks[i] for i in tr_idx]
            val_imgs = [trainval_imgs[i] for i in val_idx]
            val_masks = [trainval_masks[i] for i in val_idx]

            # Dataset pelatihan WAJIB bersih ("none")
            train_ds = CorrosionDataset(tr_imgs, tr_masks, degrade="none")
            val_ds = CorrosionDataset(val_imgs, val_masks, degrade="none")

            train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"])
            val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"])

            model = model_fn(num_classes=4).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
            criterion = torch.nn.CrossEntropyLoss() # Tetap dikirim agar tidak merusak argumen loop.py
            
            checkpoint_path = f"checkpoints/{model_name}_clean_f{fold}.pth"

            # Jalankan training loop dengan Hybrid Loss otomatis dari loop.py Anda
            output = run_training_experiment(
                model, train_loader, val_loader, optimizer, criterion, 
                device, CONFIG["epochs"], checkpoint_path, patience=CONFIG["patience"]
            )

            # Catat hasil performa Baseline (Kondisi Bersih)
            best_epoch_idx = np.argmax(output["history"]["iou"])
            baseline_dice = output["history"]["dice"][best_epoch_idx]
            baseline_boundary = output["history"]["boundary_f1"][best_epoch_idx]
            
            raw_results.append([model_name, "none", fold+1, baseline_dice, baseline_boundary])

            # Pembersihan VRAM parno OOM
            del model, optimizer, criterion, train_loader, val_loader, train_ds, val_ds
            torch.cuda.empty_cache()
            gc.collect()

        # -----------------------------------------------------------------
        # TAHAP 2: INFERENCE (EVALUASI ROBUSTNESS) PADA DATA DEGRADASI
        # -----------------------------------------------------------------
        print(f"\n>>> [EVALUASI ROBUSTNESS DEGRADASI] Model: {model_name.upper()}")
        
        for deg in DEGRADATIONS:
            if deg == "none":
                continue # Lewati karena sudah dihitung saat training di atas
                
            print(f" -> Menguji Efek Visual: {deg}")
            
            for fold, (tr_idx, val_idx) in enumerate(kfold.split(trainval_imgs)):
                val_imgs = [trainval_imgs[i] for i in val_idx]
                val_masks = [trainval_masks[i] for i in val_idx]
                
                # Memuat dataset yang diberi gangguan visual khusus
                val_ds = CorrosionDataset(val_imgs, val_masks, degrade=deg)
                val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

                # Bangkitkan arsitektur kosong, lalu muat bobot terbaik dari Tahap 1
                model = model_fn(num_classes=4).to(device)
                checkpoint_path = f"checkpoints/{model_name}_clean_f{fold}.pth"
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                
                # EVALUASI INSTAN (Hanya Inference tanpa latih ulang)
                # Mengembalikan: total_loss, val_iou, val_f1, val_dice, val_bf1
                _, _, _, deg_dice, deg_bf1 = evaluate(model, val_loader, device, num_classes=4)

                raw_results.append([model_name, deg, fold+1, deg_dice, deg_bf1])
                
                # Cetak Gambar Kualitatif Cacat Segmen untuk Bab IV
                if fold == 0: # Cukup simpan visualisasi dari fold pertama agar disk tidak penuh
                    visualize_prediction(model, f"{model_name}_{deg}", val_ds, device)

                del model, val_loader, val_ds
                torch.cuda.empty_cache()
                gc.collect()

        # Simpan checkpoint berkala berformat baris mentah per fold
        df_raw = pd.DataFrame(raw_results, columns=["model", "degradation", "fold", "Dice", "Boundary_F1"])
        df_raw.to_csv("results_raw_kfold.csv", index=False)

    # -----------------------------------------------------------------
    # TAHAP 3: AGREGASI OTOMATIS MENJADI MEAN & STD UNTUK TABEL BAB IV
    # -----------------------------------------------------------------
    print("\n================ EXPERIMENT FINISHED ================")
    print("Memproses ringkasan statistik akhir untuk Tabel Bab IV...")
    
    df_raw = pd.read_csv("results_raw_kfold.csv")
    df_summary = df_raw.groupby(["model", "degradation"]).agg(
        Dice_Mean=("Dice", "mean"),
        Dice_Std=("Dice", "std"),
        Boundary_Mean=("Boundary_F1", "mean"),
        Boundary_Std=("Boundary_F1", "std")
    ).reset_index()
    
    df_summary.to_csv("result_kfold.csv", index=False)
    print("Sukses!")

if __name__ == "__main__":
    main()
