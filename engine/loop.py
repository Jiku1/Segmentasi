import torch
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp  # Tambahan untuk integrasi DiceLoss

from utils.metrics import (
    compute_iou,
    compute_f1,
    compute_dice,
    compute_boundary_f1
)

# =====================================================
# EARLY STOPPING (Tetap Aman)
# =====================================================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return

        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# =====================================================
# LOGITS HANDLING & HYBRID LOSS FUNCTION
# =====================================================
def extract_logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs


def resize_logits(logits, masks):
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
    return logits


def compute_hybrid_loss(logits, masks):
    """
    Menghitung gabungan Cross Entropy & Dice Loss Multi-Class
    Sesuai dengan janji analisis ketidakseimbangan kelas pada Bab III.
    """
    # 1. Cross Entropy Loss Standar untuk multi-kelas
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    ce_loss = ce_loss_fn(logits.contiguous(), masks.contiguous().long())
    
    # 2. Dice Loss khusus kelas korosi [1, 2, 3], abaikan background [0]
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', classes=[1, 2, 3])
    dice_loss = dice_loss_fn(logits.contiguous(), masks.contiguous().long())
    
    # Gabungan seimbang 50:50 (Bisa disesuaikan metodenya)
    return ce_loss + dice_loss


# =====================================================
# TRAIN ONE EPOCH (Diperbarui dengan Hybrid Loss)
# =====================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(imgs)
        logits = extract_logits(outputs)
        logits = resize_logits(logits, masks)

        # Menggunakan fungsi Hybrid Loss gabungan
        loss = compute_hybrid_loss(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# =====================================================
# EVALUATION (Diperbarui dengan Hybrid Loss)
# =====================================================
@torch.no_grad()
def evaluate(model, loader, device, num_classes=4):
    model.eval()
    total_loss = 0.0

    iou_list = []
    f1_list = []
    dice_list = []
    boundary_list = []

    for imgs, masks in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).long()

        outputs = model(imgs)
        logits = extract_logits(outputs)
        logits = resize_logits(logits, masks)

        loss = compute_hybrid_loss(logits, masks)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        # Catatan: Pastikan fungsi internal di bawah ini mengekstrak macro average
        iou_list.append(compute_iou(preds, masks, num_classes))
        f1_list.append(compute_f1(preds, masks, num_classes))
        dice_list.append(compute_dice(preds, masks, num_classes))
        boundary_list.append(compute_boundary_f1(preds, masks, num_classes))

    return (
        total_loss / max(len(loader), 1),
        sum(iou_list) / max(len(iou_list), 1),
        sum(f1_list) / max(len(f1_list), 1),
        sum(dice_list) / max(len(dice_list), 1),
        sum(boundary_list) / max(len(boundary_list), 1),
    )


# =====================================================
# FULL TRAINING LOOP
# =====================================================
def run_training_experiment(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion, # Dipertahankan sebagai argumen agar tidak merusak skrip utama main.py Anda
    device,
    epochs,
    save_path,
    patience=5
):
    model.to(device)

    best_iou = 0.0
    early_stopping = EarlyStopping(patience=patience)

    history = {
        "train_loss": [],
        "val_loss": [],
        "iou": [],
        "f1": [],
        "dice": [],
        "boundary_f1": []
    }

    for epoch in range(epochs):
        # =========================
        # TRAIN (Hapus pemanggilan argumen criterion lama)
        # =========================
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )

        # =========================
        # EVAL
        # =========================
        val_loss, val_iou, val_f1, val_dice, val_bf1 = evaluate(
            model, val_loader, device, num_classes=4
        )

        # =========================
        # HISTORY
        # =========================
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["iou"].append(val_iou)
        history["f1"].append(val_f1)
        history["dice"].append(val_dice)
        history["boundary_f1"].append(val_bf1)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("------------------------------")
        print(f"Train Loss  : {train_loss:.4f}")
        print(f"Val Loss    : {val_loss:.4f}")
        print(f"IoU (mIoU)  : {val_iou:.4f}")
        print(f"F1 Score    : {val_f1:.4f}")
        print(f"Dice Coeff  : {val_dice:.4f}  <-- Target Utama BAB III")
        print(f"Boundary F1 : {val_bf1:.4f}  <-- Target Utama BAB III")

        # =========================
        # SAVE BEST MODEL (Berdasarkan metrik IoU / Dice Terpilih)
        # =========================
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"-> Pembaruan Model Terbaik Disimpan ke {save_path}!")

        # =========================
        # EARLY STOPPING
        # =========================
        early_stopping(val_iou)

        if early_stopping.stop:
            print(f"\nEarly stopping terpicu pada epoch ke-{epoch+1} karena performa tidak berkembang.")
            break

    return {
        "best_iou": best_iou,
        "history": history
    }
