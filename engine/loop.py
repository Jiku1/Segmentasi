import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import (
    compute_iou,
    compute_f1,
    compute_dice,
    compute_boundary_f1
)

# =====================================================
# LOGITS HANDLING
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


# =====================================================
# LOSS (CPU ONLY CLEAN)
# =====================================================
def compute_loss(logits, masks, criterion):
    # ensure correct format for CrossEntropy
    logits = logits.contiguous()
    masks = masks.contiguous().long()

    return criterion(logits, masks)


# =====================================================
# TRAIN ONE EPOCH
# =====================================================
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Train", leave=False):

        imgs = imgs.to(device).float().contiguous()
        masks = masks.to(device).long().contiguous()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(imgs)
        logits = extract_logits(outputs)
        logits = resize_logits(logits, masks)

        loss = compute_loss(logits, masks, criterion)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# =====================================================
# EVALUATION
# =====================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=4):

    model.eval()

    total_loss = 0.0

    iou_list = []
    f1_list = []
    dice_list = []
    boundary_list = []

    for imgs, masks in tqdm(loader, desc="Eval", leave=False):

        imgs = imgs.to(device).float().contiguous()
        masks = masks.to(device).long().contiguous()

        outputs = model(imgs)
        logits = extract_logits(outputs)
        logits = resize_logits(logits, masks)

        loss = compute_loss(logits, masks, criterion)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

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
    criterion,
    device,
    epochs,
    save_path
):

    best_iou = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "iou": [],
        "f1": [],
        "dice": [],
        "boundary_f1": []
    }

    model.to(device)

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_iou, val_f1, val_dice, val_bf1 = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["iou"].append(val_iou)
        history["f1"].append(val_f1)
        history["dice"].append(val_dice)
        history["boundary_f1"].append(val_bf1)

        print("\n==============================")
        print(f"Epoch [{epoch+1}/{epochs}]")
        print("==============================")
        print(f"Train Loss  : {train_loss:.4f}")
        print(f"Val Loss    : {val_loss:.4f}")
        print(f"IoU         : {val_iou:.4f}")
        print(f"F1          : {val_f1:.4f}")
        print(f"Dice        : {val_dice:.4f}")
        print(f"Boundary F1 : {val_bf1:.4f}")

        # SAVE BEST
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"\nBest model saved (IoU: {best_iou:.4f})")

    return {
        "best_iou": best_iou,
        "history": history
    }