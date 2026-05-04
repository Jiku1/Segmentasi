import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import compute_iou, compute_f1, compute_dice


# =========================
# TRAIN ONE EPOCH
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0

    for imgs, masks in tqdm(loader, desc="Train"):

        imgs = imgs.to(device).float()
        masks = masks.to(device).long()

        outputs = model(imgs)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# EVALUATION
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=4):

    model.eval()

    total_loss = 0
    ious, f1s, dices = [], [], []

    for imgs, masks in tqdm(loader, desc="Eval"):

        imgs = imgs.to(device).float()
        masks = masks.to(device).long()

        outputs = model(imgs)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        ious.append(compute_iou(preds, masks, num_classes))
        f1s.append(compute_f1(preds, masks, num_classes))
        dices.append(compute_dice(preds, masks, num_classes))

    return (
        total_loss / len(loader),
        sum(ious) / len(ious),
        sum(f1s) / len(f1s),
        sum(dices) / len(dices)
    )


# =========================
# FULL TRAIN LOOP
# =========================
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

    best_iou = 0

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_iou, val_f1, val_dice = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"Dice: {val_dice:.4f}"
        )

        # SAVE BEST
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path)

    return best_iou