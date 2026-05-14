import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import compute_iou, compute_f1, compute_dice


# =========================
# HELPER: OUTPUT HANDLER
# =========================
def extract_logits(outputs):
    """
    Support both:
    - Tensor output (CNN custom)
    - HuggingFace-style output (ModelOutput with .logits)
    """
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


# =========================
# TRAIN ONE EPOCH
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Train"):

        imgs = imgs.to(device).float()
        masks = masks.to(device).long()

        # forward
        outputs = model(imgs)
        logits = extract_logits(outputs)

        # resize logits to mask size
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # loss
        loss = criterion(logits, masks)

        # backward
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

    total_loss = 0.0
    ious, f1s, dices = [], [], []

    for imgs, masks in tqdm(loader, desc="Eval"):

        imgs = imgs.to(device).float()
        masks = masks.to(device).long()

        # forward
        outputs = model(imgs)
        logits = extract_logits(outputs)

        # resize
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # loss
        loss = criterion(logits, masks)
        total_loss += loss.item()

        # predictions
        preds = torch.argmax(logits, dim=1)

        # metrics (convert safely to float if needed)
        iou = compute_iou(preds, masks, num_classes)
        f1 = compute_f1(preds, masks, num_classes)
        dice = compute_dice(preds, masks, num_classes)

        ious.append(iou)
        f1s.append(f1)
        dices.append(dice)

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

    best_iou = 0.0

    for epoch in range(epochs):

        # ---- TRAIN ----
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # ---- EVAL ----
        val_loss, val_iou, val_f1, val_dice = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"\nEpoch [{epoch+1}/{epochs}]"
            f"\nTrain Loss: {train_loss:.4f}"
            f"\nVal Loss:   {val_loss:.4f}"
            f"\nIoU:        {val_iou:.4f}"
            f"\nF1:         {val_f1:.4f}"
            f"\nDice:       {val_dice:.4f}\n"
        )

        # ---- SAVE BEST MODEL ----
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"💾 Best model saved (IoU: {best_iou:.4f})")

    return best_iou