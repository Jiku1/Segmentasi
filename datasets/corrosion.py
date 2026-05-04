import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from configs.config import CONFIG


class CorrosionDataset(Dataset):

    def __init__(self, imgs, masks, degrade=None):
        self.imgs = imgs
        self.masks = masks
        self.degrade = degrade

    def __len__(self):
        return len(self.imgs)

    # =========================
    # DEGRADATION AUGMENTATION
    # =========================
    def apply_degradation(self, img):

        img = img.astype(np.float32)

        if self.degrade == "noise":
            img += np.random.normal(0, 15, img.shape)

        elif self.degrade == "blur":
            img = cv2.GaussianBlur(img, (7, 7), 0)

        elif self.degrade == "compression":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
            _, enc = cv2.imencode('.jpg', img.astype(np.uint8), encode_param)
            img = cv2.imdecode(enc, 1).astype(np.float32)

        elif self.degrade == "gamma":
            gamma = random.uniform(0.4, 2.2)
            inv = 1.0 / gamma

            table = np.array([
                ((i / 255.0) ** inv) * 255
                for i in range(256)
            ]).astype(np.uint8)

            img = cv2.LUT(img.astype(np.uint8), table).astype(np.float32)

        elif self.degrade == "shadow":
            h, w, _ = img.shape
            mask = np.zeros((h, w), dtype=np.float32)

            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = random.randint(w//2, w-1), random.randint(h//2, h-1)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

            strength = random.uniform(0.3, 0.6)
            img = img * (1 - mask[..., None] * strength)

        elif self.degrade == "glare":
            overlay = np.full(img.shape, 255, dtype=np.float32)
            alpha = random.uniform(0.2, 0.5)
            img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        return np.clip(img, 0, 255)

    # =========================
    # SAFE MASK ENCODING (FIXED)
    # =========================
    def encode_mask(self, mask):

        # =========================
        # IMPORTANT FIX: ensure uint8
        # =========================
        mask = mask.astype(np.uint8)

        unique = np.unique(mask)

        # debug optional (hapus kalau sudah stabil)
        # print("MASK UNIQUE:", unique)

        # =========================
        # MAP SAFETY (ROBUST)
        # =========================
        mask_out = np.zeros_like(mask, dtype=np.uint8)

        # background
        mask_out[mask == 0] = 0

        # IMPORTANT: adaptive mapping fallback
        mask_out[(mask > 0) & (mask <= 85)] = 1
        mask_out[(mask > 85) & (mask <= 170)] = 2
        mask_out[(mask > 170)] = 3

        # =========================
        # SAFETY CHECK (CRITICAL)
        # =========================
        assert mask_out.max() <= 3, "Invalid class detected in mask"
        assert mask_out.min() >= 0

        return mask_out

    # =========================
    # GET ITEM
    # =========================
    def __getitem__(self, idx):

        img = cv2.imread(self.imgs[idx])
        mask = cv2.imread(self.masks[idx], 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (CONFIG["img_size"], CONFIG["img_size"]))
        mask = cv2.resize(mask, (CONFIG["img_size"], CONFIG["img_size"]),
                          interpolation=cv2.INTER_NEAREST)

        # =========================
        # AUGMENTATION (IMAGE ONLY)
        # =========================
        if self.degrade and self.degrade != "none":
            img = self.apply_degradation(img)

        # =========================
        # NORMALIZATION
        # =========================
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img = torch.tensor(img).permute(2, 0, 1).float()

        # =========================
        # MASK FIX
        # =========================
        mask = self.encode_mask(mask)
        mask = torch.tensor(mask).long()

        return img, mask