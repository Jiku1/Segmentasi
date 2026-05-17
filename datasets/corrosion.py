import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from configs.config import CONFIG


class CorrosionDataset(Dataset):

    def __init__(
        self,
        imgs,
        masks,
        degrade=None,
        degradation_level=1
    ):

        self.imgs = imgs
        self.masks = masks

        self.degrade = degrade

        # progressive degradation level
        # level: 1 → ringan
        # level: 5 → berat
        self.degradation_level = degradation_level

    # =====================================================
    # LENGTH
    # =====================================================
    def __len__(self):

        return len(self.imgs)

    # =====================================================
    # DEGRADATION
    # =====================================================
    def apply_degradation(self, img):

        img = img.astype(np.float32)

        level = self.degradation_level

        # -------------------------------------------------
        # GAUSSIAN NOISE
        # -------------------------------------------------
        if self.degrade == "noise":

            noise_std = {
                1: 5,
                2: 10,
                3: 15,
                4: 25,
                5: 35
            }[level]

            noise = np.random.normal(
                0,
                noise_std,
                img.shape
            )

            img = img + noise

        # -------------------------------------------------
        # BLUR
        # -------------------------------------------------
        elif self.degrade == "blur":

            kernel = {
                1: 3,
                2: 5,
                3: 7,
                4: 9,
                5: 11
            }[level]

            img = cv2.GaussianBlur(
                img,
                (kernel, kernel),
                0
            )

        # -------------------------------------------------
        # JPEG COMPRESSION
        # -------------------------------------------------
        elif self.degrade == "compression":

            quality = {
                1: 70,
                2: 50,
                3: 30,
                4: 15,
                5: 5
            }[level]

            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                quality
            ]

            _, enc = cv2.imencode(
                ".jpg",
                img.astype(np.uint8),
                encode_param
            )

            img = cv2.imdecode(
                enc,
                1
            ).astype(np.float32)

        # -------------------------------------------------
        # GAMMA DISTORTION
        # -------------------------------------------------
        elif self.degrade == "gamma":

            gamma_ranges = {
                1: (0.8, 1.2),
                2: (0.6, 1.4),
                3: (0.4, 1.8),
                4: (0.3, 2.0),
                5: (0.2, 2.5)
            }

            gmin, gmax = gamma_ranges[level]

            gamma = random.uniform(gmin, gmax)

            inv_gamma = 1.0 / gamma

            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype(np.uint8)

            img = cv2.LUT(
                img.astype(np.uint8),
                table
            ).astype(np.float32)

        # -------------------------------------------------
        # SHADOW
        # -------------------------------------------------
        elif self.degrade == "shadow":

            h, w, _ = img.shape

            shadow_mask = np.zeros(
                (h, w),
                dtype=np.float32
            )

            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)

            x2 = random.randint(w // 2, w - 1)
            y2 = random.randint(h // 2, h - 1)

            cv2.rectangle(
                shadow_mask,
                (x1, y1),
                (x2, y2),
                1,
                -1
            )

            strengths = {
                1: 0.15,
                2: 0.25,
                3: 0.40,
                4: 0.55,
                5: 0.70
            }

            strength = strengths[level]

            img = img * (
                1 - shadow_mask[..., None] * strength
            )

        # -------------------------------------------------
        # GLARE
        # -------------------------------------------------
        elif self.degrade == "glare":

            overlay = np.full(
                img.shape,
                255,
                dtype=np.float32
            )

            alphas = {
                1: 0.10,
                2: 0.20,
                3: 0.35,
                4: 0.50,
                5: 0.70
            }

            alpha = alphas[level]

            img = cv2.addWeighted(
                img,
                1 - alpha,
                overlay,
                alpha,
                0
            )

        return np.clip(img, 0, 255)

    # =====================================================
    # SAFE MASK ENCODING
    # =====================================================
    def encode_mask(self, mask):

        mask = mask.astype(np.uint8)

        mask_out = np.zeros_like(
            mask,
            dtype=np.uint8
        )

        # background
        mask_out[mask == 0] = 0

        # corrosion level mapping
        mask_out[(mask > 0) & (mask <= 85)] = 1
        mask_out[(mask > 85) & (mask <= 170)] = 2
        mask_out[(mask > 170)] = 3

        # safety check
        assert mask_out.min() >= 0
        assert mask_out.max() <= 3

        return mask_out

    # =====================================================
    # LOAD SAMPLE
    # =====================================================
    def __getitem__(self, idx):

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------
        img = cv2.imread(self.imgs[idx])

        if img is None:

            raise ValueError(
                f"Failed to load image: {self.imgs[idx]}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # LOAD MASK
        # -------------------------------------------------
        mask = cv2.imread(
            self.masks[idx],
            0
        )

        if mask is None:

            raise ValueError(
                f"Failed to load mask: {self.masks[idx]}"
            )

        # -------------------------------------------------
        # RESIZE
        # -------------------------------------------------
        img = cv2.resize(
            img,
            (
                CONFIG["img_size"],
                CONFIG["img_size"]
            )
        )

        mask = cv2.resize(
            mask,
            (
                CONFIG["img_size"],
                CONFIG["img_size"]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------------------------------
        # APPLY DEGRADATION
        # -------------------------------------------------
        if (
            self.degrade is not None and
            self.degrade != "none"
        ):

            img = self.apply_degradation(img)

        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------
        img = img.astype(np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        img = (img - mean) / std

        # -------------------------------------------------
        # TO TENSOR
        # -------------------------------------------------
        img = torch.tensor(
            img,
            dtype=torch.float32
        ).permute(2, 0, 1).contiguous()

        # -------------------------------------------------
        # ENCODE MASK
        # -------------------------------------------------
        mask = self.encode_mask(mask)

        mask = torch.tensor(
            mask,
            dtype=torch.long
        ).contiguous()

        return img, mask