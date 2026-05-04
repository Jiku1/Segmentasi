import cv2
import numpy as np


# =========================
# GAUSSIAN NOISE
# =========================
def add_noise(img, var=25):
    noise = np.random.normal(0, var, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# =========================
# BLUR
# =========================
def add_blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)


# =========================
# GAMMA DISTORTION
# =========================
def gamma_correction(img, gamma=1.5):
    inv = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


# =========================
# SHADOW SIMULATION
# =========================
def add_shadow(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
    x2, y2 = np.random.randint(w//2, w), np.random.randint(h//2, h)

    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

    shadow = img.copy()
    shadow[mask == 1] = (shadow[mask == 1] * 0.4).astype(np.uint8)

    return shadow


# =========================
# GLARE (OVEREXPOSURE)
# =========================
def add_glare(img):
    overlay = img.copy()
    h, w = img.shape[:2]

    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    cv2.circle(overlay, (cx, cy), 200, (255, 255, 255), -1)

    return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)