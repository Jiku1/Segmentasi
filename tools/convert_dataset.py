import os
import cv2
import json
import numpy as np
from tqdm import tqdm

print("🔥 LABELME CONVERTER FINAL (THESIS STABLE PIXEL PIPELINE)")

ROOT = "dataset"
OUT = "data"
IMG_SIZE = 256

CLASS_MAP = {
    "Fair": 1,
    "Poor": 2,
    "Severe": 3
}

# =========================
# SAFE POLYGON DRAW
# =========================
def draw(mask, points, cls):

    if not points or len(points) < 3:
        return mask, False

    pts = np.array(points, dtype=np.float32)

    h, w = mask.shape[:2]

    # clamp
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    pts = pts.astype(np.int32)

    cv2.fillPoly(mask, [pts], cls)

    return mask, True


# =========================
# PROCESS SPLIT
# =========================
def process_split(split):

    img_dir = f"{ROOT}/images/{split}/images"
    json_dir = f"{ROOT}/json/{split}/label"

    out_img_dir = f"{OUT}/{split}/images"
    out_mask_dir = f"{OUT}/{split}/masks"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    if not os.path.exists(img_dir):
        print(f" IMG DIR NOT FOUND: {img_dir}")
        return

    images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n Processing {split} ({len(images)})")

    valid, empty, missing_json = 0, 0, 0

    for img_name in tqdm(images):

        base = os.path.splitext(img_name)[0]

        img_path = os.path.join(img_dir, img_name)
        json_path = os.path.join(json_dir, base + ".json")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        has_object = False

        # =========================
        # LOAD JSON SAFELY
        # =========================
        if os.path.exists(json_path):

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                shapes = data.get("shapes", [])

                for shape in shapes:

                    label = shape.get("label", "Fair")
                    points = shape.get("points", [])

                    cls = CLASS_MAP.get(label, 0)

                    mask, ok = draw(mask, points, cls)
                    has_object = has_object or ok

            except Exception as e:
                print(f"⚠ JSON ERROR {json_path}: {e}")
                missing_json += 1
                continue

        else:
            missing_json += 1

        # =========================
        # STATS
        # =========================
        if has_object and np.max(mask) > 0:
            valid += 1
        else:
            empty += 1

        # =========================
        # RESIZE
        # =========================
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # =========================
        # SAVE
        # =========================
        cv2.imwrite(f"{out_img_dir}/{base}.jpg", img)
        cv2.imwrite(f"{out_mask_dir}/{base}.png", mask)

    total = valid + empty

    print(f"✔ VALID: {valid}")
    print(f"⚠ EMPTY: {empty}")
    print(f"⚠ NO JSON: {missing_json}")
    print(f"📊 EMPTY RATIO: {empty/total:.2f}")


# =========================
# RUN ALL SPLITS
# =========================
process_split("train")
process_split("validation")
process_split("test")

print("\n✅ LABELME CONVERTER READY (THESIS STABLE + SAFE)")