import os
import cv2
import numpy as np

# 1. Tentukan path 
mask_dir = "data/train/masks"
mask_files = os.listdir(mask_dir)

# 2. Ambil sampel 5 file pertama untuk diperiksa
sample_files = [f for f in mask_files if f.endswith(('.png', '.jpg', '.jpeg'))][:5]

print("=== HASIL PEMERIKSAAN NILAI PIKSEL MASKER ===")
for file in sample_files:
    mask_path = os.path.join(mask_dir, file)
    
    # Baca masker dalam mode grayscale (hitam putih asli)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ambil nilai unik piksel yang ada di dalam gambar tersebut
    unique_values = np.unique(mask)
    
    print(f"File: {file} -> Nilai Piksel Unik: {unique_values}")