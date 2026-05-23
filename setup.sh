#!/bin/bash

# =======================================================================
# CONFIGURATION
# =======================================================================
GDRIVE_FILE_ID="1XTKq-HrSqiU33dB7vS78u35mMua1NyOU"
ZIP_NAME="Archive.zip"

echo "======================================================="
# Perbaiki jalur ke direktori kerja utama RunPod yang aman
cd /workspace || exit

# =======================================================================
# 1. UPGRADE & INSTALL SYSTEM LIBRARIES ]
# =======================================================================
echo "[1/4] Memperbarui paket sistem & dependensi OpenCV..."
apt-get update && apt-get install -y libgl1-mesa-glx unzip wget

# =======================================================================
# 2. INSTALL PYTHON LIBRARIES
# =======================================================================
echo "[2/4] Menginstal pustaka Python untuk eksperimen..."
pip install --upgrade pip
pip install segmentation-models-pytorch transformers albumentations gdown tqdm scikit-learn pandas opencv-python matplotlib

# =======================================================================
# 3. DOWNLOAD DATASET FROM GOOGLE DRIVE
# =======================================================================
echo "[3/4] Mengunduh dataset 10 GB dari Google Drive..."
# Pastikan folder penampung file arsip tersedia
mkdir -p /workspace/downloads
gdown --id "$GDRIVE_FILE_ID" -O "/workspace/downloads/$ZIP_NAME"

# =======================================================================
# 4. EXTRACT DATASET TO THE RIGHT STRUCTURE
# =======================================================================
echo "[4/4] Mengekstrak dataset ke folder data/..."
# Membuat folder data/ utama agar selaras dengan skrip Python Anda
mkdir -p /workspace/

# Mengekstrak isi zip langsung ke dalam folder data/
unzip -q "/workspace/downloads/$ZIP_NAME" -d /workspace/data/

# Opsional: Hapus file zip mentah setelah diekstrak agar Volume Disk 40 GB Anda tetap lega
rm "/workspace/downloads/$ZIP_NAME"

echo "======================================================="
echo " SETUP SELESAI SUKSES! EKOSISTEM SIAP DIGUNAKAN "
echo "======================================================="
