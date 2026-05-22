import torch

DEBUG = False  # Set ke False saat dijalankan di RunPod

CONFIG = {
    # =========================
    # DATA
    # =========================
    "img_size": 128 if DEBUG else 512,

    # =========================
    # TRAINING
    # =========================
    "batch_size": 4 if DEBUG else 32,
    "epochs": 1 if DEBUG else 50,
    "lr": 1e-4,
    "patience": 3 if DEBUG else 7,

    # =========================
    # SYSTEM (Perbaikan: Deteksi Otomatis Langsung di Sini)
    # =========================
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0 if DEBUG else 4,         
    "pin_memory": False if DEBUG else True,   

    # =========================
    # OUTPUT
    # =========================
    "save_dir": "./outputs"
}
