import torch

DEBUG = True

CONFIG = {
    # =========================
    # DATA
    # =========================
    "img_size": 128 if DEBUG else 256,

    # =========================
    # TRAINING
    # =========================
    "batch_size": 4,
    "epochs": 1 if DEBUG else 20,
    "lr": 1e-4,
    "patience": 3 if DEBUG else 5,

    # =========================
    # SYSTEM
    # =========================
    "device": "cpu", #"mps" if torch.backends.mps.is_available() else "cpu",
    "num_workers": 0,         
    "pin_memory": False,   

    # =========================
    # OUTPUT
    # =========================
    "save_dir": "./outputs"
}