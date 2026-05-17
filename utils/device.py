import torch


def get_device():

    device = "cpu" #if torch.backends.mps.is_available() else "cpu"

    use_segformer = True

    # =========================
    # FALLBACK RULES
    # =========================
    if device == "mps":
        print("\n MPS detected:")
        print("→ SegFormer will run in SAFE MODE (frozen backbone)")
        print("→ CNN (UNet) fully supported")

    return device, use_segformer