import torch
from transformers import SegformerForSemanticSegmentation


def get_segformer(num_classes=4):

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # =========================
    # HARD FREEZE EVERYTHING EXCEPT HEAD
    # =========================
    for param in model.parameters():
        param.requires_grad = False

    for param in model.decode_head.parameters():
        param.requires_grad = True

    model.eval()  # backbone BN stop tracking

    return model