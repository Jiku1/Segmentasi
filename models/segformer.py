import torch
from transformers import SegformerForSemanticSegmentation


def get_segformer(num_classes=4, pretrained=True):

    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        model = SegformerForSemanticSegmentation.from_config(
            SegformerForSemanticSegmentation.config_class(
                num_labels=num_classes
            )
        )

    # =========================
    # STABILITY FIX
    # =========================
    model.config.num_labels = num_classes

    # reset classifier bias (lebih stabil untuk fine-tuning kecil dataset)
    with torch.no_grad():
        if hasattr(model, "decode_head"):
            for m in model.decode_head.modules():
                if hasattr(m, "reset_parameters"):
                    try:
                        m.reset_parameters()
                    except:
                        pass

    return model