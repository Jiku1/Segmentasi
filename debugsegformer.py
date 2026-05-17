import torch
from transformers import SegformerForSemanticSegmentation

try:
    device = torch.device("mps")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0").to(device)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    output = model(dummy_input)
    print("SegFormer berhasil berjalan di MPS!")
except Exception as e:
    print(f"Gagal karena: {e}")