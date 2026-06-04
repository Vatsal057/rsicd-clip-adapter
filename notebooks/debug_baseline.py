"""
Debug script: run the zero-shot baseline step by step, find where it hangs.
"""
import os, sys
sys.path.insert(0, '.')

import torch
import open_clip
import numpy as np

from src.dataset   import RSICDRetrievalDataset
from src.evaluate  import encode_images, encode_captions, compute_recall_at_k
from src.utils     import get_device, set_seed

set_seed(42)
device = get_device()
print(f"Device: {device}")

model, preprocess = open_clip.create_model_from_pretrained(
    "ViT-B-32", pretrained="openai", force_quick_gelu=True
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()

test_retrieval = RSICDRetrievalDataset(
    "data/splits/test.json", "data/raw/RSICD_images", preprocess, tokenizer
)

print("Step 1: encoding images...")
img_loader = test_retrieval.get_image_loader(batch_size=64, num_workers=0)
img_embs, img_ids = encode_images(model, img_loader, device)
print(f"  img_embs: {img_embs.shape}, dtype={img_embs.dtype}")
print(f"  norms: min={np.linalg.norm(img_embs, axis=1).min():.4f}, max={np.linalg.norm(img_embs, axis=1).max():.4f}")

print("Step 2: encoding captions...")
cap_loader = test_retrieval.get_caption_loader(batch_size=256, num_workers=0)
cap_embs, cap_ids = encode_captions(model, cap_loader, device)
print(f"  cap_embs: {cap_embs.shape}, dtype={cap_embs.dtype}")

print("Step 3: T->I recall...")
t2i = compute_recall_at_k(cap_embs, cap_ids, img_embs, img_ids)
print(f"  T->I: {t2i}")

print("Step 4: I->T recall...")
i2t = compute_recall_at_k(img_embs, img_ids, cap_embs, cap_ids)
print(f"  I->T: {i2t}")
print("ALL DONE")
