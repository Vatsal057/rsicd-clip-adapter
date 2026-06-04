"""Quick check: zero-shot baseline also runs on the val split."""
import os, sys
sys.path.insert(0, '.')

import open_clip
from src.dataset  import RSICDRetrievalDataset
from src.evaluate import evaluate_model, save_results
from src.utils    import get_device, set_seed

set_seed(42)
device = get_device()

model, preprocess = open_clip.create_model_from_pretrained(
    "ViT-B-32", pretrained="openai", force_quick_gelu=True
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

val = RSICDRetrievalDataset(
    "data/splits/val.json", "data/raw/RSICD_images", preprocess, tokenizer
)
res = evaluate_model(model, val, device, split_name="val", num_workers=0)
res["model"] = "zero_shot_clip"
res["device"] = device
save_results(res, "results/metrics/baseline_zeroshot_val.json")
print("\nVal baseline done.")
