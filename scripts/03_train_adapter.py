"""Run adapter training with the base config. Thin wrapper around src.train.train."""
import sys

sys.path.insert(0, ".")
from src.train import train

if __name__ == "__main__":
    cfg   = sys.argv[1] if len(sys.argv) > 1 else "configs/adapter_base.yaml"
    tag   = sys.argv[2] if len(sys.argv) > 2 else "adapter"
    res   = train(cfg, run_tag=tag)
    print("\n=== ADAPTER TRAINING COMPLETE ===")
    print(f"  T->I  R@1 : {res['text_to_image']['R@1']:>6.2f}%")
    print(f"  T->I  R@5 : {res['text_to_image']['R@5']:>6.2f}%")
    print(f"  T->I  R@10: {res['text_to_image']['R@10']:>6.2f}%")
    print(f"  I->T  R@1 : {res['image_to_text']['R@1']:>6.2f}%")
    print(f"  I->T  R@5 : {res['image_to_text']['R@5']:>6.2f}%")
    print(f"  I->T  R@10: {res['image_to_text']['R@10']:>6.2f}%")
    print(f"  Trainable : {res['trainable_params']:,}")
