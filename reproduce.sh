#!/usr/bin/env bash
# Full reproduction of all experiments in the paper.
# Assumes: Python venv activated, RSICD dataset at data/raw/.

set -e

echo "====================================================="
echo "  RSICD Adapter-CLIP: Full Reproduction"
echo "====================================================="

cd "$(dirname "$0")"
# Activate venv if present
[ -d .venv ] && source .venv/bin/activate || true

echo ""
echo "[1/6] Preparing data splits..."
python scripts/01_prepare_splits.py

echo ""
echo "[2/6] Running zero-shot CLIP baseline..."
python scripts/02_run_baseline.py

echo ""
echo "[3/6] Training adapter model (~2 hours on T4)..."
python scripts/03_train_adapter.py configs/adapter_base.yaml adapter

echo ""
echo "[4/6] Running full fine-tune baseline (~5 hours on T4)..."
python scripts/04_run_fullfinetune.py

echo ""
echo "[5/6] Running ablation experiments (~26 GPU-hours on T4)..."
python scripts/05_ablations.py

echo ""
echo "[6/6] Generating paper figures..."
python scripts/06_qualitative.py

echo ""
echo "====================================================="
echo "All experiments complete!"
echo ""
echo "Results summary:"
python -c "
import json, os
metrics = 'results/metrics'
models = [
    ('baseline_zeroshot.json',    'Zero-shot CLIP'),
    ('adapter_results.json',      'Adapter-CLIP (ours)'),
    ('fullfinetune_results.json', 'Full fine-tune CLIP'),
]
print(f'{\"Model\":<30} {\"T->I R@1\":>9} {\"T->I R@5\":>9} {\"T->I R@10\":>10}')
print('-' * 62)
for fname, label in models:
    path = os.path.join(metrics, fname)
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        t2i = r['text_to_image']
        print(f'{label:<30} {t2i[\"R@1\"]:>9.2f} {t2i[\"R@5\"]:>9.2f} {t2i[\"R@10\"]:>10.2f}')
    else:
        print(f'{label:<30} {\"(missing)\":>9} {\"(missing)\":>9} {\"(missing)\":>10}')
"
echo ""
echo "Figures saved to: paper/figures/"
echo "====================================================="
