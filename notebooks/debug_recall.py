import numpy as np
import sys
sys.path.insert(0, '.')
from src.evaluate import compute_recall_at_k
import time

np.random.seed(0)
q = np.random.randn(100, 128).astype(np.float32)
g = np.random.randn(1000, 128).astype(np.float32)
q = q / np.linalg.norm(q, axis=1, keepdims=True)
g = g / np.linalg.norm(g, axis=1, keepdims=True)

q_ids = list(range(100))
g_ids = [i % 50 for i in range(1000)]

print("Running compute_recall_at_k...")
t0 = time.time()
result = compute_recall_at_k(q, q_ids, g, g_ids, k_values=(1, 5, 10))
print(f"  Done in {time.time()-t0:.3f}s")
print(f"  Result: {result}")
