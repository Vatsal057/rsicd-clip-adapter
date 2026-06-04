import numpy as np
print("numpy ok")
import faiss
print(f"faiss ok: {faiss.__version__}")
import sys
sys.path.insert(0, '.')
print("importing evaluate...")
from src.evaluate import compute_recall_at_k
print("imported")

q = np.random.randn(10, 8).astype(np.float32)
g = np.random.randn(20, 8).astype(np.float32)
q = q / np.linalg.norm(q, axis=1, keepdims=True)
g = g / np.linalg.norm(g, axis=1, keepdims=True)

print("calling compute_recall_at_k...")
result = compute_recall_at_k(q, [0]*10, g, [i%5 for i in range(20)], k_values=(1, 5))
print(f"result: {result}")
print("done")
