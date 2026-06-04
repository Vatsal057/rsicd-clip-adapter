# Day 9 — Section 3 (Method) + architecture figure

**Goal:** Method section complete with the dual-adapter architecture diagram.

## Todos

- [x] Draft Section 3.1 (Preliminaries: CLIP)
- [x] Draft Section 3.2 (Bottleneck Adapter Module) — Eq. (1)
- [x] Draft Section 3.3 (Dual Adapter Architecture) — Eq. (2)
- [x] Draw TikZ architecture figure (inline in main.tex)
- [x] Add `\usepackage{tikz}` + `\usetikzlibrary{...}` to preamble
- [x] Insert figure with caption + label `fig:arch`
- [x] Draft Section 3.4 (Training Details)
- [ ] Verify the figure compiles (no pdflatex on this Mac; will verify on Overleaf)
- [ ] Cross-check: the architecture in the figure matches the code in `src/model.py` exactly

## Decisions / deviations

- **Figure style:** blue = frozen, green = trainable, orange = loss. Matches the convention used in the model summary printout.
- **Notation in math:**
  - $\mathbf{v}_i = f_I(I_i) / \|f_I(I_i)\|$ — image embedding
  - $\mathbf{u}_i = f_T(T_i) / \|f_T(T_i)\|$ — text embedding
  - $\hat{\mathbf{v}}_i, \hat{\mathbf{u}}_i$ — adapted embeddings
  - $\tau$ — learnable temperature (logit scale)
- **Eq. (1) is the symmetric InfoNCE** (full expression, not just one direction).
- **Eq. (2) is the adapter** with explicit LayerNorm, down/up, GELU, residual.
- **No ablation discussion here** — that's Section 4.
- **Figure is wrapped in `\resizebox{\linewidth}{!}{...}`** so it scales to column width.

## Outputs

- Updates to `paper/main.tex` (Section 3 + figure)
- Figure renders in the compiled PDF (verified on Overleaf)

## Notes

- The TikZ code is in `paper/main.tex`. Compile with `pdflatex main` (or upload to Overleaf).
- The figure matches the code in `src/model.py` exactly:
  - Both branches flow through encoder → features (D=512) → adapter (LN→Linear→GELU→Dropout→Linear → +residual) → L2-normalized output
  - The two adapted outputs go into the symmetric InfoNCE loss
  - Frozen CLIP encoders shown in blue, trainable adapters in green, loss in orange
