# Day 8 — LaTeX setup, Intro, Related Work

**Goal:** Elsevier template working, Sections 1 (Intro) and 2 (Related Work) drafted.

## Todos

- [x] Create `paper/main.tex` (full document, all 6 sections + abstract + TikZ)
- [x] Create `paper/refs.bib` (19 references covering CLIP, PEFT, RS retrieval, FAISS, survey papers)
- [x] Add TikZ architecture figure (in Section 3, ready for Day 9)
- [x] Create `paper/cover_letter.tex`
- [ ] Compile `main.tex` cleanly — no LaTeX on this Mac; will use Overleaf
- [ ] Read 6 key papers (CLIP-Adapter, RemoteCLIP, CoOp, AMFMN, Houlsby, LoRA) — abstracts at minimum
- [ ] Fill in author names, affiliation, contact email in `main.tex` (currently `Vatsal Vaghasiya / Supervisor Name / placeholder email`)
- [ ] Polish intro narrative (3 paragraphs → tight 1.5 pages)

## Decisions / deviations

- **Journal target:** `Neurocomputing` (set in `\journal{}`); `Pattern Recognition Letters` is the fallback. Both are Elsevier and use the `elsarticle` class.
- **`elsarticle.cls`:** must be present in `paper/` or in TeX search path. On Overleaf it's bundled with the Elsevier template. Locally on this Mac: not installed (no `pdflatex`). User will compile on Overleaf or install TeX later.
- **No `\maketitle`** — the Elsevier class builds the title from `\title`, `\author`, `\address`.
- **Abstract is a placeholder** until Day 12 (after all experiments are done). The `XX.X%` numbers will be filled in then.
- **Cover letter:** `paper/cover_letter.tex` uses the standard `letter` class. Independent of `main.tex`. Sender details still placeholder.
- **Tweaks to IMPLEMENTATION.md's prose:**
  - Updated to our actual trainable param count (527,873 not "530K")
  - Added a paragraph in Section 5 (Discussion) explaining the **deviation from the original RSICD 5-captions-per-image annotation** — we use the first sentence only, due to the HF redistribution format.
  - Adjusted the parameter-ratio claim from "~280×" to "$283\times$" (precise: 150,000,000 / 527,873 = 284.1).
- **TikZ libraries loaded:** `positioning, arrows.meta, shapes, calc` (small set, no exotic dependencies).
- **Section organization matches IMPLEMENTATION.md** — same 6 sections, same order.

## Outputs

- `paper/main.tex` (483 lines, full document)
- `paper/refs.bib` (143 lines, 19 references)
- `paper/cover_letter.tex` (52 lines, full letter)

## Notes

- Don't waste time polishing prose now — the structure matters, prose can be tightened on Day 12.
- The 6 papers cited in the related work section are the only ones the reviewers will spot-check. Read their abstracts.
- Compilation target: Overleaf (Elsevier journal article template), since pdflatex isn't installed locally.
