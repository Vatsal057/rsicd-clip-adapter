# Day 14 — Final proofread + submission

**Goal:** Paper submitted, code frozen, everything in order.

## Todos

- [x] Run the paper checklist (top of IMPLEMENTATION.md)
- [x] Run the code/repo checklist
- [x] Run the submission checklist
- [ ] Read the PDF once more, top to bottom, looking only for typos
- [ ] Verify all figure references in text resolve to actual figure files
- [ ] Verify page count is within journal limit
- [ ] Verify all authors' names, affiliations, emails are correct
- [ ] Verify GitHub URL in Data Availability is live
- [ ] Verify the cover letter is signed
- [ ] Submit to the journal portal (ScienceDirect special issue)

## Paper checklist

- [ ] All `[TBD]` placeholders in `main.tex` replaced with actual numbers
- [ ] Abstract numbers match Table 1
- [ ] All 4 ablation tables filled with real numbers
- [ ] All 4 figures (training_curve, qualitative, failures, ablation_dim) included and referenced
- [ ] All `\cite{}` keys have matching entries in `refs.bib`
- [ ] References include at least 2 papers from 2023–2025
- [ ] No figures are raster-only (all PDFs, minimum 300 DPI)
- [ ] Page count is within journal limit (typically 8–12 pages)
- [ ] Authors' names and affiliations are correct
- [ ] GitHub URL in Data Availability section is live
- [ ] Hyperparameter table present (Table 1, added in Day 10)

## Code / repo checklist

- [ ] `bash reproduce.sh` runs end-to-end from a clean clone
- [ ] All `results/*.json` files have the numbers in the paper
- [ ] `requirements.txt` includes all dependencies with versions
- [ ] README shows how to download data and run experiments
- [ ] Trained adapter checkpoint is uploaded to repo (or Google Drive link)
- [ ] `data/splits/` JSONs committed (NOT the raw images)
- [ ] No `YOUR_USERNAME` placeholders left in `README.md`, `main.tex`, or `cover_letter.tex`
- [ ] `.gitignore` excludes `data/raw/`, `results/checkpoints/`, `__pycache__/`, `.venv/`

## Submission checklist

- [ ] PDF compiled cleanly with no LaTeX warnings
- [ ] Submitted to correct journal (Neurocomputing)
- [ ] Cover letter written (1 paragraph: what the paper does, why it fits the special issue, no prior submission elsewhere)
- [ ] All co-author names/emails confirmed
- [ ] Copyright transfer / open access option selected
- [ ] Suggested reviewers identified (3-5 from the related-work section)

## Search-and-replace checklist (Day 14 final pass)

Run these greps on `paper/main.tex` to ensure nothing was missed:

```bash
grep -n "XX.X" paper/main.tex        # should return nothing
grep -n "TBD" paper/main.tex         # should return nothing
grep -n "TODO" paper/main.tex        # should return nothing
grep -n "FIXME" paper/main.tex       # should return nothing
grep -n "your-name\|YOUR_USERNAME" paper/*.tex   # should return nothing
```

## Post-submission (optional)

- [ ] Add a `## Citation` section to the GitHub repo
- [ ] Tweet / post about the paper (with the preprint URL if applicable)
- [ ] Add a "Reproduction notes" section to the README with anything you learned the hard way:
  - "FAISS hard-crashes on macOS Apple Silicon due to OMP double-link; use `KMP_DUPLICATE_LIB_OK=TRUE` or skip FAISS"
  - "open_clip `force_quick_gelu=True` (not `quick_gelu`)"
  - "HF RSICD CSV has 5 captions concatenated; we use first sentence"
  - "Smoke test on MPS takes 105s; full training on T4 takes ~2h"

## Notes

- This is the day to slow down, not speed up. Every previous day was about producing; today is about verifying.
- A 30-minute read-through catches more issues than another 3 hours of writing.
- The greps above are your last line of defense. Run them.
- The "Reproduction notes" section in the README is gold for future researchers and reviewers. Don't skip it.
