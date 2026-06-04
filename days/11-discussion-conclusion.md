# Day 11 — Section 5 (Discussion) + Section 6 (Conclusion)

**Goal:** Discussion and conclusion tie the experimental results back to the paper's claims.

## Todos

- [x] Draft Section 5.1 (Why do adapters work here?)
- [x] Draft Section 5.2 (Parameter efficiency) — added disk-size analogy (2 MB vs 600 MB)
- [x] Draft Section 5.3 (**NEW** Comparison to LoRA and prompt tuning) — honest scope, controlled comparison as future work
- [x] Draft Section 5.4 (Deviation from the original RSICD annotation) — refactored, mentions configurable caption source in code
- [x] Draft Section 5.5 (Limitations) — references Figure~\ref{fig:failures} directly
- [x] Draft Section 6 (Conclusion) + 3 future-work directions
- [x] Add Data Availability section (with GitHub URL)
- [ ] Replace `[TBD]` placeholders with real numbers from `adapter_results.json` / `fullfinetune_results.json`
- [ ] Cross-check: every claim in the discussion has a matching number in Section 4

## Decisions / deviations

- **5 Discussion subsections** (was 3 in IMPLEMENTATION.md):
  1. *Why adapters work* — LayerNorm normalizes feature magnitudes, near-identity init prevents destructive early updates, frozen backbone preserves generalizable features. **Added**: explicit reference to Figure~\ref{fig:training} (loss curve).
  2. *Parameter efficiency* — concrete numbers, file size of adapter weights (2 MB), distribution argument.
  3. *Comparison to LoRA and prompt tuning* — honest scope, not a head-to-head, future work.
  4. *Deviation from the original RSICD annotation* — refactored, mentions configurable caption source in code.
  5. *Limitations* — multi-modal reasoning failures, generalization to unseen RS domains, single-dataset evaluation, single-caption annotation under-represents linguistic diversity.
- **Conclusion:** restate the three contributions, give the headline number, list 3 concrete future-work directions: (1) adapters + prompt tuning combo, (2) multi-temporal RS video retrieval, (3) zero-shot generalization to unseen RS domains.
- **Data Availability:** explicit statement that RSICD is public, with the GitHub URL (placeholder until the repo is pushed).

## Outputs

- Updated `paper/main.tex` Section 5 (5 subsections), Section 6 (Conclusion), Data Availability
- All `[TBD]` placeholders flagged for Day 12 fill-in

## Notes

- The Discussion is the second-most-read section after the Abstract. Spend time on it.
- Avoid speculation. Tie every claim to a result or a citation.
- The LoRA/CoOp discussion is a deliberate honesty move — claiming "we beat LoRA" without running it would be scientifically weak. The honest framing is "we set the adapter baseline; future work should compare".
