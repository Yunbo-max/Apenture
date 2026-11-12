# CG-v1 INTERIM results snapshot — 2026-11-12

**INTERIM, IN PROGRESS.** Uploaded for reviewer visibility only. No case is
complete yet; do not cite these numbers as final. The full matrix is still
running on the qz platform; a final snapshot will replace this one.

## What this is

Merged per-case summaries from `scripts/qz_merge_summary.py` across all 31
run roots (the frozen acceptance root `runs/aperture_conditional_gated_20261111_recheck1`
plus 30 per-card roots `runs/aperture_conditional_gated_20261112_cNN`).
Only units with a sealed selection and at least one evaluated variant
contribute rows.

- Method: conditional-gated CG-v1 (branch `research/aperture-conditional-gated-v1`,
  fix commit `dfdf0bf`, plus diagnosis commit `1734f87`)
- Batch contract: execution_batch=1 (explicit, frozen)
- robofail/robovqa units excluded per standing decision (CG01 runs 299/329 of
  the planned units)

## Progress at snapshot time

| Case | Sealed units | Note |
|---|---|---|
| CG00 | 11/15 | 4 blocked by the phi causal-audit artifact (see DIAGNOSIS_CG_PHI_AUDIT.md) |
| CG01 | 229/299 | phi units (66) excluded pending the audit fix |
| CG02 | fits+selection 18/18; evals in progress | ~3.4h per unit |
| CG03 | selection 17/36; evals in progress | ~2.3h per unit |
| CG04-CG08 | running / queued | small cases |
| CG09 | complete (read-only analysis, recheck1/CG09/report) | causal_claim=false |

## File manifest

- `CG01_merged_summary.json` — 2124 evaluated variant rows (partial seeds)
- `CG02_merged_summary.json` — 206 rows (evals in progress)
- `CG03_merged_summary.json` — 158 rows (evals in progress)
- `CG05_merged_summary.json` — 3 rows (just started)
- `CG00/CG04/CG06/CG07/CG08` — no evaluated rows yet

## Known open items

1. Phi causal-audit failures: diagnosed as a cross-length bf16 GEMM artifact
   (see `DIAGNOSIS_CG_PHI_AUDIT.md` on this branch); awaiting the branch
   author's fix (recommended: length-matched alternative candidate in
   audit.py). Phi units will be re-run after the fix.
2. internvl3--camelyon17-hospital_2 units: same audit artifact.
3. Final upload will include complete merged summaries for all cases plus
   CG09's paired error analysis, and will replace this directory.
