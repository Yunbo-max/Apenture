# CG-v1 INTERIM results snapshot — 2026-11-12 (v2 merge)

**INTERIM, IN PROGRESS.** For reviewer visibility only; no case is complete.
Final snapshot will replace this directory.

## v2 merge semantics (replaces the v1 merged_summary files)

The v1 files used `scripts/qz_merge_summary.py` as-is; reviewer feedback
confirmed its behaviors (no cross-root dedup, no budget in the group key,
scaled-preferred NLL). This v2 re-export fixes those:

- `units.jsonl` — one line per **unique (case, unit, variant)** record:
  model, domain, budget, seed, variant, source root, calibration temperature,
  full `raw` and `temperature_scaled` metrics. This is the authoritative
  per-unit export.
- `dedup_report.json` — 830 duplicate groups (same unit re-run in another
  card root; see below), kept root per group, max raw-metric disagreement
  across copies (all 0.0: the re-runs are bit-consistent).
- `summary_raw.json` / `summary_temperature_scaled.json` — aggregates keyed by
  (case, model, domain, **budget**, variant) with `independent_units` = number
  of unique (case, unit) records, never double-counted. Raw and
  temperature-scaled metrics are kept in separate files; the scaled file's
  NLL is the calibration-temperature NLL, not the raw NLL.
- `CG09_domain_analysis.json` — the read-only paired error analysis for the
  12 sealed q25 BRIGHT units (causal_claim=false).

## Why duplicates existed

The scale-out regenerated per-card assignments after some units had already
sealed in earlier card roots; the unit-distribution script only excluded
units sealed in the frozen acceptance root, so ~830 unit-variant results
were legitimately re-run (and re-sealed) in a second card root. Each seal is
internally valid; the v2 merge keeps exactly one per unit. The re-runs agree
bit-exactly (max raw-metric diff 0.0 across all 830 groups), confirming
deterministic replay rather than divergent experiments.

## Progress at snapshot time

| Case | Unique units with rows | Note |
|---|---|---|
| CG01 | 479 (case,unit,variant) rows / 229 units sealed | phi units (66) pending audit fix |
| CG02 | 87 rows | fits+selection 18/18; evals in progress |
| CG03 | 91 rows | selection 17/36; evals in progress |
| CG05 | 6 rows | alpha sweep barely started — NOT a tuned result |
| CG00/04/06/07/08 | no evaluated rows yet | running / queued |
| CG09 | complete (this snapshot) | 12 units, causal_claim=false |

## What these numbers do NOT support (per reviewer)

- CG05's conditional-gated alpha sweep is not here yet; the CG01 fixed
  configuration is not a searched gated result.
- CG02/CG03 rows are partial per-variant subsets; do not rank methods from
  them yet.
- Qwen3-VL-4B BRIGHT aggregates show the single-class-prediction signature
  (acc 33.33% / macro-F1 16.67%) across ALL method arms including Frozen and
  LoRA — per-sample verification is required before attributing it to any
  method (see CG09-style per-sample analysis once those units seal).

## Open items

1. Phi causal-audit failures: diagnosed as a cross-length bf16 GEMM artifact
   (`DIAGNOSIS_CG_PHI_AUDIT.md` on this branch); awaiting the fix
   (recommended: length-matched alternative candidate in audit.py).
2. internvl3--camelyon17-hospital_2: same artifact.
3. Final upload will contain complete cases, CG04 ablations, the full CG05
   alpha curve, CG08 budgets, and per-sample BRIGHT checks.
