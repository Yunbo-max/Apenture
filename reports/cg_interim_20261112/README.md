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

## CG05 alpha/LR sweep — partial tuned readout (17/18 units sealed)

The CG01 headline table uses the FIXED configuration (alpha=3, lr=.05). The
tuned per-dataset readout comes from CG05 (q25/g34 x PANEL x 3 seeds x 12
configs); file: `CG05_alpha_sweep_partial.json` (mean over sealed seeds).

Best gated config per dataset (mean accuracy over seeds, partial):

| model | dataset | best gated config | best gated acc | fixed (a3,lr.05) | best static acc |
|---|---|---|---|---|---|
| g34 | bright-hawaii | a2, lr.01 | 33.56 | 32.56 | 33.44 |
| g34 | bright-libya | a3, lr.05 (=fixed) | 34.56 | 34.56 | 34.11 |
| g34 | hospital_2 | a2, lr.05 | 72.67 (F1 57.0) | 72.47 (F1 54.0) | 72.47 (F1 64.0) |
| q25 | bright-hawaii | a3, lr.05 (=fixed) | 34.67* | 34.67* | 36.67 |
| q25 | bright-libya | a2/a3, lr.05 | 37.00 | 37.00 | 37.56 |
| q25 | hospital_2 | a2, lr.05 | 72.33 (F1 65.1) | 72.00 (F1 68.0) | 72.13 (F1 65.3) |

\* n=1 sealed seed; the other two seeds' best gated are a2-lr.05 (34.50 mean over 2).

Tuned-vs-fixed delta for gated is small on PANEL (+0 to +1.0 pp accuracy);
the fixed CG01 configuration is therefore close to the per-dataset optimum on
the PANEL domains, but per-dataset best configs differ (e.g. g34-hawaii
prefers a2/lr.01; g34-h2 prefers a2/lr.05, whose F1 is also more stable than
the fixed config's). On q25-hawaii the tuned STATIC configuration beats the
tuned gated one — reported as-is.

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
