# APERTURE vs project-canonical LoRA (NO HYPERPARAMETER SEARCH)

**Purpose:** A non-destructive, canonical/default-only *reporting view*. It does not delete or rewrite the tuned LoRA/DoRA artifacts. The main comparison uses fixed configurations on both sides. A separate, explicitly asymmetric view below compares calibration-selected APERTURE to default LoRA.

- Source: [`CG-v1 FINAL units.jsonl`](https://github.com/Yunbo-max/Apenture/blob/f2b868ea6dd468671d4cbdf786e01cf2147ee018/reports/cg_interim_20261112/units.jsonl) at commit `f2b868e`, blob SHA `209dfbd6b91a5282addf9e7402afc546f9a48a3a`.
- The source contains 2,745 result records. This report selects the **1,800 CG01 records** only; for the primary view it uses `frozen`, `static-full`, `conditional-gated`, `lora1`, `lora4` with all three seeds per model/domain. No tuned `CG02/03/05/08` records enter the default-only primary table.
- `lora4` is **rank-16 LoRA trained for four passes**, not rank-4. Its project canonical settings are `lora_alpha=32`, dropout `0.05`, LR `1e-4` for paired BRIGHT and `2e-4` for single-image tasks, all language decoder Q/K/V/O projection targets. `lora1` is the same setup with one pass.
- Static and gated APERTURE use rank 16, K/V at layers 14 and 27, alpha 3, LR 0.05, four full-support optimization passes, L2 0.001. Neither side undergoes per-task hyperparameter search in this primary table.
- Numbers below are **macro-F1 (%) on raw (pre-temperature) query predictions**, averaged first over the three support seeds and then equally over listed domains within each group. Same fit/calibration/query split per unit. These are retrospective, previously inspected test sets, not a new blinded study.

## Canonical comparison: all 5 registered models and all 20 registered domains

| Model | Domain group (count) | Frozen F1 | Static Full F1 | Conditional-gated F1 | Default LoRA-1pass F1 | **Default LoRA-4pass F1** | Gated − LoRA-4 (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | BRIGHT (4) | 22.2 | 31.1 | 29.2 | 29.5 | 36.6 | -7.4 |
| Qwen2.5-VL-7B | CAMELYON17 (5) | 50.3 | 53.7 | 52.6 | 39.2 | 54.1 | -1.4 |
| Qwen2.5-VL-7B | PathMNIST (1) | 29.1 | 44.4 | 41.4 | 45.1 | 68.8 | -27.4 |
| Qwen2.5-VL-7B | ManipBench-Q1 (3) | 10.1 | 13.7 | 13.2 | 74.7 | 75.6 | -62.4 |
| Qwen2.5-VL-7B | ManipBench-Q2 (6) | 48.8 | 53.8 | 54.1 | 58.6 | 63.8 | -9.8 |
| Qwen2.5-VL-7B | Guardian (1) | 41.4 | 47.8 | 48.4 | 42.1 | 49.9 | -1.5 |
| Qwen3-VL-4B | BRIGHT (4) | 16.7 | 16.7 | 16.7 | 16.7 | 16.7 | 0.0 |
| Qwen3-VL-4B | CAMELYON17 (5) | 22.5 | 22.5 | 22.5 | 35.7 | 47.3 | -24.8 |
| Qwen3-VL-4B | PathMNIST (1) | 2.2 | 22.3 | 16.3 | 49.7 | 69.9 | -53.6 |
| Qwen3-VL-4B | ManipBench-Q1 (3) | 22.8 | 29.1 | 27.8 | 35.0 | 60.2 | -32.4 |
| Qwen3-VL-4B | ManipBench-Q2 (6) | 16.1 | 24.6 | 22.8 | 50.6 | 54.8 | -32.0 |
| Qwen3-VL-4B | Guardian (1) | 33.7 | 33.7 | 33.7 | 33.2 | 33.2 | 0.5 |
| Gemma-3-4B | BRIGHT (4) | 22.4 | 20.4 | 21.9 | 21.6 | 21.8 | 0.1 |
| Gemma-3-4B | CAMELYON17 (5) | 41.5 | 52.9 | 43.6 | 33.7 | 46.5 | -2.9 |
| Gemma-3-4B | PathMNIST (1) | 16.9 | 25.7 | 20.3 | 19.7 | 40.3 | -19.9 |
| Gemma-3-4B | ManipBench-Q1 (3) | 10.0 | 11.9 | 10.4 | 31.7 | 56.6 | -46.1 |
| Gemma-3-4B | ManipBench-Q2 (6) | 45.1 | 45.5 | 45.6 | 43.7 | 52.4 | -6.8 |
| Gemma-3-4B | Guardian (1) | 50.0 | 43.6 | 44.0 | 33.2 | 36.2 | 7.8 |
| InternVL3-8B | BRIGHT (4) | 24.6 | 28.1 | 29.0 | 19.6 | 36.9 | -7.9 |
| InternVL3-8B | CAMELYON17 (5) | 25.2 | 46.4 | 43.9 | 31.1 | 47.7 | -3.8 |
| InternVL3-8B | PathMNIST (1) | 27.6 | 46.6 | 42.8 | 56.2 | 65.6 | -22.8 |
| InternVL3-8B | ManipBench-Q1 (3) | 10.0 | 10.0 | 10.0 | 36.7 | 50.0 | -40.0 |
| InternVL3-8B | ManipBench-Q2 (6) | 29.3 | 33.5 | 33.7 | 36.9 | 40.8 | -7.1 |
| InternVL3-8B | Guardian (1) | 46.7 | 42.3 | 50.5 | 33.5 | 41.0 | 9.5 |
| Phi-3.5-Vision | BRIGHT (4) | 17.2 | 23.0 | 23.6 | 25.5 | 25.0 | -1.4 |
| Phi-3.5-Vision | CAMELYON17 (5) | 40.3 | 55.0 | 47.6 | 39.3 | 56.1 | -8.5 |
| Phi-3.5-Vision | PathMNIST (1) | 14.8 | 26.5 | 24.6 | 29.5 | 63.3 | -38.7 |
| Phi-3.5-Vision | ManipBench-Q1 (3) | 17.6 | 16.2 | 16.5 | 47.1 | 61.7 | -45.2 |
| Phi-3.5-Vision | ManipBench-Q2 (6) | 28.5 | 39.6 | 37.2 | 49.1 | 51.3 | -14.1 |
| Phi-3.5-Vision | Guardian (1) | 33.0 | 48.0 | 47.1 | 40.0 | 46.2 | 0.9 |

**Full per-domain tables:** [`default_by_domain.csv`](./default_by_domain.csv) (all 100 model × domain cells), [`default_by_group.csv`](./default_by_group.csv), [`default_only.json`](./default_only.json), including accuracies and raw NLL/Brier.

## Secondary, asymmetric comparison — informative, NOT a fair tuned-baseline win claim

**Important:** The following six cells use the `CG05 ∪ CG08` **calibration-NLL-selected** gated APERTURE against **unchanged default `lora4`**. APERTURE receives a hyperparameter search; LoRA does not. Report separately and always point to already verified tuned-LoRA results (CG02/CG03).

| Model / domain | Gated canonical F1 | **Gated calibrated-selected F1** | Default LoRA-4pass F1 |
|---|---:|---:|---:|
| Gemma-3-4B / bright-hawaii-wildfire | 20.6 | **22.6** | 22.5 |
| Gemma-3-4B / bright-libya-flood | 25.3 | **19.4** | 20.8 |
| Gemma-3-4B / camelyon17-hospital_2 | 54.0 | **65.2** | 57.4 |
| Qwen2.5-VL-7B / bright-hawaii-wildfire | 27.6 | **38.9** | 34.7 |
| Qwen2.5-VL-7B / bright-libya-flood | 29.7 | **29.3** | 35.7 |
| Qwen2.5-VL-7B / camelyon17-hospital_2 | 68.0 | **68.2** | 71.7 |

Candidate selection used only raw calibration NLL, choosing among pre-existing CG05 alpha/LR (4 passes) and CG08 pass-count (fixed alpha/LR) candidates; this is a **union, not a Cartesian grid**. Query scores are read only after selection. The per-seed choices are preserved in `default_only.json`.

## Interpretation and retention of existing evidence

1. **Default-vs-default is a legitimate canonical experiment.** The primary table includes all domains; a default LoRA trained 4 passes is generally stronger than 1 pass and is the main LoRA column.
2. **Tuned-ours-vs-default-LoRA is NOT a symmetric tuning comparison.** Never label that secondary table as proof APERTURE outperforms a properly tuned LoRA.
3. **Do not delete CG02/CG03/CG05/CG08 or original result seals.** Reviewers specifically asked for stronger and longer-tuned LoRA, and deleting those experiments from the scientific record would make the conclusions less reliable. Link to the [verified tuned-LoRA report](https://github.com/Yunbo-max/Apenture/blob/5f7df1ff80bc2856e44a6be4b93b22111074771b/reports/aperture_rebuttal_closure_20261113/aperture_selected_CG02_CG03_multroot.json) as secondary evidence.
4. The project-defined canonical LoRA configuration is **not** the universal PEFT-package default, and method compute, parameter counts, and intervention target sites are unequal; disclose all three.
5. Guard against selective emphasis: robot Q1, PathMNIST, and q34 class-collapse results are included in the full tables. All these benchmarks and query labels were previously inspected; any post-hoc views remain exploratory.

## Audit checks executed for this static export

- Verified 5 models × 20 domains × 3 support seeds × 6 CG01 variants = **1,800 unique CG01 records**. Checked each model-domain for exactly one of each seed 0/1/2 and each selected method, finite raw metrics, and same query population across methods.
- Verified all 100 per-domain cells and all 30 model × group cells are represented without removing unfavorable cells.
- Independently located all 18 CG05/CG08 calibration-selected gated variants (6 model-domain settings × 3 seeds); none were selected from query metrics.

**No new GPU run, no new hyperparameter optimization, no change to `main` or source outputs, and no deletion of original tuned LoRA results.**
