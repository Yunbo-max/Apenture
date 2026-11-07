# Comprehensive code readiness ? 6 November 2026

The comprehensive runner is implemented and has passed the validation below in Ubuntu WSL. The full scientific query experiment has not been launched; there are no completed query evaluations in this run root.

## Coverage

Six checkpoints: Qwen2.5-VL-7B, Qwen3-VL-4B, Qwen3-VL-8B, Gemma3-4B-it, InternVL3-8B-Instruct, and Phi3.5 Vision.

Five arms: Frozen, LoRA one pass, LoRA four passes, Random, and APERTURE. Both LoRA arms use rank16; the numbers refer to passes.

Twenty-two tasks: four registered BRIGHT events (Hawaii, Libya, Noto, Turkey), CAMELYON17 hospitals0?4, PathMNIST, three ManipBench Q1 domains, six ManipBench Q2 domain/question tasks, and Guardian UR5Fail/RoboFail/RoboVQA. Every model runs every arm on every task, at support seeds0,1,2: **1,980 planned states**, including1,584 trainable fits.

## Changes

The new separately versioned suite adds native paired/single-image handling across all six checkpoints, decoder-only LoRA, Phi fused-QKV K/V adaptation that preserves Q, native visual-mask checks, persistent Random bases, and fresh-process checkpoint replay. Scheduling validates sealed identities and artifact hashes and refuses incomplete result exports.

Preparation parses the official Q2 answer formats and deduplicates repeated answer options while retaining the originals in metadata. Guardian now uses one fixed query set across support seeds and excludes query image groups from support. Its calibration is question-held-out and may share frames with fitting; this is explicitly documented. Shared audits always use the first configured seed, independent of execution filters.

## Verified evidence

- 262 CPU regression tests passed across all repository test directories.
- Preflight passed for1,980 states,66 task/seed partitions, and all six local model snapshots.
- **132/132 real GPU model/task support audits passed**, covering native scoring, visual masks, nonzero gradients, checkpoint recomputation, and save/restore identity.
- **30/30 model/arm fitting and fresh-process replay checks passed** on CAMELYON17 H2, seed0. This covers all five arms on each of the six models, not fitting on all22 tasks. Maximum calibration-score replay difference:0.
- All132 audit records and60 fit/replay records passed final identity and artifact-hash verification against the current source. Validation performed zero query forwards.
- Windows and WSL copies of the12 new source/config/test/runner files are byte-identical. `git diff --check` passed.

Source signature: `d97f01b719ae045507db2be10c9eba0955c9a4dce6df4d370a5a64d581385c99`.

Environment: `.venv-table`, PyTorch2.8.0+cu128, Transformers4.57.1, PEFT0.17.1, two96GB NVIDIA GPUs. BF16, no quantization or offload; native image settings are locked in the protocol.

## Limits and remaining experiment

Only the four registered BRIGHT events are included. Six of the seven additional downloaded events lack the class counts for8 support plus100 query examples per class; Bata has sufficient marginal counts but no registered split in this suite. Budgets were not reduced to include them. See `docs/APERTURE_COMPREHENSIVE_PROTOCOL.md` for eligibility counts and exact data contracts.

The1,980-state query experiment remains to run. Audit success and H2 fitting checks establish readiness, not scientific performance or successful completion of every fit. Q2 accuracy/NLL are primary; candidate-position macro-F1 is descriptive.

The plan contains789,840 state/query-example pairs. Extrapolating support-example timings gives approximately324 GPU-hours for query scoring alone; this excludes fitting, loading, repeat checks, and overhead and is not a wall-time guarantee.

Run inside Ubuntu WSL:

```bash
cd /home/aperture/apenture
.venv-table/bin/python scripts/run_aperture_comprehensive.py run --gpus 0,1
```

The runner reuses valid completed audits/fits and performs required reference-repeat checks before queries. Configuration: `configs/aperture_comprehensive_v1.json`. Current run root: `runs/aperture_comprehensive_20261106_v6`. Prepared root: `data/prepared/aperture_comprehensive_20261106_v3`.

Machine-readable evidence: `runs/aperture_comprehensive_20261106_v6/readiness_verified.json`. Pairwise coverage: `reports/comprehensive_coverage_20261106.csv`. Historical development roots are preserved and are not the current readiness evidence. No changes were committed or pushed.
