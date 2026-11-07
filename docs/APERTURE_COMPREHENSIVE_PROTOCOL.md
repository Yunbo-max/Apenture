# Comprehensive six-backbone comparison protocol v1

This is an additive experiment suite. Its results belong to this protocol and must not replace numbers from historical protocols with different populations or fitting budgets.

The configuration expands every task across Qwen2.5-VL-7B, Qwen3-VL-4B, Qwen3-VL-8B, Gemma-3-4B-it, InternVL3-8B-Instruct, and Phi-3.5-Vision. Every comparison has Frozen, LoRA-1pass, LoRA-4passes, Random, and APERTURE. Both LoRA schedules use rank16, alpha32, dropout0.05 and decoder attention projections only. Random fits the same bounded controller in seeded random orthonormal directions; APERTURE estimates its directions from fitting support. Neither method accesses query labels while fitting.

## Tasks and budgets

| Dataset | Tasks | Fit / calibration | Query per task |
| --- | --- | --- | --- |
| BRIGHT | Hawaii, Libya, Noto, Turkey | 18 / 6 | 300 |
| CAMELYON17 | Hospitals 0–4 | 12 / 4 | 500 |
| PathMNIST-224 | CRC classification | 54 / 18 | 900 |
| ManipBench Q1 | Bridge pick/place, DROID articulation, DROID pick/place | 24 / 8 | 400 |
| ManipBench Q2 | Picking point and ending tile for each of the same three domains | 24 / 8 | 400 |
| Guardian execution | UR5Fail, RoboFail, RoboVQA | 12 / 4 | 122 / 113 / 341 respectively |

The matrix has **22 tasks × 6 models × 5 arms × 3 support seeds = 1,980 states**, including 1,584 trainable fits. Execution filters choose complete comparison units; they do not change the full plan or permit a partial scientific export. Raw and held-out-temperature versions of every arm, plus Frozen output bias, are reported. LoRA schedules are additionally selected using calibration NLL before query scoring.

The eleven downloaded BRIGHT events do not all satisfy the published balanced support24/query300 budget. Instance-label counts are: Bata 7058/227/135; Beirut 21587/630/25; Congo 15097/90/1416; Haiti 15527/120/9; Hawaii 1023/347/2164; La Palma 24232/90/5513; Libya 6248/2934/419; Marshall 6904/47/649; Morocco 5735/48/7; Noto 5907/283/791; Turkey 114777/3080/1857 (intact/damaged/destroyed). Thus six of the seven additional events fail even the marginal class counts needed for 8 support plus 100 query examples/class. Bata passes marginal counts but has no registered split in this extension. The four published events remain the explicit disaster population. The code accepts additional prepared tasks; it never lowers budgets or skips a failed task automatically.

Q2 answer strings are read from the four options explicitly printed in each question. Repeated options in the official source are deduplicated in first-occurrence order; they are the same semantic answer, not independent outcomes. This yields two to four unique candidates. Original option lists are retained in metadata. The task is balanced by the correct answer's unique-candidate position; accuracy and NLL are primary. Position-based macro-F1 is descriptive and is not semantic keypoint/tile class generalization. Raw scores contain only actual candidates. Numerical calibration pads missing positions with a finite log-zero mask whose exponential is exactly zero on the fixed temperature grid. Query groups are chosen deterministically from source IDs while reserving support capacity, before any model outcomes.

## Input and adaptation contracts

Qwen inputs are 448px; InternVL inputs are 224px/64 visual tokens per image; Gemma uses its native 896px processor with pan-and-scan disabled; Phi uses 448px input and the downloaded processor's four-crop setting. All weights stay unquantized BF16 on a single GPU. Phi's bundled eager attention is selected explicitly; the other native implementations use the SDPA math execution context. No CPU offload is allowed.

BRIGHT crops are materialized using the existing 448px crop/margin policy. Both images enter the prompt, and the controller changes POST image rows only. Single-image tasks use all visual rows. Each adapter checks image placeholder counts against native processor metadata. Gemma token-type IDs are retained; Phi's numbered negative image IDs are validated. Candidate likelihoods sum answer-token log probabilities using an identical prompt prefix across candidates.

APERTURE and Random act at decoder layers14/27 with rank16, alpha3, a full bounded residual matrix, four updates, Adam LR0.05, raw-matrix L2=0.001 and gradient clipping1. Support loss reduction is mean for BRIGHT and sum for single-image tasks. LoRA uses LR1e-4/accumulation3 for BRIGHT and LR2e-4/accumulation1 for single images. Phi exposes fused QKV: explicit identity sites route only K and V slices through the controller, preserving Q and unselected tokens. LoRA targets Phi decoder qkv/o projections and the other models' decoder q/k/v/o projections.

All arms share support, calibration and query IDs. Medical source partitions retain patient/content isolation. BRIGHT and robot fit/calibration partitions are stratified within original support and may share source groups; query groups are disjoint from all support. PathMNIST has image-content isolation, not verified patient isolation. Query labels are read for split-integrity checks but never supplied to adaptation, basis estimation, temperature fitting, output-bias fitting or LoRA schedule selection.

Guardian fixes the original seed-0 query population for every support seed. Support is sampled from the full converted metadata after excluding every query group. Repeated robot frames with different subtask questions remain distinct valid examples; duplicate complete image/question/candidate inputs are rejected. Shared frames may occur between fitting and calibration within the support population, so this is question-held-out calibration rather than episode-held-out calibration. No image content or group may overlap the query population.

## Execution and validation

Use Ubuntu WSL and the existing `.venv-table` environment (Transformers4.57.1, PEFT0.17.1 and CUDA PyTorch). From `/home/aperture/apenture`:

```bash
.venv-table/bin/python scripts/run_aperture_comprehensive.py prepare
.venv-table/bin/python scripts/run_aperture_comprehensive.py plan
.venv-table/bin/python scripts/run_aperture_comprehensive.py check
.venv-table/bin/python scripts/run_aperture_comprehensive.py audit --gpus 0,1
.venv-table/bin/python scripts/run_aperture_comprehensive.py smoke --datasets camelyon17-hospital_2 --seeds 0 --gpus 0,1
.venv-table/bin/python scripts/run_aperture_comprehensive.py run --gpus 0,1
.venv-table/bin/python scripts/run_aperture_comprehensive.py status
.venv-table/bin/python scripts/run_aperture_comprehensive.py summarize
```

`smoke` uses the real registered fitting budget, saves each state, then restores it in a fresh process and checks held-out support scores. It executes zero query forwards. `fit` runs only fitting for selected units; `run` also performs the reference repeat fits, seals all five arms' calibration decisions and evaluates queries. `--models`, `--datasets` and `--seeds` restrict execution while retaining the full immutable plan. A shared model/task audit always uses the first configured seed, so later seed subsets can reuse it safely.

Every model/task must pass a real support-only audit before fitting: visual masks, zero-residual identity, direct mask isolation, nonzero gradients, checkpoint recomputation and save/restore. Each model's reference H2 APERTURE fit is repeated on the same GPU before query execution. A failed audit or replay stops the affected run. No tolerance widening or alternate checkpoint is selected automatically.

Source, model shards (including remote model Python), manifests, input pixels, library versions and completion artifacts are fingerprinted. Missing weight shards fail preflight. Completed artifacts are immutable. Source changes require a new run root; old results remain available. One worker owns each assigned GPU. Fits and calibration decisions precede query evaluation. Query evaluation verifies the restored state against its saved calibration predictions first.

The exporter requires all 1,980 states, validates their linked seals and shared queries, and recomputes metrics from predictions. It revokes stale exports on failure and never replaces missing results with zero or averages only successful runs. Resource records include actual optimizer updates, support/basis visits, state bytes, fitting/extraction time, memory and device. Scientific outcomes and full-matrix runtime remain to be measured.

## Active five-model execution scope (7 November 2026)

At the user's request, exclude Qwen3-VL-8B from execution using `bash scripts/run_aperture_five_models.sh`. The retained models are Qwen2.5-VL-7B, Qwen3-VL-4B, Gemma3-4B-it, InternVL3-8B-Instruct and Phi-3.5-Vision-Instruct. This selects 1,650 states from the immutable six-model parent plan without deleting its checkpoints or audit evidence. The current complete-only exporter still requires the parent plan's 1,980 states; five-model aggregation requires a scoped reporting change before final export.

## Baseline comparison limits

Frozen and LoRA are two baseline approaches. LoRA one/four passes are separate budgets of the same method; Random is a mechanism ablation using the same controller with random directions.

The current recipe matches support/calibration/query data, not compute or trainable parameter counts. APERTURE's four controller updates plus support-basis extraction are not equivalent to four LoRA support passes. Rank16 does not imply equal parameter counts: APERTURE acts on selected visual K/V sites at layers14/27, while LoRA adapts decoder attention projections more broadly. One/four-pass results alone cannot establish superiority over adequately tuned LoRA.

A future fairness extension should predefine a compute-matched comparison (including APERTURE basis extraction), report actual trainable parameters, time, memory and support visits, and give both methods a documented tuning budget using independent validation or support-only splits. Query outcomes must not choose recipes. This extension is proposed, not implemented by the current runner; the present publication records runnable code and support-only validation, not a completed fair-budget benchmark or scientific performance result.
