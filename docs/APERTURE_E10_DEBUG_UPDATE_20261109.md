# APERTURE E10 debugging update — 2026-11-09

This supplements [the live status report](APERTURE_E10_LIVE_STATUS_20261109.md). It records diagnostic evidence for the weak/negative early scores and the current GPU-0 run. It does not change any method, training setting, or evaluation rule.

## Current run at 2026-11-09 23:01 UTC

- The existing E10 queue PID 106016 and its worker PID 106114 are alive. They are running only on GPU 0.
- The queue status is action 6 of 111: visual / Gemma 3 4B / BRIGHT Hawaii wildfire / 24-shot / seed 0. The unit has not been sealed.
- Frozen and APERTURE outputs and LoRA1 outputs are 300/300 for all four targets. Turkey LoRA1 is 300/300; Hawaii LoRA4 is in progress at 87/300.
- The task log shows no exception. The queue setting is `per_process_allocator_cap: false`; no second run was started against these outputs.

## Debugging the negative result

The early Hawaii APERTURE result was checked against the sealed E01 run, not just against the E10 metric implementation:

- Frozen and APERTURE E10 predictions both contain 300 Hawaii examples and match the E01 evaluation examples, group IDs, labels, score vectors (maximum absolute score difference 0), and prompt-contract hashes.
- All five source-fit seals (frozen, APERTURE, LoRA1, LoRA4, and LoRA-r4) validate against their source records. E01 calibration replay has maximum score delta 0.0. E01 and E10 raw metrics are numerically identical.
- Pairing checks for Hawaii, Libya, Noto, and Turkey found the same sample/group/label ordering, candidate counts, and prompt-contract hashes between frozen and APERTURE predictions.
- This makes a sample-order, prompt, candidate-count, or E10 scoring drift unlikely for these files. It does not rule out a defect in the adaptation method itself, which requires reviewing the training/loading implementation and completing the remaining replay audits.

The measured early scores remain poor and must be retained as results, not tuned away:

| Target | Frozen accuracy / macro-F1 / NLL | APERTURE accuracy / macro-F1 / NLL | Accuracy change | NLL change |
|---|---|---|---:|---:|
| Hawaii | 0.3267 / 0.2273 / 1.5171 | 0.3267 / 0.1907 / 2.1402 | 0.0000 | +0.6231 |
| Libya | 0.3233 / 0.1736 / 1.5400 | 0.3433 / 0.2393 / 2.2022 | +0.0200 | +0.6622 |
| Noto | 0.3067 / 0.2291 / 1.3350 | 0.2933 / 0.1798 / 2.1678 | -0.0133 | +0.8328 |
| Turkey | 0.3433 / 0.1876 / 1.3875 | 0.3400 / 0.2466 / 2.1780 | -0.0033 | +0.7905 |

The raw NLL increase across all four targets indicates substantially worse probability calibration/confidence under the pinned temperature=1 metric. Accuracy changes are small and mixed; their paired 95% bootstrap intervals include zero. APERTURE class-0 recall was zero on these targets, a possible class-collapse/decision issue to examine after the unit seals. LoRA1 also has high prediction disagreement and worse NLL/Brier on the interim Hawaii and Libya comparisons. These are scoped observations from one model/support seed; they are not a final E10 verdict.

## Source-fit audit (read-only; 2026-11-09)

The frozen E10 CLI check completed successfully for `g34--bright-hawaii-wildfire--n24--s0` and reported `gpu_forward: false`. A separate read-only CPU/file audit verified the E01/E02 source-fit seals referenced by the E10 binding and checked support/calibration prediction sample IDs and label IDs against their original manifests. The class order is `intact, damaged, destroyed`; support has 6 examples per class and calibration has 2 per class. No query prediction rows were read, the base-model weights were not loaded, and no model forward was run. The saved APERTURE controller was loaded on CPU for its matrix-norm diagnostic. The JSON evidence is at `/tmp/gemma_hawaii_source_fit_audit.json` outside the checkout.

Raw candidate metrics below use the pinned temperature=1 score rule. Predicted-class counts are ordered [intact, damaged, destroyed]. “C→W / W→C” is the number of source-fit predictions changing from frozen-correct to adapted-wrong / frozen-wrong to adapted-correct.

| Fit | Support accuracy / candidate NLL | Support predicted classes | Support C→W / W→C | Calibration accuracy / candidate NLL | Calibration predicted classes | Calibration C→W / W→C |
|---|---:|---|---:|---:|---|---:|
| Frozen | 0.222 / 1.400 | [3, 15, 0] | — | 0.333 / 1.248 | [2, 4, 0] | — |
| APERTURE | 0.278 / 2.350 | [0, 16, 2] | 1 / 2 | 0.500 / 2.086 | [0, 4, 2] | 1 / 2 |
| LoRA1 | 0.278 / 2.829 | [0, 8, 10] | 4 / 5 | 0.500 / 2.675 | [0, 2, 4] | 2 / 3 |
| LoRA4 | 0.556 / 0.957 | [0, 14, 4] | 0 / 6 | 0.333 / 1.407 | [0, 6, 0] | 1 / 1 |
| LoRA-r4 | 0.389 / 1.148 | [0, 17, 1] | 0 / 3 | 0.333 / 1.140 | [0, 6, 0] | 1 / 1 |

The adapted fits all predict zero class-0 examples on this 6-example calibration set, despite balanced calibration labels. This class-decision pattern is already present in the sealed source fits, before E10 retention evaluation. APERTURE and LoRA1 gain one net correct prediction on calibration but have substantially worse raw NLL; LoRA4 and LoRA-r4 retain the frozen accuracy there. On support, the final mean negative log likelihood of the correct candidate sequence is 43.735 (frozen), 11.372 (APERTURE), 2.831 (LoRA1), 0.965 (LoRA4), and 1.156 (LoRA-r4). The recorded APERTURE mean training loss falls from 36.50 to 10.89 over four updates; LoRA4 falls from 15.35 to 0.803 and LoRA-r4 from 28.53 to 1.017. These training-loss histories are not directly comparable to calibration NLL, and they do not show that the adaptation generalizes.

The E10 source binding identifies APERTURE/LoRA1/LoRA4 as E01 fits and LoRA-r4 as an E02 fit, all on the same 18-support / 6-calibration Hawaii split. E01 uses Gemma 3 layers [14, 27], post-image masking, alpha 3, four APERTURE updates at learning rate 0.05; LoRA1 and LoRA4 use rank 16, alpha 32, with one and four passes respectively. E02 LoRA-r4 uses rank 4, alpha 8, and four passes. The saved APERTURE controller is nonzero (raw-matrix Frobenius norms about 1.77–2.01 across 14:K/V and 27:K/V). Reading the local Gemma 3 config with Transformers 4.57.1 shows both target layers are `sliding_attention` layers (34 layers total, sliding window 1024). This is an experiment-design concern to evaluate prospectively, not proof of an implementation bug; no query-driven layer change was made.

The later batch-path defects reported in the separate local patch (candidate-order reuse, Gemma token-type padding, variable Qwen patch concatenation, and omitted batch identity) do not explain this frozen run based on current evidence: the active checkout remains eabb802, the E10 binding records `query_batch_size: 1`, and the active worker is using that bound source. The separate batch patch has not been applied to this running checkout. This does not establish the patch's CUDA/BF16 behavior, nor does it prove all historical result files are unaffected.

## Scoring implementation review

The E10 metric computes a stable softmax over each row's candidate scores, then NLL, Brier, accuracy, and macro-F1. It validates unique IDs, finite scores, candidate counts, paired sample/group/label order, and query-group bootstrap. The visual backend scores each complete candidate answer by summing its answer-token log probabilities; the historical field name `mean_log_scores` is misleading, but this sum is used consistently for frozen and adapted models. The evaluation therefore currently exposes a real scoring-risk question (raw answer-length-sensitive scores and calibration), not an identified mismatch in the E10 wrapper. No result proves that the adaptation implementation is defect-free; the sealed replay audit remains necessary.

## Remaining checks

After the active unit seals, verify its five source bindings, exact row counts, prompt hashes, metric replay, and paired bootstrap outputs. Continue the frozen E10 queue and run part summaries only after their coverage gates pass. Investigate the adaptation/scoring implementation if sealed replay reveals a mismatch; do not tune on query labels or scores.
