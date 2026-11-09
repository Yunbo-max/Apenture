# APERTURE E10 debugging update — 2026-11-09

This supplements [the live status report](APERTURE_E10_LIVE_STATUS_20261109.md). It records diagnostic evidence for the weak/negative early scores and the current GPU-0 run. It does not change any method, training setting, or evaluation rule.

## Current run at 2026-11-09 22:29 UTC

- The existing E10 queue PID 106016 and its worker PID 106114 are alive. They are running only on GPU 0.
- The queue status is action 6 of 111: visual / Gemma 3 4B / BRIGHT Hawaii wildfire / 24-shot / seed 0. The unit has not been sealed.
- The active GPU process is still evaluating the unit. Noto LoRA1 has advanced to 98/300 predictions; frozen and APERTURE outputs are complete for all four targets, and LoRA1 Hawaii and Libya are complete. The unit has not been sealed.
- The current task log shows no exception.
- The queue setting is `per_process_allocator_cap: false`; do not start another run against the same outputs.

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

## Scoring implementation review

The E10 metric computes a stable softmax over each row's candidate scores, then NLL, Brier, accuracy, and macro-F1. It validates unique IDs, finite scores, candidate counts, paired sample/group/label order, and query-group bootstrap. The visual backend scores each complete candidate answer by summing its answer-token log probabilities; the historical field name `mean_log_scores` is misleading, but this sum is used consistently for frozen and adapted models. The evaluation therefore currently exposes a real scoring-risk question (raw answer-length-sensitive scores and calibration), not an identified mismatch in the E10 wrapper. No result proves that the adaptation implementation is defect-free; the sealed replay audit remains necessary.

## Remaining checks

After the active unit seals, verify its five source bindings, exact row counts, prompt hashes, metric replay, and paired bootstrap outputs. Continue the frozen E10 queue and run part summaries only after their coverage gates pass. Investigate the adaptation/scoring implementation if sealed replay reveals a mismatch; do not tune on query labels or scores.
