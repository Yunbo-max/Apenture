# APERTURE E10 live run and Hawaii diagnostic — 2026-11-09

This is a mid-run status and a scoped diagnostic, not a completed E10 report. No model, method, loss, data split, or experiment setting was changed.

## Frozen source and runtime correspondence

- The Linux checkout reports source HEAD `eabb8022d99bc34c124284d4792694b93c2ebc89`.
- The E10 runner files used in the checkout match GitHub `main` by Git blob hash: `scripts/run_aperture_retention.py` (`b1636c84748fe676e8f5d19884469d9c47fefa53`), `scripts/aperture_retention/worker.py` (`5f23c739bf8a945172d74d6fdf95865bab18dd7f`), `cli.py` (`27b5be30513293ee5ffd2a48916ccb47952384b4`), `metrics.py` (`7368fdb16b7f188fcb8b5a139254e2f4d3287a32`), `protocol.py` (`27689a0dcf2471347b293d02451aa3e1da01f79f`), `report.py` (`7894f8bf0574a988faedad79b3e3e5e03d47b7a1`), and `text.py` (`168d52ec89b1f18a4450c712bcb5119f4d251ad2`).
- E10 reuses the sealed E01/E02 source fits. In the current unit binding, seals for `frozen`, `aperture`, `lora1`, `lora4`, and `lora-r4` all match their source `DONE.json` records.

## Live execution

As observed at 2026-11-09 21:16 UTC:

- The GPU-0 queue process (PID 106016) and its E10 worker (PID 106114) were alive at action 6 of 111. The active unit is Gemma 3 4B, Hawaii wildfire, 24-shot, seed 0.
- GPU 0 used 77,559 MiB of 97,887 MiB, with 97% utilization. The queue reports `per_process_allocator_cap: false`.
- In the active unit, all four `frozen.jsonl` target predictions have 300 rows each. APERTURE predictions have 300/300 Hawaii rows and 275/300 Libya rows at this observation. The unit has no `DONE.json` yet.
- The E10 output root is `runs/aperture_retention_20261108_eabb802_v1/`. The controller checkpoint and logs are kept under `/home/aperture/experiment-control/e10-20261108/`.
- The last sealed-unit census was 3/60: visual 1/18, text 1/18, bank 1/24. The active unit is not included in that count.

## Hawaii source-domain diagnostic

All figures below are for the 300 balanced Hawaii query examples, using the E10 raw-score metric (`temperature=1`).

| Method | Accuracy | Macro-F1 | NLL | Brier |
|---|---:|---:|---:|---:|
| Frozen | 0.3267 | 0.2273 | 1.5171 | 0.8600 |
| APERTURE | 0.3267 | 0.1907 | 2.1402 | 0.9286 |

The paired native comparison found zero accuracy change, macro-F1 change -0.0365, NLL change +0.6231, 23 correct-to-wrong and 23 wrong-to-correct cases, with 24% prediction disagreement. Each of the three classes has 100 queries. APERTURE class recalls were 0%, 93%, and 5%; frozen recalls were 16%, 82%, and 0%.

This is a weak APERTURE result on Hawaii, but the checks point to a genuine result rather than an E10 data/loading/scoring drift: both E10 files (frozen and APERTURE) exactly match their sealed E01 evaluation files across all 300 sample/group/label tuples, score vectors (maximum absolute difference 0), and prompt-contract hashes. The five source fit seals verify. The original E01 raw metrics are numerically identical to E10. E01 also records calibration replay max delta 0.0. E10 intentionally reports raw scores without applying the old query temperature; that is the frozen E10 protocol.

This finding is limited to Gemma/Hawaii/seed 0. It does not establish the result for the other visual units, text, or bank experiments. Preserve it as a negative result; do not tune the method to this query set.

## Next checks

1. Let the current GPU-0 queue finish the whole unit and verify its final seal, exact population counts, source replay records, and paired bootstrap report.
2. Continue all missing E01/E02 source preparation and the remaining visual/text/bank units from the existing queue without deleting or overwriting sealed outputs.
3. Run each full-part summary only after its coverage gate passes; then audit all 60 unit seals and the three reports.

At publication time this run was still active. No claim is made that the GPU run or full E10 experiment has completed.

## Live continuation update — 2026-11-09 21:36 UTC

- Re-polled the existing GPU-0 queue PID 106016 and E10 child PID 106114; both are alive at action 6/111. The active Gemma 3 4B / Hawaii / 24-shot / seed-0 unit is still running.
- Frozen predictions remain 300/300 for each of four target domains. APERTURE predictions are complete at 300 each for Hawaii, Libya, and Noto; Turkey APERTURE has advanced to 43 rows. No unit seal has been observed yet.
- GPU 0 reports 77,559 / 97,887 MiB and 98% utilization. The queue has no per-process allocator cap. A separate workload is visible on GPU 1; this E10 queue remains assigned only to GPU 0.
- This confirms forward progress: Turkey APERTURE output appeared and grew from absent to 43 rows during the last observation interval. No current exception is visible in the task log.
- Full E10 remains incomplete; this update does not change the scoped Hawaii finding or claim any method improvement.


## Live continuation update — 2026-11-09 21:46 UTC

- Re-polled the existing queue PID 106016 and E10 task PID 106114 on GPU 0; both remain alive at action 6/111. The Gemma 3 4B / Hawaii / 24-shot / seed-0 unit remains in progress.
- Turkey APERTURE predictions have advanced from 207 to 237 rows since the last checkpoint. The unit has not emitted its completion seal. Frozen and previous target outputs remain preserved.
- GPU 0 reports 77,559 / 97,887 MiB, current utilization 67%, and the queue still has per_process_allocator_cap=false. No new task exception is present in the log.
- This is a live progress snapshot only. The 60 E10 units and valid summaries are still outstanding; the Hawaii negative result remains an exact reproduction of the sealed E01 result.


## Live continuation update — 2026-11-09 21:50 UTC

- Queue PID 106016 and task PID 106114 remain alive on GPU 0, action 6/111. The active Gemma 3 4B / Hawaii / 24-shot / seed-0 unit is not sealed yet.
- The APERTURE predictions for all four visual targets (Hawaii, Libya, Noto, Turkey) have now reached 300/300 rows. Frozen predictions are also 300/300 on all four. These are the first two method files only; the planned unit methods also include lora1, lora4, and lora-r4, so the full unit comparison is still running.
- The queue still reports no per-process allocator cap. Recent task output shows model loading and no exception.
- This checkpoint records a completed prediction stage, not a completed unit or full experiment. The full 60-unit scope and valid summaries remain outstanding.


## Read-only four-target diagnostic — 2026-11-09 21:59 UTC

For the active Gemma 3 4B / Hawaii / 24-shot / seed-0 unit, I recomputed paired raw metrics from the already complete frozen and APERTURE prediction files for all four BRIGHT targets. Each pair has 300 rows, identical sample/group/label order, identical prompt-contract hashes, and matching candidate counts. Accuracy-delta 95% paired query-group bootstrap intervals all include zero. Results:

| Target | Frozen acc / macro-F1 / NLL | APERTURE acc / macro-F1 / NLL | Accuracy delta | NLL change |
|---|---|---|---:|---:|
| Hawaii | 0.3267 / 0.2273 / 1.5171 | 0.3267 / 0.1907 / 2.1402 | 0.0000 | +0.6231 |
| Libya | 0.3233 / 0.1736 / 1.5400 | 0.3433 / 0.2393 / 2.2022 | +0.0200 | +0.6622 |
| Noto | 0.3067 / 0.2291 / 1.3350 | 0.2933 / 0.1798 / 2.1678 | -0.0133 | +0.8328 |
| Turkey | 0.3433 / 0.1876 / 1.3875 | 0.3400 / 0.2466 / 2.1780 | -0.0033 | +0.7905 |

This points to worse raw-score NLL across these four targets, with mixed macro-F1 and small accuracy changes. It does not by itself indicate a pipeline bug: pairing, prompts and candidate dimensions agree, while the large score changes reflect the adapted model's outputs. Hawaii is additionally an exact reproduction of sealed E01 predictions. Treat this as an interim negative transfer/calibration finding for this one support seed; the native E10 replay/seal audit and the remaining LoRA methods are still pending. No settings were changed and no query-set tuning was performed.


## LoRA1 Hawaii interim comparison — 2026-11-09 22:07 UTC

The lora1 prediction file for the same 300 Hawaii queries is complete while the task continues with Libya. Read-only paired comparison against the frozen predictions found matching sample/group/label ordering, candidate counts and prompt-contract hashes.

| Method | Accuracy | Macro-F1 | NLL | Brier |
|---|---:|---:|---:|---:|
| Frozen | 0.3267 | 0.2273 | 1.5171 | 0.8600 |
| APERTURE | 0.3267 | 0.1907 | 2.1402 | 0.9286 |
| LoRA1 | 0.3600 | 0.2886 | 2.8334 | 0.9430 |

LoRA1 has +0.0333 accuracy relative to frozen, but its paired query-group bootstrap interval is [-0.0712, 0.1538], which includes zero; it changes 62.33% of predictions and raises NLL/Brier. This is an interim raw-score observation for one seed, not the sealed E10 unit verdict. The current unit continues across the remaining target domains and methods. No query tuning or setting changes were made.


## Live continuation update — 2026-11-09 22:19 UTC

- Existing queue PID 106016 and worker PID 106114 are still alive on GPU 0 at action 6/111. Active unit remains Gemma 3 4B / Hawaii / 24-shot / seed 0.
- lora1 Hawaii predictions are complete at 300/300; lora1 Libya has advanced from 173 to 215/300. No unit seal yet. Task log tail shows the active unit routing line and no exception.
- GPU 0 reports 77,625 / 97,887 MiB, 98% utilization, 87 C. Queue reports per_process_allocator_cap=false.
- This is a progress checkpoint, not a completed unit: lora1 still needs the other target domains, then lora4 and lora-r4; full E10 summaries remain pending.
