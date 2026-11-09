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
