# APERTURE execution/debug audit — 9 November 2026

Audited GitHub snapshot: `14ace7807765060cdd21eb70575a9bbbc05ce8c3`.
This change is for an isolated debug branch. Do not pull it into an active
source-pinned E01/E02/E10 checkout or edit/delete old plans and seals.
No algorithm, U/R parameterization, objective, alpha, learning rate, dataset
budget, split, or negative scientific result is changed to improve a score.

## What the latest run note establishes (and what it does not)

`docs/APERTURE_E10_LIVE_STATUS_20261109.md` reports Gemma/Hawaii/seed0:
Frozen vs APERTURE accuracy 0.3267 vs 0.3267, macro-F1 0.2273 vs 0.1907,
NLL 1.5171 vs 2.1402; 23 correct-to-wrong and 23 wrong-to-correct cases.
Prediction disagreement is 24%. This is not evidence of no forgetting.
The note reports exact source-domain E01/E10 prediction equality and zero
calibration replay delta. That rules against newly introduced E10 reload or
scoring drift for this cell, not an upstream common-mode problem or poor fit.
The note says the running checkout HEAD is eabb802 plus copied E10 scripts;
GitHub main additionally contains the later batch code. A Git HEAD or result
folder name alone does not establish which code/batch trained a checkpoint.
This review has NOT accessed live private server fits or GPU processes.

## Reproduced defects

1. `engine.score`: B>1 reused the FIRST row's candidates for all rows.
   With [left,right] and [right,left], the second row's score coordinates are
   reversed, silently changing its predicted label. With different option sets,
   the native scorer instead raises unknown-candidate errors. This affects
   multirow support/calibration scoring and therefore selection, especially Q2.
   It does NOT explain fixed-candidate Hawaii by itself. The query evaluator
   currently calls score([row]), i.e. one query at a time even when B>1.
   Fix: use native row scoring for heterogeneous chunks; preserve native per-row
   prompt hashes; validate per-candidate prompt-contract consistency.
2. `Backend.batch_rows`: only input_ids/attention_mask were padded, not Gemma
   token_type_ids. Different sequence/answer lengths reproducibly fail at cat.
   Fix: pad all supported token-aligned fields together, retain right padding,
   masks and original answer spans. Reject missing processor fields explicitly.
3. Qwen's flattened patch matrix was unsqueezed as if it were a CHW image.
   Unequal patch counts cannot concatenate; equal shapes gain an unnecessary
   dimension. This does NOT prove wrong numeric outputs for equal-shape Qwen
   batches because the native patch embed may flatten them again. Fix the input
   contract: concatenate patches along axis 0; add an image axis only to Gemma
   CHW inputs. Native GPU equivalence still needs checking.
4. APERTURE_BATCH was neither frozen in the case plan nor checked before fit/
   evaluate. Invalid values silently fell back to 1; changing 1 to 16 did not
   invalidate a sealed plan. Fix strict parsing, record it in the new plan and
   unit, and reject drift before GPU setup. Old evidence is not auto-migrated.
5. Batched records hashed the whole batch contract instead of each row's native
   contract. Fixed above; this is provenance inconsistency, not by itself a
   numerical classification error.

## Settings/implementation differences, NOT proven bugs or fixes

- mean (paired BRIGHT) versus sum (single-image tasks) is inherited. Relative
  L2 is lambda vs lambda/N after writing the objective as a mean. Adam/clipping
  prevent interpreting this as an N-fold update. Do not silently standardize it.
- [14,27], alpha=3, lr=.05, four updates are the fixed original setting, not
  independently validated optima for every backbone. Read final support and
  calibration metrics before blaming alpha or extending training.
- Batch mode changes APERTURE/LoReFT fitting, but LoRA fitting still uses the
  original row-wise train_lora. Actual implementation timing is not proof of
  matched kernel/compute efficiency. Full-query scoring is still row-wise.
- Batched calibration replay repeats the same batching code. Replay alone
  cannot establish equivalence to the native row-wise reference.
- The old passing CPU tests did not exercise these heterogeneous native tensor
  and candidate layouts. They never established real VLM quality or universal
  GPU correctness.

## Diagnose the actual Hawaii source fit without touching the run

The new `scripts/audit_aperture_fit.py` is standalone (stdlib except optional
CPU weights-only controller inspection). It does not import the new training
code and can inspect the original active checkout safely. It verifies source
plans, fit seals and manifests; reports actual source hashes, recorded/frozen
batch size, per-class final support/calibration metrics, prediction transitions,
group/sample/content overlap counts, settings, layer metadata if available,
and optional controller norms. It never reads query prediction metrics or
suggests a best test-selected hyperparameter. No raw patient/sample IDs appear
in the report. Output must be a NEW file outside the sealed result roots.

From the debug checkout, for the unit named in the latest run note:

```bash
python scripts/audit_aperture_fit.py \
  --retention-unit /home/aperture/apenture/runs/aperture_retention_20261108_eabb802_v1/visual/units/g34--bright-hawaii-wildfire--n24--s0 \
  --source-repo /home/aperture/apenture \
  --inspect-controller \
  --output /tmp/aperture_hawaii_source_audit_20261109.json
```

Use the actual server paths if they differ; these are from the repository run
note, not remotely verified in this session. Existing output is never replaced.
Support improves + calibration degrades suggests a generalization issue; both
fail suggests an objective/location/optimization limitation or another bug.
These remain hypotheses until the source artifacts are inspected.

## New experiments only, after native qualification

The existing E01–E09 CLI is unchanged. On a NEW checkout and NEW configured
run_root, set APERTURE_BATCH BEFORE plan/check/run. Keep it fixed thereafter.
For a conservative diagnostic reference use APERTURE_BATCH=1; test 16 on a
separate new root. Neither changing the environment inside an existing sealed
run nor erasing identity checks is a valid migration. Compare native B1/B16
scores AND controller gradients on support before expensive new jobs.

Do not switch ongoing eabb802 E10 to this source: E10 intentionally requires the
original source snapshot. No server process was stopped or restarted here.

## Verification

The two current source files used for reproduction were reconstructed from
connector reads and matched these upstream Git blob hashes before testing:
backend.py ea3b514a1bb026bdf9f76d792fb4006f83817d38;
engine.py e69aeb7e1b3b4e10831a9648723a612a87f483c5.
Initial 13 batch assertions failed; all passed after the scoped fixes.
Four read-only auditor tests failed before implementation and passed after.
A fifth regression caught inherited, unused controller L2 being reported for
Frozen/LoRA; it failed before correction and now passes.
Final: 80 selected CPU tests pass (28 prior case + 29 prior E10 + 18 batching
contracts + 5 audit tests), zero failed or skipped; syntax/CLI help/diff checks
also passed. The tests execute actual production function bodies, using small
processor/scorer fixtures at the native VLM boundary. No model weights,
Transformers/PEFT runtime, GPU, live data audit, benchmark or whole-repository
regression has been run in this environment.
