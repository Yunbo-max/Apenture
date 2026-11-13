# APERTURE rebuttal closure: calibration-first selection, coverage, provenance

**Scope:** 2026-11-13, based on `research/aperture-conditional-gated-v1` at
`7cb992cc299aab341ab87f206e73fc6f2167a27e`. This is a separate
read-only finalization layer. Do not overwrite the historical `main` E01–E10,
change the gated formula, re-freeze an old plan, or modify active output roots.

## What “our best hyperparameters” means

- **The selected APERTURE configuration must minimize RAW calibration NLL for
  that same model × domain × total-label budget × support seed.** The query/test
  score is never involved in selecting configurations or stopping points.
- CG05 contains **6 independent fit configurations** per method:
  `alpha ∈ {1,2,3} × lr ∈ {0.01,0.05}`, with **4 support-pass updates**.
- CG08 contains `steps ∈ {1,2,4,8,16}` with fixed `alpha=3`, `lr=0.05`.
  These are **independent fits**, not optimizer checkpoints along one run.
- The CG05 + CG08 **union is not a 6 × 5 Cartesian product**. It is
  scientifically incorrect to call its best entry a global optimum. Call it
  **best among the predeclared evaluated candidates**. CG05 and CG08 may be
  combined only when fit, calibration and query SHA256 hashes, model identity,
  mask, sites, layer, rank and support seed all match exactly. Any mismatch
  blocks the comparison; no automatic dataset conversion or fallback.
- Only when *every* predeclared candidate fit has a valid immutable DONE seal
  does the read-only selector emit a winner. Otherwise status is `incomplete`.
- Keep the **canonical (alpha=3, lr=.05, 4 updates)** APERTURE row alongside
  the calibration-selected row. Do not replace historical scores.
- The original Static Full must get the same choice opportunity. Previously
  tuned LoRA/DoRA must remain honest strong baselines; reuse sealed existing
  results, not a weaker default LoRA chosen for a favorable gap.
- Existing query sets were already inspected: mark candidate comparison as
  **exploratory**. A new independent locked test set is needed before claiming
  confirmatory superiority.

## Read-only commands (run in a checkout with this script)

Use the original **absolute** run root recorded in its frozen plans, not the
repository folder or a copy of a run root.

```bash
export CG_ROOT=/absolute/original/runs/aperture_conditional_gated_20261111_v1
python scripts/aperture_rebuttal_finish.py census --root "$CG_ROOT" \
  --output /tmp/aperture_cg_census_20261113.json
python scripts/aperture_rebuttal_finish.py select --root "$CG_ROOT" \
  --cases CG05 CG08 --output /tmp/aperture_calibration_selection_20261113.json
python scripts/aperture_rebuttal_finish.py strong --root "$CG_ROOT" \
  --output /tmp/aperture_selected_CG02_CG03_20261113.json
python scripts/aperture_rebuttal_finish.py q34 --root "$CG_ROOT" \
  --output /tmp/aperture_q34_diagnostic_20261113.json
python scripts/aperture_rebuttal_finish.py cost --root "$CG_ROOT" \
  --output /tmp/aperture_cost_identity_20261113.json
```

The script reads recorded fits and evaluations, verifies DONE file hashes, and
**does not run CUDA, train, or change a result**. Default action prints JSON;
`--output` is opt-in and refuses to overwrite an existing report.
`select` does not open any query prediction or query metrics file. It hashes the
query **manifest** to prevent mismatched protocol reuse. `strong` deliberately
reads only the CALIBRATION-selected and sealed query result for each seed, then
aggregates complete seed panels; it never averages tuning-variant summary rows
or selects by query performance. `q34` is **post-hoc diagnosis** and cannot be
used for hyperparameter selection.

If the same unit is found in CG05 and CG08 with different manifest hashes,
stop and investigate. In particular, comparing configurations that used
**different fit/calibration sets is not a valid single-budget selection**.
Do not bypass the guard to manufacture a larger “best” search.

## Existing unfinished tasks: do not collide with active GPU jobs

The 2026-11-13 `7cb992c` GitHub snapshot says the following are in-flight:
`CG07` Gemma TextVQA seed 1 `lora1` generation evaluation; seed 2 `lora4`;
`CG08` Qwen2.5/Libya seed 2. These are **historical snapshot statuses**,
not a fresh inspection of the remote GPU server. Always run `census` first.

Only **after confirming the corresponding queue has stopped**, use the
**unchanged original source checkout and exactly the original frozen config**,
not this finalization worktree to re-freeze a plan. If an existing CG07 unit
has sealed fits and a sealed selection but an unfinished evaluation, run:

```bash
# Run in the ORIGINAL frozen CG checkout; provide its ORIGINAL config.
CFG=configs/aperture_conditional_gated_v1.json
export APERTURE_BATCH=1  # replace by the exact batch frozen in your plan
python scripts/run_aperture_gated.py evaluate --case CG07 \
  --unit g34--textvqa--n64--s1 --gpu 0 --execute --config "$CFG"
python scripts/run_aperture_gated.py evaluate --case CG07 \
  --unit g34--textvqa--n64--s2 --gpu 0 --execute --config "$CFG"
```

If a specific CG08 fit/selection is missing rather than its evaluation only,
run that exact previously frozen unit through `run`; otherwise use `evaluate`.
Never remove DONE files, change source hashes, force plans, or start concurrent
jobs against the same unit. If a card used a special per-card config or a
nondefault batch, replace `CFG`/`APERTURE_BATCH` with the **recorded values**.
The default config above is an illustration only and may not be the actual
frozen card. The native runner and its immutable seals must decide resumption.

## Qwen3-VL-4B failure diagnosis

`q34` compares verified Frozen vs gated **paired** query predictions and score
vectors, checks the predicted-class distribution, and attaches saved
`gate_diagnostics.json`. Equal predictions alone do not imply identical logits
or lack of gradients. If the read-only audit shows genuine inertness, next do
one original **support-only native audit** on the frozen `q34` unit, verify
visual-token masks and applied K/V sites, and record before/after hidden-state
and candidate-score changes on an independent calibration example. Do not
loosen numerical or causal tolerance or change layer choice after viewing query
scores; that would require a new versioned protocol.

## E10 retention: six DONE is not sixty

Legacy E10 uses **static APERTURE**, not the new CG-v1 payload format. The
previous `main` status (Nov 9) is stale, and subsequent six DONE units were
reported separately but are **not in GitHub**. On the actual original E10
machine/worktree, do the read-only census:

```bash
python scripts/aperture_rebuttal_finish.py e10 \
  --root /absolute/original/runs/aperture_retention_20261108_eabb802_v1 \
  --output /tmp/aperture_e10_census_20261113.json
```

The census checks per-part (visual/text/bank) frozen plan identity, each
recorded job ID, receipt file hashes, plan SHA binding, and
`all_identity_checks_passed` in sealed `result.json`. A missing or corrupted
unit is reported rather than counted as DONE. **Only upload the fresh report
once the actual on-server verification succeeds.** Do not restart the 60-unit
queue just to change documentation; do not claim no forgetting from six units.

## Cost and rebuttal corrections

- `lora4` means **four training passes**, not LoRA rank 4. The default
  `lora4` CG06 configuration has `lora_rank=16`; the read-only `cost` report
  extracts rank, passes, trainable scalars and true serialized bytes from the
  sealed fit artifacts. Do not mix parameter ratio with saved-byte ratio.
- The CG02/CG03 report must aggregate the per-**seed calibration-selected**
  method family. It is invalid to average every hyperparameter variant in
  `summary_raw.json` and call that “tuned LoRA”. The `strong` action verifies
  each choice against every valid candidate calibration NLL and reports
  incomplete seed panels as incomplete.
- CG04 mechanism/CG09 errors already exist. Explain the actual conditional
  gain or lack thereof, including regression and class imbalance.
- CG08 must report both **final fit and final calibration** NLL/F1; a low query
  F1 after longer training alone is not proof of overfitting.
- CG07 is free short-answer TextVQA, not a claim of long-form reasoning or
  dialog. It belongs in cJAm's scope response.
- Every raw/temperature-scaled metric, support budget and model family must
  remain separate; no query-based hyperparameter selection.

## Validation scope

Python stdlib-only script. Tests: `python -m unittest discover -s tests
-p test_rebuttal_finalization.py -v`; `python -m py_compile
scripts/aperture_rebuttal_finish.py`. This is **not** a full CUDA/native VLM
validation. Neither the GPU server nor the original E10 root is accessible
through the GitHub connector; they must be verified on the original machine.