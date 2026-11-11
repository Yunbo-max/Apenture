# CG-v1 runbook and reviewer mapping

## What changed

The submitted/static APERTURE remains intact. CG-v1 adds an input-conditioned
scalar gate per visual token/site:

`h=z@U; g=sigmoid((h/sqrt(mean(h*h)+1e-6))@w+b)`

`A=alpha*R/(1+||R||F); z_new=z+m*g*(h@A@U.T)`

Train R,w,b; freeze U and every backbone weight. Gate conditioning uses only
current visual activations, not query labels, class identity, validation outcomes
or stored support activations. No loss mix, GRPO or gradient surgery. Question
text appearing after an image is not necessarily visible to its gate.
R=w=b=0 initially, so residual identity holds although g=.5. Gate data gradients
are zero at step one because R=0; CG08's one-pass point does not imply the gate
has already learned input-dependent behavior. Rank16/four-site conditional has
1092 trainable scalars (static Full: 1024); include all bases/gates in state size.

No recoverable historical gated definition/checkpoint was provided or found.
This is an explicit NEW candidate, not a verified implementation of the user's
previous best run. Keep its result identity separate until that provenance is
resolved. Do not quote legacy positive OR negative numbers as CG-v1 results.

## Branch and isolated checkout

Use the original working Linux/WSL VLM environment (do not upgrade an active
training environment). No dependency install, CUDA training or background task
is performed by this delivery. From the existing repository:

```bash
git fetch origin research/aperture-conditional-gated-v1
git worktree add --detach ../Apenture-cg-v1 origin/research/aperture-conditional-gated-v1
cd ../Apenture-cg-v1
```

A worktree does not copy ignored model/data files. Create a NEW local config
pointing at the original prepared assets; this helper reads configs only:

```bash
python scripts/prepare_aperture_gated_config.py \
  --source-repo /ABSOLUTE/PATH/TO/EXISTING/Apenture
```

It creates `configs/aperture_conditional_gated.local.json` and a local absolute-
path base config. It refuses overwrites. Use `--source-base` for an existing
nondefault comprehensive config, and `--prepared-root` for existing E03/E07
prepared assets. Check these paths before execution. Do not upload local configs
containing machine-specific paths. No old trained checkpoint is auto-imported.

Default new root: `runs/aperture_conditional_gated_20261111_v1/`.
Default `execution_batch=1` is explicit and frozen. An externally set
`APERTURE_BATCH` must match; do not infer batch from an old `b16` directory name.
For another batch, create a new config/root and repeat all native audits.
No source/data/rank/precision fallback is performed when resources are absent.

## First representative unit

```bash
python -m pytest -q tests/test_conditional_gated.py tests/test_conditional_gated_integration.py
python scripts/run_aperture_gated.py list
CFG=configs/aperture_conditional_gated.local.json
UNIT=q25--camelyon17-hospital_2--n16--s0

python scripts/run_aperture_gated.py plan  --case CG01 --config "$CFG"
python scripts/run_aperture_gated.py check --case CG01 --unit "$UNIT" --config "$CFG"
python scripts/run_aperture_gated.py audit --case CG01 --unit "$UNIT" --gpu 0 --execute --config "$CFG"
python scripts/run_aperture_gated.py run   --case CG01 --unit "$UNIT" --gpu 0 --execute --config "$CFG"
```

`run` without `--execute` is dry-run. A unit includes all its registered arms;
this is NOT a command to run only the gated arm. The native support audit checks
zero identity, residual gradients, activated-gate gradients, future-answer
independence, payload restoration, off-mask isolation and no-image identity.
Any failed audit blocks fitting/query, without weakening tolerances.

After that unit succeeds, use another unit/model filter or execute a complete
case explicitly. There is no run-all default:

```bash
python scripts/run_aperture_gated.py run --case CG02 --gpu 0 --execute --config "$CFG"
python scripts/run_aperture_gated.py status --case CG02 --config "$CFG"
python scripts/run_aperture_gated.py summarize --case CG02 --config "$CFG"
```

A complete case report requires every planned unit. Runtime filters select work,
not which successful results get reported. Completed stages are reused only
after checksum verification; no `--resume` flag. Incomplete fits are archived and
retrained, not silently resumed without optimizer state. Do not delete plan/DONE
files to force compatibility. Data preparation is explicit:

```bash
python scripts/run_aperture_gated.py prepare --case CG03 --execute --config "$CFG"
python scripts/run_aperture_gated.py prepare --case CG07 --execute --config "$CFG"
python scripts/run_aperture_gated.py run --case CG03 \
  --unit q25--hospital_2--n32--s0 --gpu 0 --execute --config "$CFG"
```

CG03 uses `hospital_2` while the standard panel uses
`camelyon17-hospital_2`. Filtering CG03 by model/domain/seed includes all three
budgets; use the full unit ID for exactly one budget. Preparation requires the
existing official raw data; it does not download model weights or shrink tests.

## Cases

| Case | Scope | Main question |
|---|---|---|
| CG00 | Five models; Hawaii/Libya/H2; seed0; support only | Native gate/mask/gradient/causality contract and notation evidence |
| CG01 | Five models; all declared base tasks; seeds0/1/2 | Frozen, static Full, gated, random-gated, LoRA1/4; cross-model gains and failures |
| CG02 | q25+g34; Hawaii/Libya/H2; seeds0/1/2 | Six candidates EACH for static Full, gated, LoRA-r1/r4/r16 and DoRA-r4 |
| CG03 | q25+g34; H0/H2; total16/32/64; seeds0/1/2 | Nested larger support; static/gated and LoRA-r4/r16, six candidates each |
| CG04 | q25; Hawaii/Libya/H2; seeds0/1/2 | Static Full, conditional, fixed half, learned scalar, random/PCA-gated, K/V-only |
| CG05 | q25+g34; Hawaii/Libya/H2; seeds0/1/2 | Static AND gated alpha={1,2,3} x lr={.01,.05}, four support updates |
| CG06 | q25; Hawaii/H2; seed0 | Frozen/static/gated/LoRA1/4/16; full state bytes, extraction+fit, latency |
| CG07 | q25+g34; TextVQA; seeds0/1/2 | 64 labels=48 fit+16 cal; full5000 validation; Frozen/static/gated/LoRA1/4; greedy no-cache generation |
| CG08 | q25+g34; Hawaii/Libya/H2; seeds0/1/2 | Static/gated/LoRA at1/2/4/8/16 full-support visits; final fit and calibration metrics |
| CG09 | Read-only CG01 q25 four-BRIGHT/three-seed panel | Paired differences and fixed-hash positive AND negative cases |

CG01 uses the actual base manifest catalog (the checked-in base declares 22
subtasks); it is not a claim that 22 independent benchmarks exist or that every
old per-card scope used 22. `plan` prints the exact units/counts. Do not relabel
an old 20-domain snapshot as completion of a new 22-domain plan.

Selection is raw calibration NLL, ties by predeclared name, for both methods.
Target query/retention labels never select alpha, LR, rank, training budget or
stop criteria. All candidates are fitted and saved before selection in each
unit. HPO matches six candidates and labels, NOT FLOPs or parameters. Static and
gated use alpha1/2/3 and LR .01/.05; LoRA/DoRA use baseLR/4,baseLR,4*baseLR and
4/16 passes, with baseLR=1e-4 paired and2e-4 single. Record every failure and cost.
CG05 reports all combinations, not a query-chosen replacement method.
CG08 uses separate fits from the base, not optimizer-restored checkpoints of one
trajectory. Each budget counts as a candidate; neither a flat low score nor a
small parameter count proves less overfitting. Do not swap generation loss for
classification NLL when diagnosing train/validation gaps.

## Evidence files and aggregation

`ROOT/CGxx/units/UNIT/` contains `preflight.json`, `audit/`, `fits/VARIANT/`,
`selection.json`, `selection_receipt/`, `evaluation/VARIANT/`, and `logs/`.
Each fit includes controller/adapter, deployment metadata, final support scores,
calibration scores, fit.json and gate_diagnostics.json. Payload format/mode/alpha/
bases must agree on restore; legacy states are rejected by the gated loader.
Each evaluation replays calibration in a freshly loaded model before query.
Generation never receives answer choices or ground truth as its input.

`ROOT/CGxx/report/` retains total_labels, seed, method family, selected variant,
and separate raw / temperature_scaled / free_generation views. VQA accuracy is
not discarded. Selection across different winning candidates is aggregated by
method family, not by taking the best test candidate. Missing/tampered results
block a complete aggregate. Do NOT use the legacy qz_merge_summary.py for CG.
`fit_calibration_diagnostics.json` preserves final-state metrics and gate stats;
these are not results of new inference-time optimization.

New E10-style visual/text retention is NOT ported in this delivery. The old E10
loader cannot read CG payloads. Do not infer no-forgetting from CG08 or rename
old E10 results. This is separate from the reviewers' explicit baseline/support/
open-generation/cost requests.

## Reviewer-by-question mapping

| Reviewer question | Cases | Necessary manuscript work / boundary |
|---|---|---|
| qmC7: framing vs measured gains | CG01,CG06 | Separate new gated revision from submitted static method; no universal accuracy/speed claim |
| qmC7: mixed results | CG01,CG05,CG09 | Every domain/seed, positive and negative; different protocols never mixed |
| qmC7: narrow novelty | CG02,CG04 | Isolate input-conditioning from simple attenuation and scalar scaling |
| qmC7: Hawaii/Libya explanation | CG09 | Qualitative examples under fixed rules; association not causation |
| cJAm: stronger baselines | CG02 | Small-rank and rank16 LoRA, DoRA. Official LoReFT not enabled: historical frozen-identical outputs require repair/behavioral validation |
| cJAm: larger support / longer LoRA | CG03,CG08 | Keep budgets separate; give LoRA legitimate selection opportunity |
| cJAm: fixed-candidate limitation | CG07 | Short-answer free VQA only; reasoning/dialogue remains unsupported |
| cJAm: fitting slower | CG06 | Include extraction, full state, search cost, unmerged and score-verified merged LoRA |
| SXxo: task definition | CG01,CG03 + writing | Supervised few-shot target-domain adaptation |
| SXxo: K/V location | CG00 + writing | Per-layer projection outputs, widths, masks; downstream normalization remains model-native |
| SXxo: xi,yi,N,Mi,question | CG00 + writing | Define x=(image,question), y=answer, N=fit count, m=visual-token mask; update figure |
| SXxo: stage connections | CG00,CG04 + writing | Support -> U -> train R,w,b -> new image; sensitive subspace is not semantic truth |
| SXxo: overview before equations | documentation | Supplied design text, but no manuscript/PDF automatically edited |

Reviewer mapping is coverage of a proposed experiment, NOT a claim that any
reviewer concern is resolved or any score has improved. All reused query sets
were previously inspected; report this follow-up as exploratory unless a new
independent locked evaluation is supplied. Native GPU results are not present.
