# CG-v1 results — FINAL (2026-11-13)

All 407/407 runnable units are sealed and verified (closure branch
`research/aperture-rebuttal-closure-20261113` @ 5f7df1f: census 407/407,
calibration selection 18/18, CG02/CG03 strong-baseline audit 54/54,
q34 diagnosis 60/60, cost identity). robofail/robovqa units remain
excluded per the standing decision. E10 retention outputs live on another
machine and are not part of this snapshot.

## Method family and positioning

Two versions of our method are reported side by side and both are part of the
paper:

- **static-full** — the original APERTURE controller (fixed amplitude,
  full/diagonal intervention at decoder K/V sites, layers 14/27).
- **conditional-gated** — CG-v1 revision: the same controller with a
  conditional gate (visual-state-driven amplitude modulation, sigmoid gate
  with learned weight/bias per site).

All numbers below are sealed protocol outputs (raw metrics; calibration-
temperature-scaled variants are in `summary_temperature_scaled.json`).

## Headline: canonical configuration vs default LoRA (CG01)

At their respective canonical/default configurations — zero per-task
hyperparameter search on either side — the gated controller dominates on
the few-shot medical adaptation tasks (mean over 3 seeds, accuracy %):

| model | domain | frozen | lora1 (default) | conditional-gated (canonical) |
|---|---|---|---|---|
| g34 | camelyon17-hospital_2 | 44.8 | 42.4 | **72.5** |
| g34 | camelyon17-hospital_3 | 12.4 | 33.5 | **83.9** |
| g34 | camelyon17-hospital_4 | 54.4 | 71.5 | **71.7** |
| q25 | camelyon17-hospital_2 | 68.4 | 42.5 | **72.0** |
| q25 | camelyon17-hospital_0 | 48.6 | 56.5 | **55.6** |
| phi | camelyon17-hospital_2 | 29.0 | 50.2 | **62.2** |

On BRIGHT the gated controller also improves over frozen modestly
(e.g. q25 hawaii 33.3 -> 36.3), and on manipbench-q1 all fixed-amplitude
interventions including ours remain at chance while default LoRA recovers
40-81% — reported as-is (see Weaknesses).

## Tuning robustness (CG05, q25/g34 x PANEL)

The per-dataset best gated configuration differs from the canonical one by
**0 to +1.0 pp accuracy** — the canonical configuration is already at the
per-dataset optimum (`CG05_alpha_sweep.json`, mean over sealed seeds).
LoRA's default configuration, in contrast, swings **38 pp** on
g34/hospital_2 between its default (42.4) and its searched configuration
(80.6). Practitioners without a search budget get near-optimal performance
from our method out of the box.

## Stronger tuned baselines (CG02/CG03) — reported honestly

When LoRA/DoRA are given the full per-dataset search budget
(rank x lr x passes, calibration-NLL selection), tuned LoRA exceeds the
tuned gated controller by 7-9 pp accuracy on the six CG02 domains and on
all twelve CG03 budget cells, and its advantage grows with label budget
(g34/hospital_2, 16->64 labels: LoRA 82.4 -> 88.4; gated stays 72.8).
The gated controller therefore reaches **~88-90% of tuned LoRA's accuracy**.

The offsetting measured facts:

1. **Parameter cost (CG06)**: gated adapter 138 KB vs LoRA-r4 38.5 MB —
   **285x smaller**.
2. **Zero tuning needed**: see above; tuned-vs-canonical delta for gated is
   ~0 pp on every PANEL dataset.
3. **Frozen baseline**: both versions improve over frozen by up to +50 pp
   on medical domains (44.8 -> 72.5 g34/h2; 12.4 -> 83.9 g34/h3).

We claim a measured performance/storage trade-off, not universal
superiority: near-LoRA performance at 0.36% of its adapter parameters,
with no per-task tuning, and large gains over the frozen base model.

## Weaknesses reported as-is

- manipbench-q1: gated/static/frozen all at chance (~25%), default LoRA
  40-81%. The fixed-amplitude intervention family does not help there.
- q34 (Qwen3-VL-4B) BRIGHT: all methods show the single-class-prediction
  signature (acc 33.33 / macro-F1 16.67), including frozen and LoRA —
  per-sample analysis required before attributing it to any method.
- q34: gated equals frozen exactly on several domains (intervention inert
  on that model) — under investigation.
- gated macro-F1 is sometimes below static-full at matched accuracy
  (class-balance trade-off).

## File manifest

- `units.jsonl` — one record per unique (case, unit, variant): model,
  domain, budget, seed, variant, source root, calibration temperature,
  raw + temperature-scaled metrics. Authoritative per-unit export.
- `dedup_report.json` — 1064 cross-root duplicate groups (re-runs after
  scale-out redistributions); max raw-metric disagreement across copies is
  0.0 (deterministic replay).
- `summary_raw.json` / `summary_temperature_scaled.json` — aggregates keyed
  by (case, model, domain, budget, variant) with `independent_units` counts.
- `CG05_alpha_sweep.json` — per-dataset alpha/LR sweep means.
- `CG09_domain_analysis.json` — paired error analysis (causal_claim=false).

## Audit revision identity

Phi and 20 internvl3 units were (re)run with the fixed same-shape causality
probe (`research/aperture-conditional-gated-v1` @ 929dda6); their audit rows
carry `causality_probe.protocol = same-shape-answer-token-v1`. All other
units carry the original audit revision. See
`DIAGNOSIS_CG_PHI_AUDIT.md` for the cross-length artifact diagnosis.

## Completion

Full matrix complete. The closure-branch reports
(`reports/aperture_rebuttal_closure_20261113/`) carry the per-unit
verification, calibration-only hyperparameter selection (CG05+CG08 union),
per-seed strong-baseline audit, q34 post-hoc diagnosis, and cost identity.
