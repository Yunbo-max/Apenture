# APERTURE: final reviewer-specific experiment cases

**Method freeze.** Keep the original support-gradient second-moment basis, frozen backbone/fixed basis, bounded Full K/V controller, original token-mean answer loss, rank16 and layer indices14/27. No candidate loss, loss mix, controller mix or learned basis is added. Hyperparameter sensitivity and named mechanism ablations are separate experiments, not silent replacements of the submitted method.

This is an additive package on top of `60fb37073cd7a043a58ebd545b7702df64616851`. Existing source files, scientific results, six-model parent plans, and active run roots are not edited. No GitHub Actions workflow is introduced. The new runner imports the original native VLM backend and original APERTURE extraction/fitting. It does not claim that CPU software tests establish VLM performance.

## Cases and fixed scopes

Every case is independently invoked and saved. Model abbreviations: q25=Qwen2.5-VL-7B, q34=Qwen3-VL-4B, g34=Gemma3-4B, internvl3=InternVL3-8B, phi=Phi3.5-Vision. Qwen3-VL-8B is NOT executed.

| Case | Scope | What it answers |
|---|---|---|
| E01 | Five models x existing22 tasks x Frozen/LoRA-1pass/LoRA-4passes/Random/APERTURE x3 support seeds =1,650 states | Cross-model/domain evidence and negative cases.22 tasks are not22 independent datasets. Own five-model export; does not require the excluded sixth model. |
| E02 | q25+g34; Hawaii, Libya, H2;3 seeds. Frozen; LoRA rank1/4/16; DoRA rank4; official LoReFT operator rank4; APERTURE. Four predeclared candidates per trainable family. | Stronger baselines and a bounded tuning budget.450 fitted states;126 selected query states. Four candidates per LoRA rank, not twelve candidates secretly competing as one row. |
| E03 | q25+g34; hospitals0/2; total labels16/32/64;3 seeds. Frozen/APERTURE/LoRA-r4/LoRA-r16. | Larger support and longer/tuned LoRA.36 comparison units,468 fits,144 selected query states. Nested fit and calibration separately, fixed500 query patches, patient-disjoint roles. This is a new curve protocol, not a replacement of old H2 results. |
| E04 | q25; Hawaii, Libya, H2;3 seeds; Frozen, APERTURE, Random, activation PCA, same-Frobenius-bound Diagonal, unbounded Full, K-only, V-only | Gradient basis, cross-direction mixing, bound, and K/V contributions.72 query states. Negative results/failed fits remain recorded. |
| E05 | q25+g34; Hawaii, Libya, H2;3 seeds; Frozen and alpha{1,3} x LR{0.01,0.05};4 updates | Whether the original method is configuration-sensitive.90 states. Report all settings and independently calibration-selected setting; never choose a query winner. |
| E06 | q25; Hawaii+H2; seed0; Frozen, LoRA1/4/16 passes, APERTURE | Full saved-state bytes, extraction/fitting costs, update/support visits and memory;2 warmups,12 full candidate-scoring queries,3 timing repeats. Both unmerged and score-verified merged LoRA latency are attempted.10 states. |
| E07 | q25+g34; official TextVQA;3 support seeds;64 labels=48 fit+16 calibration; all5,000 validation questions; Frozen, LoRA1/4 selected by calibration, APERTURE | Truly free short-answer VQA, not candidate scoring.24 fits,18 selected query states. No external OCR or candidate answers. All arms use identical greedy no-cache decoding,max32 tokens. |
| E08 | Read-only analysis of E01's q25/four-BRIGHT/three-seed panel; all five arms required for each unit | Class-level trade-offs, paired accuracy differences with group bootstrap, deterministic examples of fixes AND regressions. Does not wait for non-BRIGHT models/tasks to finish. |
| E09 | Five models; Hawaii/Libya/H2; seed0; no fitting or query scoring | Native layer/site dimensions, visual-mask contracts, zero-residual identity and gradient/row-isolation audit.15 audit units plus the method-writing checklist below. |

The full catalog is substantial; there is deliberately no `run all` default. Existing results do not get relabeled as results of these new cases. Do not launch E01 again merely because an equivalent historical study already exists: first decide whether that historical evidence will be retained as its own protocol. This runner does not auto-import or silently trust old prediction directories.

## Exact correspondence to reviewers

R1 = qmC7; R2 = cJAm; R3 = SXxo. A coverage mapping is NOT a claim that reviewer concerns have already been resolved.

| Question | Cases | Required explanation beyond experiments |
|---|---|---|
| R1.1 Framing does not match accuracy/storage gains | E01,E06 | Frame storage-performance trade-off; no universal accuracy or speed claim. |
| R1.2 Mixed results/regression versus LoRA | E01,E08 | Clearly distinguish comparison with Frozen from comparison with LoRA; retain every domain/seed. |
| R1.3 Narrow novelty | E02,E04 | Distinguish task-sensitive subspace, visual-row selection and bounded reusable controller. Do not claim gradient eigenspaces or ordinary correct-answer loss are new. More data alone does not prove novelty. |
| R1.4 Why Hawaii/Libya/Noto/etc differ | E08,E04 | Per-class confusion/recall and fixed-hash selection of both fixed and regressed examples. Associations are not causal explanations. |
| R2.1 Stronger baselines | E02 | Both LoRA schedules are the same baseline approach. Random is an ablation. LoReFT is the official operator in a declared VLM placement, not an official VLM benchmark reproduction. |
| R2.2 Larger support; more trained LoRA | E03,E02 | Count all fit+calibration labels. Keep query fixed and support roles nested. Use only calibration for selection. |
| R2.3 Candidate classification only | E07 | TextVQA tests free short-answer generation. **Complex reasoning and multi-turn dialogue remain unestablished and must be scoped out explicitly**, not claimed to be covered by the dataset name. |
| R2.4 Fitting slower than LoRA | E06 | Acknowledge existing timing. Include basis extraction and all candidate-search costs; show actual Pareto trade-off rather than redefine speed. |
| R3.1 Few-shot versus cross-domain generalization | E01,E03+writing | Define supervised few-shot target-domain adaptation, not novel-class episodic meta-learning or unsupervised TTA. |
| R3.2 Which visual K/V states/layers? | E09+writing | Decoder K/V projection outputs at explicit code indices, selected image-token rows; not support KV-cache retrieval. Include model-specific normalization/position operations after the hook. |
| R3.3 x,y,N,M and missing question | E09+writing | Add notation table and image+question to the method overview/Fig.2. y is the answer, M is a token-position mask, not segmentation labels. |
| R3.4 Link between support gradients,directions,controller | E04,E09+writing | First stage selects sensitivity subspace; second stage learns a signed Full transform. No semantic-correctness guarantee. |
| R3.5 Need overview before equations | E09+writing | Add the overview below before Section3's component descriptions. |

## Run independently

Use a NEW checkout/virtual environment for this addition; do not alter the source of a running old experiment. Keep the original native model snapshots and image processing. The host needs the existing training dependencies, and E02 additionally requires the official `pyreft` package and its dependencies. PEFT must support `use_dora`. Installing dependencies into a new environment is preferable to upgrading an active environment. Source and library versions are bound into fitted-state evidence.

```bash
# From the Apenture repository; no GPU is touched by list or plan.
python scripts/run_aperture_cases.py list
python scripts/run_aperture_cases.py plan --case E02
python scripts/run_aperture_cases.py check --case E02 --model q25 --domain camelyon17-hospital_2 --seed 0

# One complete comparison unit, one GPU. No --execute means a dry run.
python scripts/run_aperture_cases.py run --case E02 --model q25 --domain camelyon17-hospital_2 --seed 0 --gpu 0 --execute

# Or independently launch the remaining fixed E02 units on a selected GPU.
python scripts/run_aperture_cases.py run --case E02 --gpu 0 --execute
python scripts/run_aperture_cases.py status --case E02
python scripts/run_aperture_cases.py summarize --case E02
```

`--unit <exact-id-from-plan>` selects one registered unit. `--model`, `--domain`, and `--seed` are execution filters, not changes to the declared report population. A unit contains its complete baseline/candidate comparison. If only some units finished, `status` reports missing/failed units and `summarize` refuses a complete scientific aggregate. Individual sealed outputs remain accessible.

Other cases launch separately:

```bash
python scripts/run_aperture_cases.py run --case E01 --gpu 0 --execute
python scripts/run_aperture_cases.py run --case E04 --gpu 0 --execute
python scripts/run_aperture_cases.py run --case E05 --gpu 0 --execute
python scripts/run_aperture_cases.py run --case E06 --gpu 0 --execute
python scripts/run_aperture_cases.py audit --case E09 --gpu 0 --execute
python scripts/run_aperture_cases.py run --case E08 --execute
```

Use separate invocations and different GPUs for independent units/cases. OS locks prevent two invocations from owning the same unit or the same GPU under the shared new run root. They release on process exit; no manual deletion of live lock files. Different run roots or other programs are not globally locked: do not assign an already occupied GPU. Execution is foreground; nothing runs unattended after this conversation.

`audit`, `fit`, and `evaluate` are also separate actions. `fit` performs audit+fitting+calibration selection, but no queries. `evaluate` requires the sealed selection. Every evaluation restores the fitted state in a fresh process and compares calibration scores before touching query images. A failed replay stops that cell; tolerance is not widened automatically.

## Input preparation

E01/E02/E04/E05/E06/E09 consume the already prepared comprehensive manifests described in `configs/aperture_comprehensive_v1.json`. They do not overwrite them. If those manifests are absent, use the original documented comprehensive `prepare` command first. `check` fails rather than inventing or shrinking a dataset.

Edit only the local asset paths in `configs/aperture_reviewer_cases_v1.json` BEFORE creating any case plan. A changed source/config requires a new output root.

E03 prepares its own fixed patient-disjoint CAMELYON17 pools from the complete official raw metadata. It reuses the existing official-metadata partitioner and verifies nestedness and shared query IDs:

```bash
python scripts/run_aperture_cases.py prepare --case E03 --execute
python scripts/run_aperture_cases.py plan --case E03
python scripts/run_aperture_cases.py run --case E03 --gpu 0 --execute
```

E07 needs the official `TextVQA_0.5.1_train.json`, `TextVQA_0.5.1_val.json`, and local images. Set their paths in the new config. Preparation uses only official TRAIN for support, excludes validation image groups/content, uses image-disjoint fit/calibration and all5,000 validation questions. The majority normalized human answer is the training target. It downloads only the pinned official MMF evaluator source, not images or model weights; its Git-blob SHA is verified before execution.

```bash
python scripts/run_aperture_cases.py prepare --case E07 --execute
python scripts/run_aperture_cases.py plan --case E07
python scripts/run_aperture_cases.py run --case E07 --gpu 0 --execute
```

Official evaluator: `facebookresearch/mmf/mmf/utils/m4c_evaluators.py`, blob `1be4fc7acc8920d6e8224509b808821c209eec02`. Official LoReFT operator: `stanfordnlp/pyreft`, class `LoreftIntervention`. New VLM placement is decoder block INPUT, selected visual rows at14/27. It uses the official learned rotation/source parameterization, not the old fixed-basis hidden-state control. No external OCR is provided to VQA. Its no-cache decoder is a correctness-first extension, not a claim of optimized generation latency.

## Output, integrity and resume

```
runs/aperture_rebuttal_cases_20261107_v1/
  E02/
    plan.json
    units/q25--camelyon17-hospital_2--n16--s0/
      preflight.json
      audit/audit.json, DONE.json
      fits/<variant>/controller.pt OR adapter/, deployment.json,
                     calibration.jsonl, support_final.jsonl, fit.json, DONE.json
      selection.json, selection_receipt/
      evaluation/<selected-variant>/predictions.jsonl, metrics.json, DONE.json
      logs/<stage-and-variant>.log, .job.json, .exit.json
    report/coverage.json, results.json, results.csv, summary.json
```

Re-run the same command to resume. Completed artifacts are hash-verified, not overwritten. Incomplete fitting attempts are archived before a deterministic refit; they never become hidden additional search candidates. Query scoring resumes only a correctly bound prediction prefix. A malformed partial JSON line fails visibly rather than being silently repaired. Source, model-shard, manifest, image-content and state fingerprints guard replay. Search choices are sealed BEFORE query evaluation. `summary` never turns missing cells into0, averages only successful subsets, or silently selects the best query score. A stale complete scientific export is revoked before revalidation.

The report contains raw and calibration-temperature metrics, all state bytes including fixed bases and deployment metadata, measured fitting/basis costs, actual visits/updates, and independent SUPPORT-seed means/SD. Optimization seeds remain fixed. These are not three independent pretraining runs. E08 uses query-group-paired bootstrap; with few patients/tile groups its intervals are weak and flagged. PathMNIST grouping does not establish patient-disjoint evaluation. Q2 candidate-position macro-F1 is descriptive; accuracy/NLL are primary.

E02 gives each family four prespecified calibration candidates. Equal candidates/labels do not imply equal compute. Rank1/4/16 LoRA are shown separately, not an undisclosed best-of12 winner. E06 additionally selects the best-calibration LoRA schedule whose measured fitting time does not exceed APERTURE's measured fitting+extraction time; if none is feasible this field is null. This is a budget comparator, not exact equal-GPU-time evidence. All candidate-search costs remain visible. Timing includes state-save/update checks; model loading, audit, and calibration are separately excluded from the fitting timer.

## Suggested reviewer-facing method overview

> We study supervised few-shot target-domain adaptation. Given a small support set of image–question–answer examples, APERTURE first estimates loss-sensitive subspaces at selected visual-token key/value projection outputs of a frozen VLM. Keeping these subspaces fixed, it then fits bounded residual transformations using the same fitting support. At evaluation, the learned transformations are reused on each new image's freshly computed activations; support demonstrations, stored support activations, and per-query optimization are not used. The method is designed for a compact adaptation-state/performance trade-off, not guaranteed semantic discovery or universal superiority to LoRA.

| Symbol | Definition to put before the component equations |
|---|---|
| x_i | image(s) AND the question/prompt |
| y_i | labeled answer string, not a segmentation mask |
| N | number of fitting-support examples; calibration labels counted separately |
| M_i | token-position mask identifying eligible image rows from processor metadata |
| z^{l,K/V} | selected decoder projection-output state; specify shape and code index |
| U | support-derived fixed orthonormal basis |
| R | learned small Full matrix |
| A | bounded transformation computed from R and fixed alpha |

Revise Fig.2 to show the question alongside the image. These notes do not automatically edit or recompile the manuscript. Keep the ICLR revision anonymous, distinguish author-version arXiv files, and update the AI disclosure truthfully. Rebuttal comments should give one short answer and one table/figure reference per concern, not paste this runbook.

## Verification and unresolved limits

Only new local CPU regression tests and syntax checks are claimed for this delivery. A full legacy-suite run, model download, real GPU execution, LoReFT/DoRA integration qualification and full TextVQA run have NOT been performed here. Every native experiment must pass its support audit and fresh-process replay on the target host. Existing historical132 audits are not audits of this new package.

No new scientific scores are filled in. Complex reasoning and multi-turn dialogue remain explicit scope limitations; novelty and clearer presentation still require writing. Do not label those concerns “solved by experiment.” Use native GPU qualification first, then E02/E03/E04/E06, while existing breadth experiments proceed separately. E07 is supplementary free-generation evidence, not a prerequisite to rewriting the explanation.
