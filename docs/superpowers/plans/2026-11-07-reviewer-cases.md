# Reviewer-specific APERTURE cases implementation plan

Goal: independently launch, resume, seal and report reviewer-targeted experiments without changing the submitted APERTURE method or existing source fingerprints.

Architecture: additive `aperture_rebuttal_cases` package; reuse native comprehensive VLM loading/scoring and original APERTURE extraction/fitting. New case plans, experiment manifests, comparison-unit selection receipts, fit/replay gates, resource measurements and reports live in separate roots. No loss/controller mixing, no trainable basis, no query-based selection.

Spec: `docs/APERTURE_REBUTTAL_CASES.md`.

Constraints: preserve all existing files/results; explicit GPU execution; no model quantization/offload/substitution; independent case/unit/variant directories; missing evidence is missing; full selected-case export only. Strong baselines are VLM adaptations, not claimed official benchmark replications. Open VQA is short-answer generation, not multi-turn reasoning.

1. RED/GREEN pure case catalog and immutable plans: all review questions mapped, independent units, frozen model/data/config/source identity, no hidden all-case launch.
2. RED/GREEN nested support sampling, fixed query population, patient/image/group leakage checks; TextVQA official train/val preparation.
3. RED/GREEN baseline configuration, bounded/unbounded/diagonal ablation operators and mask isolation; reuse original Full for main results.
4. GPU worker: support-only audit before fits, calibration-only selection, independent fresh-process replay before query, raw and temperature scores, fixed-resource timing, official TextVQA scoring.
5. RED/GREEN complete-only aggregates, group-paired comparisons, deterministic qualitative examples, descriptive independent support-seed variation.
6. Verify local added tests and syntax, inspect exact tree diff, publish additive files with guarded fast-forward update. No GPU outcome or full legacy-suite claim.
