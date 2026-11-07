# Comprehensive experiment suite implementation plan

Goal: execute the five comparison arms (Frozen, LoRA one/four passes, Random, APERTURE) for all six downloaded checkpoints on all downloaded dataset families, with explicit task manifests and no family exclusions.

Architecture: add `aperture_comprehensive` as a separately versioned suite. Reuse the existing bounded controller, numerical metrics, sealed state machinery and native model integrations. Preserve historical suite definitions. Normalize paired and single-image examples while retaining original questions and per-example candidate labels. Phi requires explicit K/V slices of its fused QKV projection.

Constraints: BF16, no offload or quantization; fixed native resolutions; rank16; layers14/27; support-only adaptation and calibration; shared queries across arms/models/seeds; immutable plans; actual model audits before fits and saved-state replay before queries. Existing user edits remain intact.

- [x] Add regression tests for complete coverage, task preparation, native masks, Phi slice isolation, Random persistence and failure on missing shards.
- [x] Implement registry, preparation and locked plans for registered BRIGHT4 (with eligibility counts for all11), CAMELYON17 hospitals0â€“4, PathMNIST, ManipBench Q1/Q2 and Guardian.
- [x] Implement six-backbone paired/single adapters and the five-arm worker with integrity gates.
- [x] Add scheduling, support-only validation, progress and complete-only aggregation commands.
- [x] Run CPU regression suite, prepare/check actual local data, and real GPU audits/fits for model architectures and input formats; document measured validation and remaining full-run work.

Review focus: fused QKV must preserve Q; paired masks must select POST only; candidate options may vary per Q2 example; no query labels in fitting/calibration; interrupted or incomplete states must never be exported as complete.

Implementation choices are made under the user's instruction to fix the previously identified coverage gaps. The new protocol is an extension, not a relabeling of historical results. LoRA1/4 denotes passes, both at rank16.


Validation ledger:

- 262 CPU tests passed across every repository test directory.
- Full preflight passed: 1,980 states, 66 task/seed partitions, six model snapshots.
- All 30 model/arm H2 support fits and fresh-process replays passed; maximum replay delta0, query forwards0.
- Review finding fixed: shared audits now always use the first configured seed, independently of execution filters. Regression passed.
- Source-data corrections: Q2 duplicate options are deduplicated; Guardian repeated frames with different questions remain valid; Guardian support seeds share one fixed query set.
- Scope ruling: retain the four published BRIGHT events. Six extra events lack the marginal counts for the fixed protocol; Bata has no registered split. No budget or resolution was silently reduced.
- All132 model/task GPU support audits passed under `runs/aperture_comprehensive_20261106_v6`; final artifact seals and source identities verified. Full query evaluations remain unstarted. See `reports/comprehensive_code_readiness_20261106.md`.
