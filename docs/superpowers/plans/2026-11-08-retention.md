# E10 Capability Retention Implementation Plan

**Goal:** Add read-only capability retention evaluation of sealed E01/E02 fits without changing APERTURE training or old source fingerprints.
**Architecture:** New entry point and package under scripts/ (outside existing src fingerprint). Plans bind original plans, adapter receipts, model assets, evaluation manifests and new evaluator code. Visual-active, text-active/disabled, and explicit-domain module-bank experiments are independent.
**Tech stack:** Python, existing APERTURE loaders, PyTorch, NumPy, pytest. Linux/WSL execution, unquantized native models.
**Spec:** docs/APERTURE_E10_RETENTION.md and the user's accepted E10-A/B/C design.

## Constraints and review focus
- Never train, alter old fits, change query-based selections, or overwrite E01-E09 evidence.
- Source snapshot must match every recorded source file; permit no unrecorded src files. Additions live outside src.
- New-domain APERTURE/LoRA remain active on retention tasks; disabled LoRA is a separate text control.
- No inference labels or old-domain temperature selection. Raw candidate scores only.
- Source task gain must accompany retention; capability retention is not sequential continual-learning proof.
- Medical/BRIGHT group isolation and pixel overlap checks; source replay precedes new query scoring.
- Missing cells stay missing; no zero denominator invention; serialized state includes bases.

## Tasks
- [ ] Write failing offline tests for protocol matrix, source tamper checks, metrics, text normalization, label-free scoring and CLI safety.
- [ ] Implement planning/provenance and text preparation from pinned official datasets.
- [ ] Implement visual/text scoring, source replay, disabled controls, and explicit module-bank retrieval.
- [ ] Implement sealed per-unit outputs and complete-scope reports.
- [ ] Run offline regression, CLI dry runs and syntax checks; inspect final diff; push only additions with expected-head guard.
