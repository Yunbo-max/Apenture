# Conditional-gated APERTURE Implementation Plan

> Execution: inline with test-first checks; no subagent or GPU execution.

**Goal:** ship a separate runnable gated branch without rewriting old results.
**Architecture:** a standalone torch controller and additive CG case package,
reusing native VLM loading, support preparation and scoring contracts.
**Tech Stack:** Python, PyTorch, pytest; existing native Transformers/PEFT runtime.
**Spec:** docs/APERTURE_CONDITIONAL_GATED_DESIGN.md

## Tasks
- [ ] Test and implement gated operator, mask, gradients, payload validation.
- [ ] Test and implement case registry with alpha=1/2/3 and matched baselines.
- [ ] Integrate native fitting/replay, support-only behavioral audit and generation.
- [ ] Preserve budget/seed/raw-scaled/VQA fields in complete-case reporting.
- [ ] Add CLI, configuration, launch examples and reviewer-by-question mapping.
- [ ] Run CPU, fake-backend integration and subprocess tests; publish branch and
      read back exact hashes. Explicitly separate these from native GPU evidence.

## Review focus
No backbone parameters registered; no labels consumed by gate; zero-init gate
learning verified after R update; off-mask identity; changed checkpoint mode
must fail; no legacy state silently loaded; no query scoring before selection;
no accidental root collision; no missing support budgets or VQA metrics.
