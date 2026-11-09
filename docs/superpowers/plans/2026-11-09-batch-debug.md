# APERTURE batch-contract debugging plan

**Goal:** repair reproduced execution errors, not tune APERTURE to query results.
**Architecture:** preserve the row-wise scorer and learning objectives; use exact
per-row candidate contracts and native tensor layouts. Freeze batch size into
new case plans. Provide a standalone read-only source-fit audit for the active
E10 run. Existing fits and plans remain untouched.
**Base:** `14ace7807765060cdd21eb70575a9bbbc05ce8c3`.

- [x] Reproduce with the actual production score/batch_rows/plan functions.
  Thirteen assertions fail before changes. Fixtures stand in for the native
  processor/model only; they do not establish GPU model equivalence.
- [x] Fix heterogeneous candidates, per-row hashes, Gemma token-type padding,
  Qwen flattened patch layout and empty/mismatched batch inputs.
- [x] Strict batch-size parsing and frozen execution batch identity; reject
  resumes or direct fit/evaluate calls with another runtime batch size.
- [x] Add source-fit diagnostic: sealed hashes, actual code fingerprints,
  train/calibration class metrics, class transitions, group/sample overlap,
  actual settings and available controller norms. No query-based tuning.
- [x] Run new and retained CPU tests, compile, CLI checks and diff review.
- [ ] Publish isolated debug branch, not main: active E10 uses source-pinned
  eabb802 checkout. Do not switch or stop the ongoing run from this session.

No GPU, real checkpoint, native Transformers/PEFT integration or performance
improvement is claimed. The GitHub mid-run Hawaii note is evidence of a poor
result, not proof of the cause. Original APERTURE loss, U, R, alpha and data
budgets are not redesigned.
