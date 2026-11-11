# CG-v1 verification boundaries

Validation was performed in an isolated local source fixture assembled from
prior conversation-delivered source archives, plus the new additive package.
The existing reviewer adapters/CLI/data files were checked against remote blob
identities. This was NOT a full git clone or full repository test suite.

Tests use real PyTorch tensors/Adam, but toy model/backend and stubbed CUDA/asset
interfaces for integration. They cover zero identity; gradients after activating
R; per-token conditioning and batch independence; nonvisual-row isolation;
FP32/BF16 CPU forward; bounded operator; fixed-half / fixed-one controls; strict
payload save/restore; invalid/missing states; support-only audit; answer-suffix
independence and deliberate leakage rejection; fit -> select -> fresh restore ->
evaluate; no query forward during fit; sealed-stage reuse; all ten case CLI
plans; budget-aware/family-aware/raw-scaled/VQA reports; incomplete/tampered
report rejection; independent asset-path configuration.

No real Qwen/Gemma/InternVL/Phi weights, datasets, CUDA device, Transformers or
PEFT native execution was available here. CPU success is NOT evidence of native
GPU integration, favorable performance, absence of overfitting or retention.
Native support-only audits and fresh-process calibration replay are mandatory
before launching the larger suite. Do not relax a failed tolerance to proceed.

The synthetic VQA alternative-answer audit first failed because the constant
probe was outside the backend's candidate membership contract. It was repaired
by adding that constant only to the SUPPORT audit row; no query generation input
or training target was changed. A test deliberately makes visual states depend
on a future answer and confirms the audit rejects it.

No independent reviewer agent was available. Review was performed locally using
source inspection plus regression and command checks. DoRA is an explicit native
baseline dependency; the previously inactive LoReFT baseline is not enabled by
default. The registered evaluator still needs a representative native run.

Run tests in the complete remote checkout:

```bash
python -m pytest -q tests/test_conditional_gated.py tests/test_conditional_gated_integration.py tests/test_reviewer_cases.py
python -m compileall -q src/aperture_conditional_gated scripts/run_aperture_gated.py scripts/prepare_aperture_gated_config.py
```

Final verification: 73 selected CPU tests passed (45 new gated tests plus
28 existing reviewer-case regressions), 0 failed and 0 skipped; 30 actual CLI
subprocess checks matched their expected exit status. Per-file hashes and
JUnit/command records are in the delivered verification archive. There are no scientific performance measurements in that archive.
