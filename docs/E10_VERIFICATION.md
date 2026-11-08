# E10 software verification — 2026-11-08

This is software evidence, not a model-quality or GPU result.

- 29 new offline tests passed (protocol scope, paired metrics, undefined denominators, old-source tampering, source/query isolation, text formatting/counts, all-zero text masks, prefix-resume validation, original-method identity, state byte accounting, frozen plans and CLI safety).
- Combined local regression: 77 passed, 0 failed, 0 skipped. The additional 48 are the previously delivered case/command contract tests run from the mounted prior patch packages. This is **not** a fresh run of the entire latest repository/native-model test suite.
- The authentic comprehensive configuration used in the contract tests was checked against Git blob `997e2489f38e73c54746ef4fb796df9ff0545004`.
- Red phases observed missing E10 implementation failures and the missing original-method guard; both were followed by passing runs. An initial combined-test attempt lacked the base configuration; adding the verified original file resolved those 16 fixture errors. No tests were removed to hide failures.
- `compileall`, CLI help, case listing and text-preparation dry run succeeded. Source/data/model preflight and native checkpoint replay still need real server artifacts.
- The working container lacks native VLM dependencies/weights and CUDA. No original GPU experiment, dataset download, throughput, no-forgetting outcome or LoRA superiority was measured here.
- All changes are additions outside `src/` and the existing run entrypoints. Existing E01–E09 files, methods, losses, workflow settings, batch implementation and outputs are untouched.

Runtime source, environment, model, manifest and replay checks deliberately fail closed rather than treating these CPU tests as native qualification. See `APERTURE_E10_RETENTION.md` for commands and statistical/claim limits.
