# APERTURE execution contract

1. Read `docs/CLAIM_FREEZE.md`, `docs/EXPERIMENT_MATRIX.md`, and the suite-specific protocol before running GPUs.
2. Do not select data, seeds, methods, stopping budgets, layers, ranks, or learning rates from query outcomes.
3. Preserve all completed results. New runs use a new versioned run root and never overwrite sealed artifacts.
4. Run `prepare`, `plan`, `check`, and real-model support-only audits before training or evaluation.
5. Query labels never enter basis estimation, controller fitting, LoRA selection, temperature fitting, or output-bias fitting.
6. Do not silently substitute quantization, smaller checkpoints, lower resolution, CPU offload, or reduced datasets.
7. Report actual optimizer updates, support visits, extraction visits, model IDs, image sizes, state bytes, fitting time, latency, and failures.
8. Keep oracle and query-label diagnostics separate from primary results.
9. Never publish raw medical/robot data, patient metadata, model files, tokens, or ignored prediction/controller artifacts.
10. A failed audit stops the affected suite. A finite unfavorable result is retained and changes the claim rather than the population.
