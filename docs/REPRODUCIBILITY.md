# Reproducibility

Every named result block keeps its model ID, processor, image resolution, support/calibration/query IDs, random seed, tokenization contract, answer span, intervention sites, rank, residual bound, optimizer, visits, updates, and scoring rule. Result blocks with different protocols are not pooled.

Trainable GPU runs are not claimed bitwise deterministic. Frozen predictions must replay exactly; adapted predictions may vary because of accelerator nondeterminism. Original sealed values are retained rather than replaced by favorable reruns. Query labels are excluded from fitting and selection; query-label analyses are marked oracle diagnostics.
