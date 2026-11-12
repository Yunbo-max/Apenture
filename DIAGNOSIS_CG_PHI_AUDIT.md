# Diagnosis: CG-v1 causal audit failures for Phi-3.5-vision (and internvl3-hospital_2)

Server-side read-only diagnosis, 2026-11-12. No protocol code was modified.

## Symptom

`audit --case CG01 --unit phi--bright-hawaii-wildfire--n24--s0` (and all other
phi units, plus internvl3--camelyon17-hospital_2) fails with:

```
Conditional-gated case blocked: Visual gate depends on future candidate; causal/prompt audit failed
```

## Reproductions (instrumented replicas on 1xH100, checked code imported unchanged)

1. Faithful replica of `audit.py` per support row: rows with 2-token labels
   ('intact', 'damaged') pass with gate_delta exactly 0.0; the row with the
   1-token label ('destroyed') fails with gate_delta = 0.079 (matches the
   production failure).
2. Length-discrimination test (same row, candidate strings of 1/2/4 tokens):
   - alt='a' (1 tok, same length as label): delta 0.0 PASS
   - alt='damaged'/'intact'/'zz' (2 tok): delta 0.078979 FAIL, IDENTICAL for all contents
   - alt='intact intact' (4 tok): delta 0.098415 FAIL
   => the gate delta depends on the answer LENGTH only, not its content.
3. Raw hidden states (no controller): comparing label run (1576 tok) vs alt
   run (1577 tok) restricted to prefix positions:
   - embeddings (vision_embed_tokens output): image-region max diff 0.0
   - layer 1: image-region 0.25, question-region 0.016
   - layer 7: 1.0 / 0.25; layer 14: 352 / 0.5; layer 27: 336 / 4.0
   => divergence appears at L1 and amplifies through the layers.
4. A/B: stripping the processor's 2D attention_mask (forcing the model's own
   causal branch) changes nothing — the mask is not the mechanism.
5. Same-length control runs are bit-identical (delta exactly 0.0); the model
   runs under torch.use_deterministic_algorithms(True).

## Mechanism

The prefix divergence starts at layer 1 with embeddings that are bit-identical,
a causal attention mask, and no train-mode stochasticity. The remaining
length-dependent quantity is the GEMM shape itself: the bf16 linear layers
(e.g. fused qkv_proj on (1,1576,3072) vs (1,1577,3072)) select different
deterministic cuBLAS kernels/tilings for different M, producing different
rounding for the same rows. The tiny per-layer differences (0.25 at L1) are
amplified by the residual stack (352 at L14). The gate (sigmoid of a linear
functional of the K/V site states at layers 14/27) inherits the noise, which
exceeds the audit's 0.005 threshold for some rows/models (phi, and
internvl3 for some inputs; q25/q34/g34 happen to stay below it).

Conclusion: the gate does NOT see the answer content. The audit's failure is a
cross-length numerical artifact, not a causality violation. The check compares
candidate strings of different tokenized lengths; the model is deterministic
per shape but not bit-stable across shapes at bf16.

## Evidence trail

All runs and logs are on the server:
- worktree: backup/Apenture-cg-checked @ dfdf0bf
- instrumented script: runs/qz_jobs/cg_debug_audit.py (read-only, imports the
  checked modules; executed on qz 1xH100 jobs)
- logs: runs/qz_jobs/cg_debug_audit_phi.log (final run) and the per-attempt
  job history in the qz workspace

## Recommended fixes (for the branch author; not applied on the server)

1. (Preferred) Length-matched causality probe in audit.py: pick the alternative
   candidate with the SAME tokenized length as the row label (fall back to
   padding candidates to equal length with a neutral token). This preserves
   the check's intent (visual state independent of answer CONTENT) and removes
   the cross-shape artifact. All models then compare like with like.
2. Alternatively, document a per-model numeric noise floor and set the
   gate_delta threshold above it (e.g., 0.1) — weaker guarantee, not recommended.
3. Moving the gate to vision-tower features remains an option but is a method
   redesign, not needed to fix this audit.

After any fix: re-run the 105-test CPU suite and repeat the native acceptance
unit (q25) plus one phi unit, then re-freeze plans if source files changed.
