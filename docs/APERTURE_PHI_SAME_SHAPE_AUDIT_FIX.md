# Phi / InternVL CG audit: same-shape answer-token probe (2026-11-12)

## Scope and evidence

Base branch tip inspected: `391ed2de416ae9eedd9f0e2b679bd71740bf942b`.
The server's `DIAGNOSIS_CG_PHI_AUDIT.md` reports gate deltas depending on
candidate token length (same length: zero; different length: about .079/.098).
This fix removes that confound from the audit; it does not claim an independently
reproduced cuBLAS diagnosis, a native H100 pass, or improved scientific scores.

Only audit logic and tests change. The gate formula, U/R/w/b, alpha, rank, layers,
loss, optimizer, support split, BF16 loader and training/query scoring remain
unchanged. No old results, seals, manifests or plan files are edited. `main` is
not modified. The fix also applies to the reported InternVL/H2 failure without
special-casing or excluding any model.

## Exact fix

`src/aperture_conditional_gated/causal_probe.py` prepares one support row once
through the existing `Backend.batch`. It then performs four sequential no-grad
forwards through the existing `Backend.forward_batch`: reference, reference
replay, and two answer-content probes. The native single-row sequence shape is
identical for all four. Each probe changes every token in the EXACT answer span
to an ordinary in-vocabulary token ID. Three allowed IDs are used to make both
probes different from the original and each other at every position. They need
not decode to a task label: this is a causality diagnostic, not a prediction.

No string retokenization, no extra neutral token, no padding, no vocabulary-size
or precision change. Prefix/suffix IDs, Phi negative image placeholders, image
pixels/grids/sizes, attention/position/type tensors and visual mask are identical.
Special/image/EOS/PAD IDs are excluded from replacement. Inputs are cloned before
every forward in case a native image embedder normalizes a temporary ID tensor
in place. Full logits are discarded immediately, never scored as query results.
The native full-logit forward is used consistently for reference AND probes; the
ordinary selective-logit scoring path is unchanged.

The original gate threshold stays **0.005** for all models. Nonfinite/missing
traces, invalid spans, masked answer tokens, empty visual masks, or unstable
same-shape reference replay fail closed. Deliberate answer-content leakage still
fails. A success receipt is stored in each audit row as `causality_probe` with
protocol `same-shape-answer-token-v1`, input shape, answer span, replay delta,
per-site probe deltas and unchanged-field policy. Existing identity/gradient/
payload/off-mask checks remain in the surrounding audit.

This finite probe is evidence on tested support rows, not a universal causal
proof, nor a guarantee of no overfitting. A native failure after this patch must
be investigated; do not raise the threshold to make it pass.

## Verification executed here

- Baseline: the previous 73-test CPU subset passed before edits.
- Red: a causal toy backend with length-dependent prefix noise reproduced the
  old `Visual gate depends on future candidate` failure.
- Green: the same example passes with this fixed-shape check. A second backend
  deliberately leaking answer CONTENT still fails with no DONE seal written.
- Final: 144 selected CPU tests passed, 0 failed, 0 skipped (37 new probe tests;
  existing gate, pipeline, reporting and batch-contract tests included).
- 11 actual CLI subprocess checks matched expected exit codes using a synthetic
  asset catalog (help/list/plans/targeted dry-runs and invalid-argument rejection).
  Python compilation passed.
- Five tests execute the repository's actual `Backend.forward_batch` dispatcher
  for Phi, Qwen2.5, Qwen3, Gemma and InternVL, with tiny PyTorch models replacing
  pretrained weights. They are NOT five native-model acceptance runs.

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, no CUDA and no Transformers/PEFT
installation. Validation uses an isolated fixture assembled from prior delivered
archives. The inherited backend was reconstructed byte-identically to remote
blob `2477229fd1e8fd97c02a0b25d6b4d9b20969cde1`; plan/report/adapter source identities
were checked. The unchanged `fit_coefficients` was extracted from its remote
source for two default-trainer tests; that fixture is NOT shipped as a repository
replacement. This is not a complete repository test run. No independent reviewer
agent or live GPU service was used. Native acceptance remains required.

## Safe server application and targeted replay

Keep running checkouts untouched. Fetch the branch, create a NEW worktree at the
fix commit, and create a NEW output root. Source checks intentionally reject
continuing a frozen old plan under new source. Do not delete `plan.json`/DONE or
copy old audit seals into the new root. Completed model fits/results stay valid
as their recorded old runs; only blocked/missing units need new training.

From the current CG checkout, remember the actual local config you used:

```bash
OLD_REPO="$(pwd)"
# Set this to YOUR active local config; do not guess a different asset layout.
OLD_CFG="$OLD_REPO/configs/aperture_conditional_gated.recheck.local.json"
git fetch origin research/aperture-conditional-gated-v1
git worktree add --detach ../Apenture-cg-phi-fixed \
  origin/research/aperture-conditional-gated-v1
cd ../Apenture-cg-phi-fixed
```

Copy the existing absolute-path local config, changing only the new output root.
The following rejects relative internal asset roots rather than silently looking
for models/data in the new worktree:

```bash
python - "$OLD_CFG" <<'PY'
import json, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
cfg = json.loads(source.read_text())
base_path = Path(cfg['base_config'])
if not base_path.is_absolute():
    raise SystemExit('Use the absolute-path local config produced by prepare_aperture_gated_config.py.')
base = json.loads(base_path.read_text())
if not Path(base['prepared_root']).is_absolute() or any(not Path(m['path']).is_absolute() for m in base['models']):
    raise SystemExit('Normalize model/prepared paths with prepare_aperture_gated_config.py first.')
cfg['run_root'] = str(Path('runs/aperture_conditional_gated_20261112_phi_fix_v1').resolve())
cfg['execution_batch'] = 1
out = Path('configs/aperture_conditional_gated.phi-fix.local.json')
with out.open('x', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
print(out)
PY
```

Run code checks and support-only audits on the three representative units before
any training. These audit commands use the actual base model but do not evaluate
query examples:

```bash
python -m pytest -q tests/test_conditional_gated_phi_probe.py \
  tests/test_conditional_gated.py tests/test_conditional_gated_integration.py \
  tests/test_conditional_gated_final_checks.py tests/test_reviewer_cases.py \
  tests/test_batch_contract_regressions.py
export APERTURE_BATCH=1
CFG=configs/aperture_conditional_gated.phi-fix.local.json
python scripts/run_aperture_gated.py plan --case CG01 --config "$CFG"
for UNIT in phi--bright-hawaii-wildfire--n24--s0 \
            internvl3--camelyon17-hospital_2--n16--s0 \
            q25--camelyon17-hospital_2--n16--s0; do
  python scripts/run_aperture_gated.py check --case CG01 --unit "$UNIT" --config "$CFG" || exit
  python scripts/run_aperture_gated.py audit --case CG01 --unit "$UNIT" \
    --gpu 0 --execute --config "$CFG" || exit
done
```

Only after all three pass, train/evaluate a blocked representative unit:

```bash
python scripts/run_aperture_gated.py run --case CG01 \
  --unit phi--bright-hawaii-wildfire--n24--s0 \
  --gpu 0 --execute --config "$CFG"
```

`run` includes all registered comparison arms and fresh-process calibration
replay. Remove `--execute` for dry-run. Do not launch all CG01 units simply to
work around missing ones; select the known failed/missing unit IDs. New partial
roots cannot independently claim whole-case completion. When combining old and
new results retain unique unit identity, source/audit revision and raw/scaled
views. Do not use the old non-deduplicating qz merger to relabel these runs.

Upload `audit.json`, stage failure logs if any, and the standard public-safe
fit/metrics summaries after the native checks. Do not upload model weights,
private patient data, credentials or raw medical images.
