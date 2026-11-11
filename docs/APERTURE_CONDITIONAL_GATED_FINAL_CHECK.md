# CG-v1 final recheck — 2026-11-11

This rechecks the explicit CG-v1 candidate at `3bd2b22`, not an unlocated
historical best-performing gate. It fixes execution/reporting contracts only:
the gate formula, loss, sites, rank, alpha/LR grid, initialization, and baseline
budgets are unchanged. Do not relabel old results. Main is not modified.

## Reproduced and repaired

1. **Nondefault execution batch:** the inherited planner read APERTURE_BATCH
   while the new planner separately wrote batch_size. With configured batch=2
   and no environment override, one field became 1 and the other 2; the second
   plan call could reject its own frozen plan. Both unit fields now use the
   explicit config. Runtime binding rejects a different batch before asset
   loading/fitting/evaluation. Default batch=1 is unchanged.
2. **Complete query population:** hashes of produced files alone do not prove
   that a producer scored every registered query. Complete and analysis reports
   now compare ordered prediction IDs to the actual query manifest, including
   label/group identity where applicable. Free VQA checks IDs/groups, not its
   training-only placeholder label_id; answers are not passed to generation.
3. **Fail-closed reports:** an empty scientific plan cannot count as complete.
   Failed/incomplete/corrupt aggregation removes stale aggregate and
   fit_calibration_diagnostics files, and records incomplete_or_invalid.
   A failed error-analysis refresh removes the previous analysis artifact.

Six new regressions were first observed to fail on the old code and pass after
repair. Additional checks cover default single-row training (both mean/sum),
empty selections, failure cleanup and the distinct free-generation contract.

## Executed verification

- 105 selected CPU tests passed, 0 failed, 0 skipped.
- 39 real CLI subprocess checks matched expected statuses (including intended
  rejection of wrong flags, wrong task IDs, incomplete cases and batch mismatch).
- All ten plans were built from the checked-in 22-task comprehensive config;
  a CG01 plan has 330 units / 1,980 arms. This is a plan size, not completed work.
- Python compilation and whitespace checks passed.
- The original 73-test selection was rerun before editing and passed too; the
  additional tests exercise failures absent from that earlier suite.

Validation used an isolated local fixture, NOT a full git clone or the complete
repository test suite. The eight original gated source files matched the remote
Git blob IDs. Required inherited plan/report/adapters/data/generation sources and
comprehensive config were matched to repository blobs. The inspected unchanged
fit_coefficients function was exercised with real PyTorch/Adam on toy data for
the default batch=1 route. Native loader/CUDA/assets were mocked for integration;
no Transformers/PEFT/native VLM weights, private datasets or CUDA GPU were tested.
There was no independent reviewer agent; code review and tests were local.

These checks are NOT a native GPU acceptance test, an improved F1 result, or a
proof of reduced overfitting/forgetting. Keep the native support-only audit and
fresh-process calibration replay gates enabled. Do not weaken tolerances.

## Start safely after upgrading

Do not update a checkout that is actively training. Use a NEW worktree and a
NEW local config/root; do not delete old plan.json or DONE.json to resume with
changed source. From the existing asset repository, in the working environment:

```bash
OLD_REPO="$(pwd)"
git fetch origin research/aperture-conditional-gated-v1
git worktree add --detach ../Apenture-cg-checked \
  origin/research/aperture-conditional-gated-v1
cd ../Apenture-cg-checked
python scripts/prepare_aperture_gated_config.py \
  --source-repo "$OLD_REPO" \
  --out configs/aperture_conditional_gated.recheck.local.json \
  --run-root runs/aperture_conditional_gated_20261111_recheck1

python -m pytest -q tests/test_conditional_gated.py \
  tests/test_conditional_gated_integration.py \
  tests/test_conditional_gated_final_checks.py \
  tests/test_reviewer_cases.py tests/test_reviewer_command_checks.py

CFG=configs/aperture_conditional_gated.recheck.local.json
UNIT=q25--camelyon17-hospital_2--n16--s0
export APERTURE_BATCH=1
python scripts/run_aperture_gated.py plan --case CG01 --config "$CFG"
python scripts/run_aperture_gated.py check --case CG01 --unit "$UNIT" --config "$CFG"
python scripts/run_aperture_gated.py audit --case CG01 --unit "$UNIT" --gpu 0 --execute --config "$CFG"
python scripts/run_aperture_gated.py run --case CG01 --unit "$UNIT" --gpu 0 --execute --config "$CFG"
```

Check generated asset paths first; --source-base can point at the actual
comprehensive source config. Helper writes only new configs, not old data or
weights. Without --execute, run/audit/fit/evaluate are dry-run only. One complete
unit does not qualify for full-case summarize. See the main runbook for CG03/07
preparation and the unchanged reviewer mapping. Legacy E10 still cannot read a
CG checkpoint; capability-retention porting is outside this recheck.
