# APERTURE

**APERTURE: Learning to Refocus Visual Evidence in Frozen Vision–Language Models**

APERTURE fits one reusable, domain-level visual lens from a small labeled support set. The VLM backbone remains frozen and produces fresh activations for every query; APERTURE applies a bounded full-matrix residual only to selected visual-token K/V projection outputs. It does not append demonstrations, retain support activations, or optimize on each query.

This is the canonical runnable release consolidated from `yuhanlydia/StateAdaptation` and the earlier `Yunbo-max/EventTune` project. Historical package names (`eventttt`, `vigor_handoff`) are preserved for artifact replay, while the `aperture` command provides one entry point for the maintained suites.

## What is ready

- **Primary runnable suite:** matched BRIGHT disaster adaptation with Frozen, LoRA-1E, LoRA-4E, Random visual control, and APERTURE.
- **Completed evidence packages:** BRIGHT, CAMELYON17-H2, PathMNIST, selected robot visual-decision tasks, calibration, image-dependence, placement, objective, and residual-dose analyses.
- **Optional extensions:** five-hospital CAMELYON17 replication and controlled rank-16 resource measurement.
- Compact public results and provenance are included. Raw datasets, model weights, credentials, patient metadata, and multi-GB run directories are intentionally excluded.

## 1. CPU validation

```bash
git clone https://github.com/Yunbo-max/Apenture.git
cd Apenture
python3 -m pip install -e '.[test]'
python3 scripts/validate_release.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
python3 -m pytest -q tests tests_*
```

The same validation runs automatically in GitHub Actions.

## 2. GPU environment

Install a PyTorch build appropriate for the local CUDA driver first, then install the training dependencies:

```bash
python3 -m pip install -e '.[train,test]'
```

The locked BF16 7B protocols require a CUDA GPU with at least 24 GB memory. The runners do not silently substitute quantization, a smaller checkpoint, reduced resolution, CPU offload, or a smaller dataset.

Set the local model and dataset paths in the selected configuration before running `prepare` or `check`.

## 3. Run the canonical BRIGHT suite

```bash
aperture prepare   --suite core
aperture plan      --suite core
aperture check     --suite core
aperture run       --suite core --gpus 0
aperture summarize --suite core
```

For multiple GPUs, pass a comma-separated list such as `--gpus 0,1,2,3`. Completed, fingerprint-matching artifacts are resumed rather than overwritten.

## 4. Optional maintained suites

```bash
# Final calibration and robustness protocol
aperture plan --suite final-calibration
aperture run  --suite final-calibration --gpus 0,1,2,3

# Five-hospital Qwen2.5-VL / InternVL3 replication
aperture plan --suite medical-multihospital
aperture run  --suite medical-multihospital --gpus 0,1,2,3

# Rank-16 resource measurement on fixed domains
aperture plan --suite efficiency-rank16
aperture run  --suite efficiency-rank16 --gpus 0
```

## Repository map

- `src/aperture/`: unified command-line interface.
- `src/eventttt/`, `src/vigor_handoff/`, `src/visual_lens/`: maintained implementation and compatibility modules.
- `configs/`: locked experiment configurations.
- `scripts/`: prepare/plan/check/run/summarize entry points.
- `results/` and `reports/`: compact completed, exploratory, and negative-result records.
- `paper/visual_lens/`: current manuscript source and figures.
- `provenance/`: source revision locks and migration manifest.

Read `AGENTS.md`, `docs/CLAIM_FREEZE.md`, `docs/EXPERIMENT_MATRIX.md`, and `docs/RUNBOOK.md` before GPU execution.
