# APERTURE

**APERTURE: Few-Shot Refocusing of Visual Evidence in Vision–Language Models**

**Jiaquan Zhang¹, Yunbo Long², WenJie Chen¹, Guangya Hao², Pengze Li¹, Ziheng Hua¹, Yunchen Long³, Zihang Zeng¹, Xi Chen¹**  
¹ Fudan University · ² University of Cambridge · ³ Guangdong University of Technology

APERTURE performs few-shot target-domain adaptation of frozen vision–language models (VLMs) through a reusable, support-fitted representation-level visual lens. From a small labeled support set, it learns a bounded full-matrix residual over selected visual-token key/value (K/V) projection outputs. Each query produces fresh activations. There are no backbone weight updates, appended demonstrations, stored support activations, or per-query optimization.

This is the canonical runnable release consolidated from `yuhanlydia/StateAdaptation` and the earlier `Yunbo-max/EventTune` project. Historical package names (`eventttt`, `vigor_handoff`) are preserved for artifact replay, while the `aperture` command provides one entry point for the maintained suites.

## APERTURE-specific rebuttal entry point

[Exact scope, commands, and verification limits](docs/ICLR_REBUTTAL_ZH.md) · [Citation metadata](CITATION.cff)

The new foreground driver reuses the existing five-hospital and rank-16 resource suites in a fresh, source/config-bound run root. It does not import SPECTRUM or modify APERTURE's learning algorithm or historical results. Configure local paths first, then:

```bash
python scripts/run_aperture_rebuttal.py plan --root runs/aperture_rebuttal_20261104 --gpus 0
python scripts/run_aperture_rebuttal.py run --root runs/aperture_rebuttal_20261104 --gpus 0 --execute
```

Without `--execute`, no experiment subprocess is launched. Each suite executes `prepare -> plan -> check -> run -> summarize`; native support-only audit gates remain in place. Completed BRIGHT, pathology and robot blocks are not automatically rerun. New entry-point tests passed locally; no new GPU result or full historical-suite acceptance is claimed.

## What is ready

- **Primary runnable suite:** matched BRIGHT disaster adaptation with Frozen, LoRA-1E, LoRA-4E, Random visual control, and APERTURE.
- **Completed evidence packages:** BRIGHT, CAMELYON17-H2, PathMNIST, selected robot visual-decision tasks, calibration, image-dependence, placement, objective, and residual-dose analyses.
- **Optional extensions:** five-hospital CAMELYON17 replication and controlled rank-16 resource measurement.
- Compact public results and provenance are included. Raw datasets, model weights, credentials, patient metadata, and multi-GB run directories are intentionally excluded.

## 1. Local CPU validation

There is no GitHub Actions workflow. Validation is run locally and explicitly.

```bash
git clone https://github.com/Yunbo-max/Apenture.git
cd Apenture

# Install a CPU PyTorch build for local tests.
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -e '.[train,test]'

bash scripts/check_local.sh
python3 -m pytest -q tests/test_rebuttal_entrypoint.py
```

The local checker validates the release structure, compiles Python sources, checks shell syntax, runs every CPU test directory, verifies the unified CLI, and generates the core experiment plan without loading a model.

## 2. GPU environment

On the experiment machine, install a PyTorch build appropriate for the local CUDA driver first, then install the project:

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
- `scripts/`: prepare/plan/check/run/summarize entry points and local validation.
- `results/` and `reports/`: compact completed, exploratory, and negative-result records.
- `paper/visual_lens/`: historical manuscript source snapshot, not the separately delivered author-version ZIP/PDF; see the build limitation in the rebuttal guide.
- `provenance/`: source revision locks and migration manifest.

Read `AGENTS.md`, `docs/CLAIM_FREEZE.md`, `docs/EXPERIMENT_MATRIX.md`, and `docs/RUNBOOK.md` before GPU execution.
