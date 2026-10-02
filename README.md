# APERTURE

**APERTURE: Learning to Refocus Visual Evidence in Frozen Vision–Language Models**

APERTURE fits a reusable, domain-level visual lens from a small labeled support set. The frozen VLM produces fresh activations for every query; APERTURE applies a bounded full-matrix residual only to selected visual-token K/V projection outputs. It does not append demonstrations, retain support activations, or optimize on each query.

This repository is the canonical consolidation of `yuhanlydia/StateAdaptation` and the earlier `Yunbo-max/EventTune` project. It preserves historical package names (`eventttt`, `vigor_handoff`) for replay compatibility while exposing a unified `aperture` command.

## Evidence status

- **Gate A:** PASS. The method is implemented and validated on real VLMs.
- **Main completed evidence:** BRIGHT disaster assessment; CAMELYON17-H2 and PathMNIST pathology; selected robotic visual-decision suites; calibration, image-dependence, placement, objective, and residual-dose analyses.
- **Required final additions:** five-hospital CAMELYON17 validation on Qwen2.5-VL-7B and InternVL3-8B; controlled rank-16 resource measurement.
- Negative results and protocol-specific limitations are retained under `reports/`, `results/`, and `docs/` rather than deleted.

## Quick start

```bash
git clone https://github.com/Yunbo-max/Apenture.git
cd Apenture
python3 -m pip install -e '.[test]'
python3 scripts/validate_release.py

# Existing matched core suite
aperture plan --suite core
aperture check --suite core

# Required five-hospital validation
aperture plan --suite medical-multihospital
aperture run --suite medical-multihospital --gpus 0,1,2,3

# Required rank-16 resource measurement
aperture plan --suite efficiency-rank16
aperture run --suite efficiency-rank16 --gpus 0
```

Read `AGENTS.md`, `docs/EXPERIMENT_MATRIX.md`, and `docs/RUNBOOK.md` before GPU execution. Raw datasets, patient records, model weights, credentials, and ignored multi-GB run directories are intentionally not stored here.
