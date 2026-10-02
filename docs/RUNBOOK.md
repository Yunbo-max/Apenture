# Runbook

All suites follow `prepare -> plan -> check -> audit -> fit -> seal support-only choices -> evaluate -> summarize`.

```bash
python3 -m pip install -e '.[test]'
python3 scripts/validate_release.py

aperture prepare --suite core
aperture plan --suite core
aperture check --suite core
aperture run --suite core --gpus 0,1,2,3
aperture summarize --suite core

aperture prepare --suite medical-multihospital
aperture check --suite medical-multihospital
aperture run --suite medical-multihospital --gpus 0,1,2,3
aperture summarize --suite medical-multihospital

aperture prepare --suite efficiency-rank16
aperture check --suite efficiency-rank16
aperture run --suite efficiency-rank16 --gpus 0
aperture summarize --suite efficiency-rank16
```

The fixed BF16 protocols require a CUDA GPU with at least 24 GB. Missing assets or failed audits stop execution; do not substitute a smaller or quantized model under the same protocol name.
