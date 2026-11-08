# APERTURE E10: Capability retention after adaptation

[Protocol, fairness constraints, sources, and exact commands](docs/APERTURE_E10_RETENTION.md)

```bash
python scripts/run_aperture_retention.py list
```

Independent evaluation-only parts: visual-active retention, text-active/disabled controls, and explicit-domain archived-module retrieval/storage. Reuses sealed E01/E02 fits; no method change, no new training, no asserted no-forgetting result. Existing src and run entrypoints are untouched.
