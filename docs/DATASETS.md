# Dataset registry

- **BRIGHT:** paired pre/post-event building classification on Hawaii, Libya, Noto, and Turkey. Main matched block uses support24, query300/event, three support seeds, tile-disjoint support/query.
- **CAMELYON17-WILDS:** normal/tumor pathology patches. Existing H2 blocks and planned five-hospital patient-disjoint validation are distinct protocols.
- **PathMNIST-224:** nine-class colorectal pathology, using the official 224-pixel asset and recorded checksum.
- **ManipBench Q1:** Bridge, DROID-place, and DROID-arti four-option visual decisions.
- **RoboFail / UR5-Fail / RoboVQA / Guardian:** recorded execution-verification diagnostics and boundaries; do not pool their splits.

Raw datasets are not redistributed by default. Provenance, checksums, selected portable subsets where permitted, and exact manifests remain under `data/`, `reports/`, and suite-specific documentation.
