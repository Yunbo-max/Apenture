#!/usr/bin/env python3
"""Regenerate the pathmnist fresh prepared manifests for the comprehensive chain.

Produces (mirroring the camelyon17 fresh layout):

    runs/aperture_medical_fresh_20261106/prepared/pathmnist_crc/query.jsonl   (900: 100/class, seed-independent)
    runs/aperture_medical_fresh_20261106/prepared/pathmnist_crc/n72/seed_{seed}/fit.jsonl      (54: 6/class)
    runs/aperture_medical_fresh_20261106/prepared/pathmnist_crc/n72/seed_{seed}/calibration.jsonl (18: 2/class)

Population rows come from the audited `aperture_medical.data.path_population`
(official PathMNIST-224 test cohort, content-hash groups, image-content isolation).
Partitioning is deterministic hash-order per class; roles are pairwise
group-disjoint by construction. No model outcomes are consulted.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aperture_medical.data import path_population, PATH_MD5
from vigor_handoff.protocol import atomic_json, file_hash

REPO = Path(__file__).resolve().parents[1]
SPLIT_SEED = 271107
QUERY_PER_CLASS = 100
FIT_PER_CLASS = 6
CAL_PER_CLASS = 2
CLASSES = 9


def order(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256(f'{salt}|{r["sample_id"]}'.encode()).hexdigest())


def take_balanced(pool, per_class, salt, classes):
    chosen = []
    counts = Counter()
    for r in order(pool, salt):
        if counts[r['label_id']] < per_class:
            chosen.append(r)
            counts[r['label_id']] += 1
        if all(counts[c] >= per_class for c in range(classes)):
            break
    if any(counts[c] != per_class for c in range(classes)):
        raise ValueError(f'balanced selection failed: {dict(counts)}')
    return chosen


def main():
    npz = REPO / 'data/medical_raw/pathmnist_224.npz'
    if not npz.is_file():
        raise FileNotFoundError(f'{npz}: download with scripts/download_aperture_medical.py pathmnist first')
    cache = REPO / 'data/raw/pathmnist_cache'
    rows, receipt = path_population(
        {'dataset': 'pathmnist', 'source_root': str(npz), 'name': 'pathmnist_crc'}, str(cache))
    if len(rows) != 7180:
        raise ValueError(f'expected 7180 official test rows, got {len(rows)}')
    groups = {r['group_id'] for r in rows}
    if len(groups) != len(rows):
        raise ValueError('content-hash groups must be unique per row')

    root = REPO / 'runs/aperture_medical_fresh_20261106/prepared/pathmnist_crc'
    query = take_balanced(rows, QUERY_PER_CLASS, f'pathmnist|query|{SPLIT_SEED}', CLASSES)
    q_ids = {r['sample_id'] for r in query}
    remaining = [r for r in rows if r['sample_id'] not in q_ids]
    _locked_write(root / 'query.jsonl', query)
    for seed in (0, 1, 2):
        cal = take_balanced(remaining, CAL_PER_CLASS, f'pathmnist|cal|{seed}', CLASSES)
        cal_ids = {r['sample_id'] for r in cal}
        fit_pool = [r for r in remaining if r['sample_id'] not in cal_ids]
        fit = take_balanced(fit_pool, FIT_PER_CLASS, f'pathmnist|fit|{seed}', CLASSES)
        out = root / 'n72' / f'seed_{seed}'
        _locked_write(out / 'fit.jsonl', fit)
        _locked_write(out / 'calibration.jsonl', cal)
        record = {'protocol': 'aperture_medical_fresh_pathmnist_v1', 'split_seed': SPLIT_SEED,
                  'classes': CLASSES, 'query_per_class': QUERY_PER_CLASS, 'fit_per_class': FIT_PER_CLASS,
                  'calibration_per_class': CAL_PER_CLASS, 'seed': seed,
                  'isolation': 'image_content_only; patient/slide IDs unavailable (official cohort)',
                  'source': receipt, 'query_ids_sha256': hashlib.sha256(
                      ''.join(sorted(q_ids)).encode()).hexdigest(),
                  'manifests': {n: file_hash(out / f'{n}.jsonl') for n in ('fit', 'calibration')}}
        atomic_json(out / 'split.json', record)
        print(f'Prepared pathmnist_crc seed {seed}: {len(fit)} fit / {len(cal)} calibration / {len(query)} query')
    atomic_json(root / 'cohort_lock.json', {'protocol': 'aperture_medical_fresh_pathmnist_v1',
                  'source_npz_md5': PATH_MD5, 'split_seed': SPLIT_SEED, 'seeds': [0, 1, 2],
                  'budget_total_labels': 72, 'query_count': QUERY_PER_CLASS * CLASSES,
                  'query_sampling': 'hash-ordered balanced per class; image-content isolation'})


def _locked_write(path, rows):
    path = Path(path)
    text = ''.join(json.dumps(deepcopy(r), sort_keys=True, allow_nan=False) + '\n' for r in rows)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'locked manifest changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


if __name__ == '__main__':
    raise SystemExit(main())
