#!/usr/bin/env python3
"""Extract recorded GPU peak memory from rebuttal run records (run after GPU stages).

Scans run_root/<case>/units/<unit>/... for fit/audit records and prints a
per-model, per-stage peak table (GiB). Records are the protocol-sealed
peak_allocated_bytes / peak_reserved_bytes written by the engine workers.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

FIELDS = ('peak_allocated_bytes', 'peak_reserved_bytes',
          'extraction_peak_allocated', 'extraction_peak_reserved')


def scan(root: Path):
    rows = []
    for record in sorted(root.rglob('*.json')):
        if record.name in ('plan.json', 'preflight.json', 'task.json'):
            continue
        try:
            data = json.loads(record.read_text())
        except Exception:
            continue
        hits = {k: v for k, v in data.items() if k in FIELDS}
        if not hits:
            continue
        stage = 'audit' if 'audit' in str(record.parent) else (
            'fit' if 'fits' in record.parts else 'evaluate' if 'evaluate' in str(record) else 'other')
        rel = record.relative_to(root)
        unit = rel.parts[2] if len(rel.parts) > 2 and 'units' in rel.parts else rel.parts[0]
        rows.append({'unit': unit, 'stage': stage, 'record': str(rel), **hits})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='runs/aperture_rebuttal_cases_20261107_v2')
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        raise SystemExit(f'no such run root: {root}')
    rows = scan(root)
    if not rows:
        raise SystemExit('no peak records found yet (GPU stages not run)')
    print(f'{"model":10s} {"stage":9s} {"alloc GiB":>9s} {"reserved GiB":>11s}  record')
    for r in sorted(rows, key=lambda x: (x['unit'], x['stage'])):
        model = r['unit'].split('--')[0]
        alloc = r.get('peak_allocated_bytes') or 0
        resv = r.get('peak_reserved_bytes') or 0
        print(f'{model:10s} {r["stage"]:9s} {alloc/2**30:9.2f} {resv/2**30:11.2f}  {r["record"]}')
    # per-model summary of the max
    best = {}
    for r in rows:
        m = r['unit'].split('--')[0]
        a = (r.get('peak_allocated_bytes') or 0) / 2**30
        v = (r.get('peak_reserved_bytes') or 0) / 2**30
        best[m] = (max(best.get(m, (0, 0))[0], a), max(best.get(m, (0, 0))[1], v))
    print('\nper-model max (GiB alloc/reserved):')
    for m, (a, v) in sorted(best.items()):
        print(f'  {m:10s} {a:6.2f} / {v:6.2f}')


if __name__ == '__main__':
    main()
