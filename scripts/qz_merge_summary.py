#!/usr/bin/env python3
"""Reporting helper: merge sealed per-unit results across split run roots.

Each unit's evaluation/metrics.json and fits are protocol-sealed inside its
own root; this script (outside the protocol) aggregates them into one table.
Usage: qz_merge_summary.py --case E03 --roots runs/<rootA> runs/<rootB> [--out report]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_units(root, case):
    units_dir = Path(root) / case / 'units'
    if not units_dir.is_dir():
        return []
    out = []
    for u in sorted(units_dir.iterdir()):
        sel = u / 'selection.json'
        if not sel.is_file():
            continue
        selection = json.loads(sel.read_text())
        rows = []
        for variant in selection.get('evaluate', []):
            m = u / 'evaluation' / variant / 'metrics.json'
            if not m.is_file():
                continue
            metrics = json.loads(m.read_text())
            raw = metrics.get('temperature_scaled') or metrics.get('raw') or {}
            rows.append({'unit': u.name, 'variant': variant, **raw})
        out.extend(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = []
    for r in a.roots:
        rows.extend(load_units(r, a.case))
    print(f'case={a.case} units_evaluated_variants={len(rows)}')
    by = defaultdict(list)
    for r in rows:
        key = (r['unit'].split('--')[0], r['unit'].split('--')[1], r['variant'])
        by[key].append(r)
    table = []
    for key, rs in sorted(by.items()):
        model, domain, variant = key
        agg = {}
        for field in ('nll', 'accuracy', 'macro_f1'):
            vals = [r.get(field) for r in rs if r.get(field) is not None]
            if vals:
                agg[field] = sum(vals) / len(vals)
        table.append({'model': model, 'domain': domain, 'variant': variant,
                      'seeds': len(rs), **agg})
    out_path = a.out or f'runs/{a.case}_merged_summary.json'
    Path(out_path).write_text(json.dumps(table, indent=1) + '\n')
    print('wrote', out_path)
    print(json.dumps(table[:12], indent=1))


if __name__ == '__main__':
    main()
