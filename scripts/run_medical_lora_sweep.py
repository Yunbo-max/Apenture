#!/usr/bin/env python3
"""CLI for the medical LoRA rank-storage sweep (Figure 2)."""
from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from aperture_medical_sweep import run as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['plan', 'fit', 'select', 'query',
                                       'baselines', 'csv', 'list'])
    ap.add_argument('--root', default='runs/aperture_medical_lora_rank_sweep_v1')
    ap.add_argument('--pair', help='model--domain')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--cell', help='r4-p1-lr0.0002 style key')
    ap.add_argument('--source-roots', nargs='*',
                    default=S.default_source_roots())
    a = ap.parse_args()
    root = Path(a.root)
    if a.action == 'plan':
        m = S.plan(a.source_roots, root)
        print(json_out({'manifest': str(root / 'manifest.json'),
                        'cells': m['total_cells'],
                        'reused': m['reused_cells']}))
    elif a.action == 'fit':
        m = json.loads((root / 'manifest.json').read_text())
        pair = tuple(a.pair.split('--'))
        cell = S.load_cell(m, pair, a.seed, a.cell)
        unit = S._load_unit_for_key(cell['unit'], a.source_roots)
        S.fit_cell(cell, unit, root)
        print(json_out({'status': 'sealed', 'cell': a.cell,
                        'unit': cell['unit']}))
    elif a.action == 'select':
        m = json.loads((root / 'manifest.json').read_text())
        pair = tuple(a.pair.split('--'))
        rec = S.select(pair, a.seed, m, root)
        print(json_out(rec))
    elif a.action == 'query':
        m = json.loads((root / 'manifest.json').read_text())
        pair = tuple(a.pair.split('--'))
        rec = json.loads((root / 'selection' /
                          f'{pair[0]}--{pair[1]}--s{a.seed}.json').read_text())
        res = S.query_eval(pair, a.seed, m, root, rec)
        print(json_out(res))
    elif a.action == 'baselines':
        rows = S.baseline_rows(a.source_roots)
        print(json_out({'rows': len(rows), 'sample': rows[:2]}))
    elif a.action == 'csv':
        import glob, json as j
        qrows = {}
        for f in glob.glob(str(root / 'query' / '*.json')):
            qrows[f] = j.loads(Path(f).read_text())
        bros = S.baseline_rows(a.source_roots)
        S.figure_csv(root, None, qrows, bros)
        print(json_out({'csv': str(root / 'figure2_data.csv')}))
    elif a.action == 'list':
        m = json.loads((root / 'manifest.json').read_text())
        for c in m['cells']:
            print(c['unit'], c['cell'],
                  'REUSED' if c['reused'] else 'NEW')


def json_out(obj):
    import json
    return json.dumps(obj, indent=2, default=str)


if __name__ == '__main__':
    main()
