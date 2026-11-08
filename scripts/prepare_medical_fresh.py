#!/usr/bin/env python3
"""Regenerate the medical-fresh prepared manifests consumed by the comprehensive
rebuttal data chain.

The comprehensive config (configs/aperture_comprehensive_v1.json) references
`runs/aperture_medical_fresh_20261106/prepared/...` source manifests. The medical
CLI's plan stage enforces `aperture_medical_multimodel_*` root names, so this
script calls the partition prepare directly (same audited partitioner, no plan
jobs) and writes the fresh layout expected downstream:

    prepared/hospital_{h}/query.jsonl                        (hoisted, seed-independent)
    prepared/hospital_{h}/n16/seed_{seed}/fit.jsonl          (12 rows: 6/class)
    prepared/hospital_{h}/n16/seed_{seed}/calibration.jsonl  (4 rows: 2/class)

Parameters (documented in cohort_lock.json and layout_hoist.json):
    hospitals [2], support_seeds [0,1,2], budget 16, query 500, split_seed 271107.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aperture_medical_multimodel.data import prepare
from vigor_handoff.protocol import atomic_json, file_hash

REPO = Path(__file__).resolve().parents[1]


def main():
    args = [a for a in sys.argv[1:]]
    cfg_path = Path(args.pop(0)) if args and not args[0].startswith('--') else REPO / 'configs/aperture_medical_fresh.json'
    target_root = None
    if '--target-root' in args:
        target_root = Path(args[args.index('--target-root') + 1])
    cfg = json.loads(cfg_path.read_text())
    cfg['data_root'] = str((REPO / cfg['data_root']).resolve())
    cfg['prepared_root'] = str((REPO / cfg['prepared_root']).resolve())
    cfg['run_root'] = str((REPO / cfg['run_root']).resolve())
    prepare(cfg)

    # Hoist into the fresh layout the comprehensive config expects.
    root = (target_root or Path(cfg['prepared_root']))
    src_root = Path(cfg['prepared_root'])
    if root != src_root:
        # extension mode: prepare ran in its own root (own cohort_lock);
        # copy hospital layouts into the shared fresh root without touching locks
        for h in cfg['hospitals']:
            src = src_root / f'hospital_{h}'
            dst = root / f'hospital_{h}'
            if not src.is_dir():
                raise ValueError(f'missing prepared hospital dir: {src}')
            dst.mkdir(parents=True, exist_ok=True)
            import shutil
            for item in src.iterdir():
                target = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                elif item.is_file():
                    if target.exists():
                        if target.read_bytes() != item.read_bytes():
                            raise ValueError(f'locked file changed: {target}')
                    else:
                        shutil.copy2(item, target)
    budgets = sorted(set([cfg['primary_support']] + cfg.get('curve_supports', [])))
    hoist = {}
    for h in cfg['hospitals']:
        base = root / f'hospital_{h}'
        queries = [base / f'support_{b}' / f'seed_{s}' / 'query.jsonl'
                   for b in budgets for s in cfg['support_seeds']]
        missing = [str(p) for p in queries if not p.is_file()]
        if missing:
            raise ValueError(f'hospital {h}: missing prepared query manifests: {missing}')
        contents = {p: p.read_text() for p in queries}
        if len(set(contents.values())) != 1:
            raise ValueError(f'hospital {h}: query manifest differs across seeds/budgets; expected seed-independent query')
        canonical_query = base / f'support_{cfg["primary_support"]}' / f'seed_{cfg["support_seeds"][0]}' / 'query.jsonl'
        q_target = base / 'query.jsonl'
        _locked_copy(canonical_query, q_target)
        for b in budgets:
            for s in cfg['support_seeds']:
                for name in ('fit', 'calibration'):
                    src = base / f'support_{b}' / f'seed_{s}' / f'{name}.jsonl'
                    dst = base / f'n{b}' / f'seed_{s}' / f'{name}.jsonl'
                    _locked_copy(src, dst)
                    hoist[f'hospital_{h}/n{b}/seed_{s}/{name}.jsonl'] = {
                        'source': str(src.relative_to(root)), 'sha256': file_hash(dst)}
        hoist[f'hospital_{h}/query.jsonl'] = {
            'source': str(canonical_query.relative_to(root)), 'sha256': file_hash(q_target)}
    lock_path = root / 'layout_hoist.json'
    existing = {}
    if lock_path.exists():
        existing = json.loads(lock_path.read_text()).get('mapping', {})
    merged = {**existing, **hoist}
    if existing and any(existing[k] != merged[k] for k in existing if k in merged):
        raise ValueError(f'layout hoist mapping conflict for {set(existing) & set(hoist)}')
    atomic_json(lock_path, {
        'protocol': 'aperture_medical_fresh_layout_v1',
        'description': 'query hoisted to hospital level (budget/seed independent); '
                       'fit/calibration exposed as n{budget}/seed_{seed}/ files',
        'mapping': merged})
    print(json.dumps({'status': 'prepared_and_hoisted', 'prepared_root': str(root),
                      'hospitals': cfg['hospitals'], 'budgets': budgets}, indent=2))


def _locked_copy(src, dst):
    src, dst = Path(src), Path(dst)
    text = src.read_text()
    if dst.exists():
        if dst.read_text() != text:
            raise ValueError(f'locked manifest changed: {dst}')
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(dst)


if __name__ == '__main__':
    raise SystemExit(main())
