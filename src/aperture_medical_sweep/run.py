"""Medical LoRA rank-storage sweep (Figure 2). Independent of CG01-CG09 protocol.

Reuses frozen CG01 unit definitions (same manifests/models/prompts), the
existing LoRA training path (vigor_handoff.worker.train_lora) and the native
backend. Selection is calibration-only, per (pair, seed, rank); query metrics
are computed once per selected configuration.
"""
from __future__ import annotations

import json, os, sys, time, math, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from aperture_rebuttal_cases import store  # noqa: E402
from aperture_rebuttal_cases.report import metrics  # noqa: E402

SWEEP = 'aperture-medical-lora-sweep-v1'
PAIRS = [
    ('g34', 'camelyon17-hospital_3'),   # Panel A: Gemma-3-4B / RUMC
    ('internvl3', 'camelyon17-hospital_4'),  # Panel B: InternVL3-8B / LPON
    ('phi', 'camelyon17-hospital_3'),   # Panel C: Phi-3.5 / RUMC
    ('q25', 'camelyon17-hospital_3'),   # Panel D: Qwen2.5-VL-7B / RUMC
]
SEEDS = (0, 1, 2)
RANKS = (4, 16, 64)
PASSES = (1, 4, 16)
LRS = (5e-5, 2e-4, 8e-4)


def cell_key(rank, passes, lr):
    return f'r{rank}-p{passes}-lr{lr:g}'


def grid():
    out = []
    for r in RANKS:
        for p in PASSES:
            for lr in LRS:
                out.append({'key': cell_key(r, p, lr), 'rank': r,
                            'passes': p, 'lr': lr, 'alpha': 2 * r})
    return out


def find_unit_definition(pair, seed, source_roots):
    """Return the frozen CG01 unit dict for (model, domain, seed)."""
    model, domain = pair
    key = f'{model}--{domain}--n16--s{seed}'
    for root in source_roots:
        pp = Path(root) / 'CG01' / 'plan.json'
        if not pp.is_file():
            continue
        plan = json.loads(pp.read_text())
        for u in plan['units']:
            if u['key'] == key:
                return u, str(root)
    raise FileNotFoundError(f'unit definition not found: {key}')


def verify_cg01_seal(fit_dir):
    """Hash-verify an existing CG01 fit DONE seal. Returns the DONE state."""
    return store.verify(Path(fit_dir), None)


def plan(source_roots, out_root):
    out = Path(out_root)
    cells = []
    reused = 0
    for pair in PAIRS:
        for seed in SEEDS:
            unit, src_root = find_unit_definition(pair, seed, source_roots)
            fit_roots = [Path(r) / 'CG01' / 'units' / unit['key'] / 'fits'
                         for r in source_roots]
            for cell in grid():
                ck = cell['key']
                cell_dir = out / 'units' / f"{unit['key']}__{ck}"
                is_reused = False
                src = None
                # CG01 lora variants: rank16, alpha32, lr2e-4, passes 1/4
                if cell['rank'] == 16 and cell['lr'] == 2e-4 \
                        and cell['passes'] in (1, 4):
                    name = 'lora1' if cell['passes'] == 1 else 'lora4'
                    for candidate in [fr / name for fr in fit_roots]:
                        if candidate.is_dir() and \
                                (candidate / 'DONE.json').is_file():
                            try:
                                verify_cg01_seal(candidate)
                                is_reused = True
                                src = str(candidate)
                                reused += 1
                            except ValueError:
                                pass
                            break
                cells.append({
                    'pair': pair, 'seed': seed, 'unit': unit['key'],
                    'cell': ck, 'rank': cell['rank'], 'passes': cell['passes'],
                    'lr': cell['lr'], 'alpha': cell['alpha'],
                    'reused': is_reused, 'source': src,
                    'cell_dir': str(cell_dir),
                })
    manifest = {
        'sweep': SWEEP, 'pairs': PAIRS, 'seeds': SEEDS,
        'ranks': RANKS, 'passes': PASSES, 'lrs': LRS,
        'total_cells': len(cells), 'reused_cells': reused,
        'selection_rule': 'min raw calibration_nll per (pair,seed,rank); '
                          'deterministic tie-break by passes then lr',
        'cells': cells,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_cell(manifest, pair, seed, ck):
    for c in manifest['cells']:
        if c['pair'] == list(pair) and c['seed'] == seed and c['cell'] == ck:
            return c
    raise KeyError((pair, seed, ck))


def fit_cell(cell, unit, out_root):
    """Fit one LoRA cell; write fit.json + calibration + seal. No query access."""
    from aperture_conditional_gated.adapters import loaded
    from aperture_conditional_gated.engine import score
    from aperture_rebuttal_cases.data import load_roles
    from aperture_comprehensive.numeric import score_arrays
    from aperture_final.numeric import fit_temperature
    from vigor_handoff.worker import train_lora
    import torch

    folder = Path(cell['cell_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'DONE.json').is_file():
        store.verify(folder, binding_for(cell, unit))
        return
    variant = {
        'name': cell['cell'], 'family': f"lora-r{cell['rank']}",
        'engine': 'lora', 'lora_rank': cell['rank'],
        'lora_alpha': cell['alpha'], 'lora_dropout': 0.05,
        'lora_lr': cell['lr'], 'lora_passes': cell['passes'],
        'lora_accumulation': 1, 'method_version': SWEEP,
    }
    unit2 = {**unit, 'output': str(folder)}
    roles = load_roles(unit2, images=True)
    support = roles['support']
    job = {**unit2, **variant}
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with loaded(unit2, variant) as (model, backend, controller):
        history, updates, count = train_lora(model, backend, support, job)
        model.save_pretrained(folder / 'adapter', safe_serialization=True)
        torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - start
        cal = score(backend, roles['calibration'], controller, vqa=False)
        cal_nll = metrics(cal)['nll']
        ss, ys = score_arrays(cal, unit2['classes'])
        temp = fit_temperature(ss, ys, penalty=.01)['temperature']
        final_support = score(backend, roles['support'], controller, vqa=False)
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        adapter_bytes = sum(p.stat().st_size
                            for p in (folder / 'adapter').rglob('*')
                            if p.is_file())
        peak_allocated = torch.cuda.max_memory_allocated()
    store.write_rows(folder / 'calibration.jsonl', cal)
    store.write_rows(folder / 'support_final.jsonl', final_support)
    stats = {
        'name': cell['cell'], 'family': variant['family'], 'engine': 'lora',
        'lora_rank': cell['rank'], 'lora_alpha': cell['alpha'],
        'lora_dropout': 0.05, 'lora_lr': cell['lr'],
        'lora_passes': cell['passes'], 'lora_accumulation': 1,
        'calibration_nll': cal_nll, 'temperature': temp,
        'selection_criterion': 'raw_calibration_nll',
        'optimizer_updates': updates, 'support_visits': len(support) * cell['passes'],
        'trainable_scalars': trainable, 'adapter_bytes': adapter_bytes,
        'fit_seconds': fit_seconds, 'peak_allocated_bytes': peak_allocated,
        'final_calibration_metrics': metrics(cal),
        'final_support_metrics': metrics(final_support),
        'query_forwards': 0, 'history': history,
        'method_version': SWEEP,
    }
    if not math.isfinite(cal_nll):
        raise RuntimeError('Nonfinite calibration objective')
    (folder / 'config.json').write_text(
        json.dumps({'unit': {k: unit[k] for k in
                             ('key', 'model_key', 'model_id', 'model_path',
                              'domain', 'total_labels', 'support_seed',
                              'classes', 'layers', 'mask', 'image_size',
                              'kind', 'task_type')},
                    'variant': variant, 'sweep': SWEEP},
                   indent=2) + '\n')
    store.atomic(folder / 'fit.json', stats)
    store.seal(folder, binding_for(cell, unit))


def binding_for(cell, unit):
    manifests = {k: store.sha(Path(unit[k])) if Path(unit[k]).is_file()
                 else None for k in ('support', 'calibration', 'query')}
    return {'sweep': SWEEP, 'cell': cell['cell'], 'unit': unit['key'],
            'unit_manifests': manifests, 'reused': cell['reused'],
            'source': cell.get('source')}


def select(pair, seed, manifest, out_root):
    """Per-rank calibration selection with deterministic tie-break."""
    from aperture_rebuttal_cases import store as st
    picks = {}
    cells = [c for c in manifest['cells']
             if c['pair'] == list(pair) and c['seed'] == seed]
    for rank in RANKS:
        candidates = []
        for c in cells:
            if c['rank'] != rank:
                continue
            folder = Path(c['cell_dir'])
            if c['reused']:
                info = json.loads((Path(c['source']) / 'fit.json').read_text())
                cal_nll = info.get('calibration_nll')
            else:
                if (folder / 'DONE.json').is_file():
                    st.verify(folder)  # internal seal consistency
                info = json.loads((folder / 'fit.json').read_text())
                cal_nll = info['calibration_nll']
            if not isinstance(cal_nll, (int, float)) or not math.isfinite(cal_nll):
                continue
            candidates.append((cal_nll, c['passes'], c['lr'], c['cell'], c))
        if not candidates:
            raise ValueError(f'no valid candidates for {pair} s{seed} r{rank}')
        # deterministic tie-break: lower passes, then lower lr
        best = min(candidates, key=lambda t: (t[0], t[1], t[2]))
        picks[rank] = best[4]
    out = Path(out_root) / 'selection'
    out.mkdir(parents=True, exist_ok=True)
    rec = {'pair': pair, 'seed': seed,
           'selection': {str(r): {'cell': picks[r]['cell'],
                                  'rank': r, 'passes': picks[r]['passes'],
                                  'lr': picks[r]['lr'],
                                  'reused': picks[r]['reused']}
                         for r in RANKS},
           'criterion': 'raw_calibration_nll',
           'tie_break': 'passes asc, lr asc'}
    (out / f'{pair[0]}--{pair[1]}--s{seed}.json').write_text(
        json.dumps(rec, indent=2) + '\n')
    return rec


def unit_of(manifest, cell):
    key = cell['unit']
    for c in manifest['cells']:
        if c['unit'] == key:
            break
    # unit definition embedded at plan time is not stored per-cell; recover
    # from the frozen source plan of the same root layout.
    from aperture_medical_sweep.run import _load_unit_for_key
    return _load_unit_for_key(key)


def query_eval(pair, seed, manifest, out_root, selection_rec):
    """Evaluate the three rank winners on the fixed 500-query set."""
    from aperture_conditional_gated.adapters import loaded
    from aperture_conditional_gated.engine import score
    from aperture_rebuttal_cases.data import load_roles
    import torch
    out = Path(out_root) / 'query'
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    unit = _load_unit_for_key(
        f"{pair[0]}--{pair[1]}--n16--s{seed}")
    for rank, rec in selection_rec['selection'].items():
        cell = load_cell(manifest, pair, seed, rec['cell'])
        folder = Path(cell['cell_dir'])
        if cell['reused']:
            variant = {'name': ('lora1' if rec['passes'] == 1 else 'lora4'),
                       'engine': 'lora', 'lora_rank': 16, 'lora_alpha': 32,
                       'lora_dropout': 0.05, 'lora_lr': 2e-4,
                       'lora_passes': rec['passes'], 'lora_accumulation': 1}
            restore = Path(cell['source'])  # the CG01 fit dir itself
        else:
            variant = {'name': rec['cell'],
                       'family': f"lora-r{rec['rank']}", 'engine': 'lora',
                       'lora_rank': rec['rank'], 'lora_alpha': 2 * rec['rank'],
                       'lora_dropout': 0.05, 'lora_lr': rec['lr'],
                       'lora_passes': rec['passes'], 'lora_accumulation': 1,
                       'method_version': SWEEP}
            restore = folder
        unit2 = {**unit, 'output': str(folder)}
        roles = load_roles(unit2, images=True)
        with loaded(unit2, variant, restore=restore) as (model, backend, controller):
            with torch.no_grad():
                rows = score(backend, roles['query'], controller, vqa=False)
        m = metrics(rows)
        results[rank] = {'pair': pair, 'seed': seed, 'rank': int(rank),
                         'cell': rec['cell'], 'reused': rec['reused'],
                         'passes': rec['passes'], 'lr': rec['lr'],
                         'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
                         'nll': m['nll'], 'brier': m['brier'],
                         'n': m['n']}
        store.atomic(out / f"{pair[0]}--{pair[1]}--s{seed}--r{rank}.json",
                     results[rank])
    return results


# -- baseline rows (reused sealed CG01) -----------------------------------

def baseline_rows(source_roots):
    rows = []
    variants = ('frozen', 'static-full', 'conditional-gated', 'lora1', 'lora4')
    for pair in PAIRS:
        for seed in SEEDS:
            key = f"{pair[0]}--{pair[1]}--n16--s{seed}"
            for root in source_roots:
                folder = (Path(root) / 'CG01' / 'units' / key)
                if not (folder / 'selection_receipt' / 'DONE.json').is_file():
                    continue
                sel = json.loads(
                    (folder / 'selection.json').read_text())
                for vn in variants:
                    ev = folder / 'evaluation' / vn
                    if not (ev / 'DONE.json').is_file():
                        continue
                    mf = json.loads((ev / 'metrics.json').read_text())
                    raw = mf.get('raw', {})
                    fitd = folder / 'fits' / vn
                    info = json.loads((fitd / 'fit.json').read_text()) \
                        if (fitd / 'fit.json').is_file() else {}
                    rows.append({'pair': pair, 'seed': seed, 'variant': vn,
                                 'macro_f1': raw.get('macro_f1'),
                                 'accuracy': raw.get('accuracy'),
                                 'nll': raw.get('nll'),
                                 'bytes': info.get('serialized_state_bytes'),
                                 'trainable': info.get('optimized_scalars'),
                                 'root': str(root)})
                    break_ev = True
                break
    return rows


def figure_csv(out_root, selection_dir, query_rows, baseline_rows):
    import statistics as st
    lines = ['panel,model,domain,method,macro_f1_mean,macro_f1_std,'
             'state_bytes,state_mb,trainable_scalars,rank,passes,lr,'
             'reused,source,notes']
    for pair in PAIRS:
        model, domain = pair
        q = [r for r in query_rows.values()
             if r['pair'] == pair]
        for rank in sorted({r['rank'] for r in q}):
            rs = [r for r in q if r['rank'] == rank]
            f1 = [r['macro_f1'] for r in rs]
            r0 = rs[0]
            lines.append(
                f"{domain},{model},{domain},LoRA-r{rank},"
                f"{st.mean(f1):.4f},{st.pstdev(f1):.4f},"
                f"{r0.get('bytes','')},{r0.get('bytes',0)/1e6 if r0.get('bytes') else ''},"
                f"{r0.get('trainable','')},{rank},{r0['passes']},{r0['lr']:g},"
                f"{r0['reused']},,calibration-selected per (pair,seed,rank)")
        for vn in ('frozen', 'static-full', 'conditional-gated'):
            rs = [r for r in baseline_rows if r['pair'] == pair
                  and r['variant'] == vn and r['macro_f1'] is not None]
            if not rs:
                continue
            f1 = [r['macro_f1'] for r in rs]
            b = rs[0]['bytes']
            lines.append(
                f"{domain},{model},{domain},{vn},"
                f"{st.mean(f1):.4f},{st.pstdev(f1):.4f},"
                f"{b},{b/1e6 if b else ''},{rs[0]['trainable']},,,,"
                f"False,,CG01 canonical (no search)")
    Path(out_root, 'figure2_data.csv').write_text('\n'.join(lines) + '\n')
    return lines


_load_cache = {}


def _load_unit_for_key(key, source_roots=None):
    if key in _load_cache:
        return _load_cache[key]
    roots = source_roots or default_source_roots()
    u, _ = find_unit_definition(
        (key.split('--')[0], key.split('--')[1]), int(key.split('--')[-1][1:]),
        roots)
    _load_cache[key] = u
    return u


def default_source_roots():
    wts = [
        '/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/'
        'Apenture-cg-checked',
        '/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/'
        'Apenture-phi-fix',
    ]
    import glob
    roots = [w + '/runs/aperture_conditional_gated_20261111_recheck1'
             for w in wts]
    for w in wts:
        roots += sorted(glob.glob(w + '/runs/aperture_conditional_gated_20261112_c*'))
        roots += sorted(glob.glob(w + '/runs/aperture_conditional_gated_20261112_phi_fix*'))
    return roots
