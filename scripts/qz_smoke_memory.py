#!/usr/bin/env python3
"""GPU smoke probe: per-model peak memory vs batch (rows per forward).

Runs on the GPU platform inside a qz job. Uses the REAL loading path
(aperture_comprehensive.backend.loaded) and REAL support rows from the frozen
E01 plan (camelyon17-hospital_2--n16--s0 units). Sweeps B in [1,2,4,8] rows per
single model forward (no candidate scoring, activation memory only) plus one
APERTURE-style backward at B=1. Writes runs/qz_smoke/report.json.
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'runs/aperture_rebuttal_cases_20261107_v2/E01/plan.json'
OUT = REPO / 'runs/qz_smoke'
BATCHES = (1, 2, 4, 8, 16, 64, 128)


def load_plan():
    plan = json.loads(PLAN.read_text())
    by_key = {u['key']: u for u in plan['units']}
    return plan, by_key


def rows_of(unit, n):
    import json as _json
    path = Path(unit['support'])
    rows = [_json.loads(line) for line in path.read_text().splitlines()]
    return rows[:n]


def build_row_batch(backend, unit, rows, candidate=None):
    """One batched input dict for `rows` (activation-memory probe only)."""
    per = [backend.batch(r, candidate if candidate is not None else r['label']) for r in rows]
    batch = {}
    keys = set().union(*[set(p[0].keys()) for p in per])
    for k in sorted(keys):
        if all(k in p[0] and torch.is_tensor(p[0][k]) for p in per):
            tensors = [p[0][k] for p in per]
            if k == 'input_ids' or k == 'attention_mask':
                maxlen = max(t.shape[1] for t in tensors)
                device = tensors[0].device
                pad = torch.zeros(1, 1, dtype=torch.long, device=device)
                stacked = []
                for t in tensors:
                    n = maxlen - t.shape[1]
                    if n:
                        t = torch.cat([pad.expand(t.shape[0], n), t.to(device)], dim=1)
                    stacked.append(t.to(device))
                batch[k] = torch.cat(stacked, dim=0)
            else:
                batch[k] = torch.cat(tensors, dim=0)
    batch = {k: v.to(backend.device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if 'pixel_values' in batch:
        batch['pixel_values'] = batch['pixel_values'].to(dtype=next(backend.model.parameters()).dtype)
    return batch


def forward_internvl(model, batch):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    lm = base.language_model
    ids = batch['input_ids']
    embeddings = lm.get_input_embeddings()(ids).clone()
    vision = base.extract_feature(batch['pixel_values'])
    slots = ids == base.config.img_context_token_id
    assert int(slots.sum()) == vision.shape[0] * vision.shape[1], 'InternVL embedding/token mismatch'
    embeddings[slots] = vision.reshape(-1, vision.shape[-1])
    return lm(inputs_embeds=embeddings, attention_mask=batch.get('attention_mask'),
              use_cache=False, return_dict=True)


def probe_one(model_key, unit, plan):
    from aperture_comprehensive.backend import loaded, decoder_sites
    from vigor_handoff.core import ResidualController
    unit = dict(unit)
    unit['rank'] = 16; unit['alpha'] = 3.; unit['sites'] = ['K', 'V']
    result = {'model': model_key, 'family': unit['family'], 'image_size': unit['image_size'],
              'kind': unit['kind'], 'model_bytes': None}
    with loaded({**unit, 'arm': 'frozen'}) as (model, backend, _):
        import aperture_comprehensive.backend as B
        result['model_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
        rows = rows_of(unit, max(BATCHES))
        print(f'[{model_key}] loaded {result["model_bytes"]/2**30:.2f}GiB weights; rows={len(rows)}', flush=True)
        # forward sweep (OOM-tolerant: stop enlarging this model's batch on OOM)
        for b in BATCHES:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            sel = [(rows[i % len(rows)]) for i in range(b)]
            batch = build_row_batch(backend, unit, sel)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    if unit['family'] == 'internvl_chat':
                        forward_internvl(model, batch)
                    else:
                        kwargs = {'use_cache': False, 'return_dict': True}
                        model(**batch, **kwargs)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if 'out of memory' not in str(exc).lower():
                    raise
                result[f'fwd_b{b}'] = {'oom': True}
                print(f'[{model_key}] fwd B={b}: OOM at {torch.cuda.memory_allocated()/2**30:.2f}GiB alloc '
                      f'{torch.cuda.memory_reserved()/2**30:.2f}GiB reserved', flush=True)
                torch.cuda.empty_cache()
                break
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            result[f'fwd_b{b}'] = {'alloc_gib': torch.cuda.max_memory_allocated() / 2**30,
                                   'reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                                   'seconds': round(dt, 2)}
            print(f'[{model_key}] fwd B={b}: alloc={result[f"fwd_b{b}"]["alloc_gib"]:.2f}GiB '
                  f'reserved={result[f"fwd_b{b}"]["reserved_gib"]:.2f}GiB {dt:.2f}s', flush=True)
        # one APERTURE-style backward at B=1
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        sites = decoder_sites(model, unit['layers'])
        bases = {}
        for l, k, m in sites:
            g = torch.Generator(device='cpu').manual_seed(l + (0 if k == 'K' else 100))
            bases[(l, k)] = torch.linalg.qr(torch.randn(m.out_features, 16, generator=g), mode='reduced').Q
        control = ResidualController(sites, bases, 3., 'full')
        try:
            row = rows[0]
            lp, mask, _ = backend.candidate(row, row['label'], control)
            (-lp.mean()).backward()
            torch.cuda.synchronize()
            result['fit_b1'] = {'alloc_gib': torch.cuda.max_memory_allocated() / 2**30,
                                'reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                                'grad_ok': bool(any(p.grad is not None and torch.isfinite(p.grad).all()
                                                   and p.grad.abs().sum() > 0 for p in control.parameters()))}
            print(f'[{model_key}] fit B=1: alloc={result["fit_b1"]["alloc_gib"]:.2f}GiB '
                  f'reserved={result["fit_b1"]["reserved_gib"]:.2f}GiB grad_ok={result["fit_b1"]["grad_ok"]}', flush=True)
        finally:
            control.close()
    return result


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    print(f'GPU: {torch.cuda.get_device_name(0)} total={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB', flush=True)
    plan, by_key = load_plan()
    results = []
    for model_key in ('q25', 'q34', 'g34', 'internvl3', 'phi'):
        key = f'{model_key}--camelyon17-hospital_2--n16--s0'
        if key not in by_key:
            print(f'[{model_key}] unit {key} not in plan, skipping', flush=True)
            continue
        try:
            results.append(probe_one(model_key, by_key[key], plan))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({'model': model_key, 'error': f'{type(exc).__name__}: {exc}'})
        gc.collect(); torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'gpu': torch.cuda.get_device_name(0), 'batches': list(BATCHES), 'results': results}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REPORT:', json.dumps(report, indent=2)[:2000], flush=True)
    print('SMOKE_DONE', flush=True)


if __name__ == '__main__':
    main()
