#!/usr/bin/env python3
"""GPU equivalence probe: row-wise (B=1) vs batched (APERTURE_BATCH>1) paths.

A. Per-row candidate logprob sums: row-wise backend.candidate vs
   backend.candidate_batch over the full support set (no controller).
B. One APERTURE fit step: vigor_handoff.fit_coefficients (row-wise) vs
   engine.fit_controller_batched (B=5 and B=16) on identical controllers.
Reports max deltas. Runs inside a qz job; writes runs/qz_smoke/equiv.json.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / 'runs/aperture_rebuttal_cases_20261108_b16/E01/plan.json'
OUT = REPO / 'runs/qz_smoke'


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    torch.manual_seed(0)
    plan = json.loads(PLAN.read_text())
    unit = next(u for u in plan['units'] if u['key'] == 'q25--camelyon17-hospital_2--n16--s0')
    rows = [json.loads(line) for line in Path(unit['support']).read_text().splitlines()]
    print(f'GPU {torch.cuda.get_device_name(0)}; support rows={len(rows)}', flush=True)
    report = {}

    from aperture_comprehensive.backend import loaded, decoder_sites
    from aperture_rebuttal_cases.engine import fit_controller_batched
    from vigor_handoff.core import ResidualController, fit_coefficients

    with loaded({**unit, 'arm': 'frozen'}) as (model, backend, _):
        # A. logprob equivalence (no controller)
        with torch.no_grad():
            row_lps = []
            for r in rows:
                lp, _, _ = backend.candidate(r, r['label'])
                row_lps.append(float(lp.sum()))
            # B=1 batch path must reproduce the row-wise path exactly
            single_lp, _, _, _ = backend.candidate_batch([rows[0]], None)
            batched_lps, _, _, meta = backend.candidate_batch(rows, None)
        deltas = [abs(a - b) for a, b in zip(row_lps, [float(x) for x in batched_lps])]
        single_delta = abs(row_lps[0] - float(single_lp[0]))
        report['logprob_max_delta'] = max(deltas)
        report['logprob_deltas'] = deltas
        report['b1_batch_delta'] = single_delta
        print(f'A. B=1 batch delta: {single_delta:.3e}', flush=True)
        print(f'A. B=12 batch deltas: ' + ' '.join(f'{d:.3e}' for d in deltas), flush=True)
        print(f'A. row lps: ' + ' '.join(f'{v:.5f}' for v in row_lps), flush=True)
        print(f'A. batched lps: ' + ' '.join(f'{float(x):.5f}' for x in batched_lps), flush=True)

        # B. one fit step: row-wise vs batched B=5 and B=16
        sites = decoder_sites(model, unit['layers'])
        basis_spec = []
        for l, k, m in sites:
            g = torch.Generator(device='cpu').manual_seed(l + (0 if k == 'K' else 100))
            basis_spec.append((l, k, torch.linalg.qr(
                torch.randn(m.out_features, unit['rank'], generator=g), mode='reduced').Q))
        job = {**unit, 'engine': 'aperture', 'steps': 1, 'lr': .05, 'l2': .001,
               'loss_reduction': unit['loss_reduction']}

        def make_ctrl():
            return ResidualController(sites, {(l, k): b for l, k, b in basis_spec}, 3., 'full')

        # diagnostic: replicate the proven smoke-probe direct backward first
        diag = make_ctrl()
        try:
            dlp, dmask, _ = backend.candidate(rows[0], rows[0]['label'], diag)
            print(f'  diag: loss.requires_grad={dlp.requires_grad}, mask_any={bool(dmask.any())}', flush=True)
            (-dlp.mean()).backward()
            grads = [p.grad for p in diag.parameters() if p.requires_grad]
            print(f'  diag: params={len(list(diag.parameters()))}, grad_ok={bool(grads and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads))}', flush=True)
        finally:
            diag.close()

        def params_of(ctrl):
            return {k: v.detach().float().cpu().clone() for k, v in ctrl.raw.items()}

        ref = make_ctrl()
        h_ref = fit_coefficients(model, ref, rows,
                                 lambda r: backend.loss_and_mask(r, ref)[0],
                                 steps=1, lr=.05, l2=.001, reduction=unit['loss_reduction'])
        ref_params = params_of(ref)
        for label, B in (('b5', 5), ('b16', 16)):
            os.environ['APERTURE_BATCH'] = str(B)
            ctrl = make_ctrl()
            h_bat = fit_controller_batched(model, backend, ctrl, rows, job)
            got = params_of(ctrl)
            delta = max(float((got[k] - ref_params[k]).abs().max()) for k in got)
            obj_delta = abs(h_ref[0]['objective'] - h_bat[0]['objective'])
            report[f'fit_{label}'] = {'param_max_delta': delta, 'objective_delta': obj_delta,
                                      'objective_rowwise': h_ref[0]['objective'],
                                      'objective_batched': h_bat[0]['objective']}
            print(f'B. fit {label}: param max delta {delta:.3e}, objective delta {obj_delta:.3e}', flush=True)
        os.environ.pop('APERTURE_BATCH', None)

    (OUT / 'equiv.json').write_text(json.dumps(report, indent=2) + '\n')
    print('EQUIV_DONE', flush=True)


if __name__ == '__main__':
    main()
