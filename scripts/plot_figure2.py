#!/usr/bin/env python3
"""Figure 2: Performance-Storage Trade-off Across Medical Domains and VLM Backbones.

Four square panels, log-x storage (MB) vs Macro-F1 (%). Frozen grey dashed
line; LoRA r4/r16/r64 one hue family with distinct markers; Static Full
diamond; Conditional Gated star. Validated categorical palette.
"""
import json, glob, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/'
            'backup/Apenture-medsweep/runs/aperture_medical_lora_rank_sweep_v1')
SRC_ROOTS = ['/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/'
             'Apenture-cg-checked',
             '/inspire/hdd/global_user/zhangjiaquan-253108540222/backup/'
             'Apenture-phi-fix']

PANELS = [
    ('g34', 'camelyon17-hospital_3', 'Gemma-3-4B / RUMC'),
    ('internvl3', 'camelyon17-hospital_4', 'InternVL3-8B / LPON'),
    ('phi', 'camelyon17-hospital_3', 'Phi-3.5 / RUMC'),
    ('q25', 'camelyon17-hospital_3', 'Qwen2.5-VL-7B / RUMC'),
]

LORA = {'r4': ('#2a78d6', 'o'), 'r16': ('#2a78d6', 's'),
        'r64': ('#2a78d6', '^')}
STATIC = ('#eb6834', 'D')
GATED = ('#1baf7a', '*')
FROZEN = '#8a8985'
GRID = '#d8d7d3'


def load_query():
    rows = {}
    for f in sorted(glob.glob(str(ROOT / 'query' / '*.json'))):
        d = json.loads(Path(f).read_text())
        rows[(d['pair'][0], d['pair'][1], d['seed'], d['rank'])] = d
    return rows


def load_baselines():
    out = {}
    import os
    variants = ('frozen', 'static-full', 'conditional-gated')
    for pair, dom, _ in PANELS:
        for seed in (0, 1, 2):
            key = f"{pair}--{dom}--n16--s{seed}"
            for w in SRC_ROOTS:
                import glob as g
                roots = [w + '/runs/aperture_conditional_gated_20261111_recheck1']
                roots += sorted(g.glob(w + '/runs/aperture_conditional_gated_20261112_c*'))
                roots += sorted(g.glob(w + '/runs/aperture_conditional_gated_20261112_phi_fix*'))
                for r in roots:
                    folder = Path(r) / 'CG01' / 'units' / key
                    if not (folder / 'selection_receipt' / 'DONE.json').is_file():
                        continue
                    for vn in variants:
                        ev = folder / 'evaluation' / vn
                        if not (ev / 'DONE.json').is_file():
                            continue
                        mf = json.loads((ev / 'metrics.json').read_text())
                        raw = mf.get('raw', {})
                        fitd = folder / 'fits' / vn
                        info = json.loads((fitd / 'fit.json').read_text()) \
                            if (fitd / 'fit.json').is_file() else {}
                        out[(pair, seed, vn)] = {
                            'macro_f1': raw.get('macro_f1'),
                            'bytes': info.get('serialized_state_bytes'),
                        }
                    break
                else:
                    continue
                break
    return out


def adapter_bytes(pair, seed, rank, qrow):
    m = json.loads((ROOT / 'manifest.json').read_text())
    cell = next(c for c in m['cells']
                if c['pair'] == list(pair) and c['seed'] == seed
                and c['cell'] == qrow['cell'])
    if cell['reused']:
        info = json.loads((Path(cell['source']) / 'fit.json').read_text())
        return info.get('serialized_state_bytes')
    info = json.loads((Path(cell['cell_dir']) / 'fit.json').read_text())
    return info.get('adapter_bytes')


def main():
    q = load_query()
    b = load_baselines()
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.1), sharey=True)
    csv_lines = ['panel,model,domain,method,macro_f1_mean,macro_f1_std,'
                 'state_bytes,state_mb,rank,passes,lr,reused,source,notes']
    for ax, (pair, dom, title) in zip(axes, PANELS):
        # frozen dashed line
        fs = [b[(pair, s, 'frozen')]['macro_f1'] for s in (0, 1, 2)
              if (pair, s, 'frozen') in b
              and b[(pair, s, 'frozen')]['macro_f1'] is not None]
        if fs:
            ax.axhline(st.mean(fs), color=FROZEN, ls='--', lw=1.4,
                       zorder=1, label='Frozen')
        for rank in (4, 16, 64):
            rs = [q[(pair, dom, s, rank)] for s in (0, 1, 2)
                  if (pair, dom, s, rank) in q]
            if not rs:
                continue
            f1 = [r['macro_f1'] for r in rs]
            byt = st.mean([adapter_bytes((pair, dom), r['seed'], rank, r)
                           for r in rs])
            color, marker = LORA[f'r{rank}']
            ax.scatter(byt / 1e6, st.mean(f1) * 100, s=64, color=color,
                       marker=marker, edgecolors='white', linewidths=1.2,
                       zorder=3, label=f'LoRA r{rank}')
            csv_lines.append(
                f'{title},{pair},{dom},LoRA-r{rank},{st.mean(f1):.4f},'
                f'{st.pstdev(f1):.4f},{byt},{byt/1e6:.2f},{rank},'
                f'{rs[0]["passes"]},{rs[0]["lr"]:g},{rs[0]["reused"]},'
                f',calibration-selected per (pair,seed,rank)')
        for vn, (color, marker), lbl in (('static-full', STATIC, 'Static Full'),
                                         ('conditional-gated', GATED, 'Cond. Gated')):
            rs = [b[(pair, s, vn)] for s in (0, 1, 2)
                  if (pair, s, vn) in b and b[(pair, s, vn)]['macro_f1'] is not None]
            if not rs:
                continue
            f1 = [r['macro_f1'] for r in rs]
            byt = st.mean([r['bytes'] for r in rs if r['bytes']])
            ax.scatter(byt / 1e6, st.mean(f1) * 100, s=96, color=color,
                       marker=marker, edgecolors='white', linewidths=1.2,
                       zorder=4, label=lbl)
            csv_lines.append(
                f'{title},{pair},{dom},{vn},{st.mean(f1):.4f},'
                f'{st.pstdev(f1):.4f},{byt},{byt/1e6:.2f},,,,False,'
                f',CG01 canonical (no search)')
        ax.set_xscale('log')
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel('Saved Adaptation State (MB)', fontsize=9)
        ax.grid(True, which='both', color=GRID, lw=0.6, alpha=0.6)
        ax.tick_params(labelsize=8)
        ax.set_aspect('auto')
    axes[0].set_ylabel('Macro-F1 (%)', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Performance–Storage Trade-off Across Medical Domains '
                 'and VLM Backbones', fontsize=13, y=1.08)
    fig.tight_layout()
    out = ROOT / 'report'
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'figure2_performance_storage.{ext}',
                    dpi=300, bbox_inches='tight')
    (out / 'figure2_data.csv').write_text('\n'.join(csv_lines) + '\n')
    print(f'figure saved to {out}')


if __name__ == '__main__':
    main()
