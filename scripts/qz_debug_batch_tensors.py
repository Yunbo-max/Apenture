#!/usr/bin/env python3
"""CPU-only debug: compare batched input assembly vs per-row tensors."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import transformers
from aperture_comprehensive.backend import Backend

REPO = Path(__file__).resolve().parents[1]
plan = json.loads(Path(REPO / 'runs/aperture_rebuttal_cases_20261108_b16/E01/plan.json').read_text())
unit = next(u for u in plan['units'] if u['key'] == 'q25--camelyon17-hospital_2--n16--s0')
rows = [json.loads(l) for l in Path(unit['support']).read_text().splitlines()][:4]

processor = transformers.AutoProcessor.from_pretrained(unit['model_path'], trust_remote_code=True, local_files_only=True)
config = transformers.AutoConfig.from_pretrained(unit['model_path'], trust_remote_code=True, local_files_only=True)
b = object.__new__(Backend)
b.job = unit; b.processor = processor
b.model = torch.nn.Linear(1, 1).to(torch.bfloat16)
b.base = None
b.config = config; b.device = torch.device('cpu')

per = [b.batch(r, r['label']) for r in rows]
batch, meta = b.batch_rows(rows, None)
print('batched keys:', sorted(batch.keys()))
print('row lens:', [p[0]['input_ids'].shape[1] for p in per])
print('row span lens:', [p[1][1]-p[1][0] for p in per])
for k in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'):
    if k not in batch:
        print(k, 'NOT in batch'); continue
    print(k, 'batch shape:', tuple(batch[k].shape), '| per-row shape:', tuple(per[0][0][k].shape) if k in per[0][0] else 'n/a')
    if k not in per[0][0]:
        continue
    for i, p in enumerate(per):
        if k in ('input_ids', 'attention_mask'):
            n = p[0][k].shape[1]
            row_t = p[0][k][0]
            bat_t = batch[k][i][:n]
            same = torch.equal(bat_t.cpu(), row_t.cpu())
            print(f'  row{i}: {"EQUAL" if same else "DIFF"} (prefix slice vs original)')
            if not same:
                d = (bat_t.cpu().float() - row_t.cpu().float()).abs().max().item()
                print(f'    max diff={d:.3g}')
        else:
            a = batch[k][i:i+1]
            b0 = p[0][k]
            if a.dim() == 3 and b0.dim() == 2:
                b0 = b0.unsqueeze(0)
            same = torch.equal(a.cpu(), b0.cpu())
            print(f'  row{i}: {"EQUAL" if same else "DIFF"}')
            if not same:
                d = (a.cpu().float() - b0.cpu().float()).abs().max().item()
                print(f'    max diff={d:.3g}')
