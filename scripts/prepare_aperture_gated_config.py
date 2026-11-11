#!/usr/bin/env python3
"""Create NEW local CG config with absolute paths into an existing asset checkout.

Reads configuration only; does not copy images, models, old fits or results,
change the old checkout, install dependencies, or start training.
"""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-repo',required=True)
    ap.add_argument('--new-repo',default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--source-base',default='configs/aperture_comprehensive_v1.json')
    ap.add_argument('--out',default='configs/aperture_conditional_gated.local.json')
    ap.add_argument('--run-root',default='runs/aperture_conditional_gated_20261111_v1')
    ap.add_argument('--prepared-root',help='Existing E03/E07 prepared root, absolute or source-repo relative')
    a=ap.parse_args()
    try:
        old=Path(a.source_repo).resolve();new=Path(a.new_repo).resolve()
        out=(new/a.out).resolve();baseout=out.with_suffix('.base.json')
        if not out.is_relative_to(new/'configs'):raise ValueError('Output must be a new config under new-repo/configs')
        if out.exists() or baseout.exists():raise ValueError('Refusing to overwrite existing local config')
        base=json.loads((old/a.source_base).read_text())
        for m in base['models']:m['path']=str((old/m['path']).resolve())
        for field in ('prepared_root','run_root'):base[field]=str((old/base[field]).resolve())
        for d in base['datasets']:
            for field in ('source_support','source_fit','source_calibration','source_query','root','metadata','composites'):
                if field in d:d[field]=str((old/d[field]).resolve())
        cfg=json.loads((new/'configs/aperture_conditional_gated_v1.json').read_text())
        prior=old/'configs/aperture_reviewer_cases_v1.json'
        prior=json.loads(prior.read_text()) if prior.exists() else {}
        for field in ('prepared_root','medical_data_root','textvqa_train_json','textvqa_val_json','textvqa_images'):
            value=prior.get(field,cfg[field])
            if field=='prepared_root' and a.prepared_root:value=a.prepared_root
            cfg[field]=str((old/value).resolve())
        cfg.update(base_config=str(baseout),run_root=str((new/a.run_root).resolve()),execution_batch=1)
        out.parent.mkdir(parents=True,exist_ok=True)
        for path,value in ((baseout,base),(out,cfg)):
            with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
        print(json.dumps({'config':str(out),'base_config':str(baseout),'status':'created_not_run',
                          'note':'check paths and run native preflight; old states are not reused'}))
        return 0
    except (OSError,ValueError,KeyError) as exc:
        print('Configuration blocked: '+str(exc),file=__import__('sys').stderr);return 2

if __name__=='__main__':raise SystemExit(main())
