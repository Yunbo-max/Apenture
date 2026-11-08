"""One E10 part/unit per request. No model forward without --execute."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from aperture_rebuttal_cases import store
from . import protocol

REPO=Path(__file__).resolve().parents[2]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['list','prepare-text','plan','check','run','status','summarize','internal'])
    p.add_argument('--config',default='configs/aperture_retention_v1.json')
    p.add_argument('--part',choices=protocol.PARTS)
    p.add_argument('--unit');p.add_argument('--model',choices=protocol.MODELS);p.add_argument('--domain');p.add_argument('--seed',type=int,choices=(0,1,2))
    p.add_argument('--gpu',default='0');p.add_argument('--execute',action='store_true');p.add_argument('--task')
    a=p.parse_args(argv)
    try:
        if not a.gpu.isdigit():raise ValueError('One GPU index, e.g. --gpu 0; no --gpus or comma lists')
        if a.action=='list':
            result={k:{'units':len(protocol.job_specs(k)),'jobs':protocol.job_specs(k)} for k in protocol.PARTS}
        elif a.action=='internal':
            if not a.task:raise ValueError('Missing internal task')
            t=store.read(a.task)
            if store.sha(t['plan'])!=t['plan_sha256']:raise ValueError('Task plan changed')
            plan=store.read(t['plan']);job=next(j for j in plan['jobs'] if j['key']==t['unit'])
            from .worker import run
            result=run(plan,job)
        elif a.action=='prepare-text':
            cfg=protocol.resolve_config(REPO,REPO/a.config)
            if a.execute:
                from .text import prepare
                result=prepare(cfg['text_root'])
            else:result={'status':'dry_run','action':'download pinned full ARC-Challenge test and BoolQ validation','execute_required':True}
        else:
            if a.part is None:raise ValueError('Select --part visual, text or bank')
            cfg=protocol.resolve_config(REPO,REPO/a.config);plan=protocol.freeze(REPO,cfg,a.part)
            jobs=protocol.select_jobs(plan['jobs'],a.unit,a.model,a.domain,a.seed)
            if a.action=='plan':result={'part':a.part,'planned_units':len(plan['jobs']),'selected_units':[j['key'] for j in jobs],'training':False}
            elif a.action=='check':
                from .worker import check_job
                for j in jobs:check_job(plan,j)
                result={'checked':[j['key'] for j in jobs],'gpu_forward':False}
            elif a.action in ('status','summarize'):
                if any(x is not None for x in (a.unit,a.model,a.domain,a.seed)):raise ValueError('Reporting uses full frozen part, not a selected subset')
                from .report import coverage,aggregate
                result=coverage(plan) if a.action=='status' else aggregate(plan)
            elif not a.execute:result={'status':'dry_run','part':a.part,'units':[j['key'] for j in jobs],'execute_required':True}
            else:
                from .worker import check_job
                root=Path(cfg['run_root']);pp=root/a.part/'plan.json';attempts=[]
                with store.lock(root/f'.gpu-{a.gpu}.lock'):
                    for j in jobs:
                        check_job(plan,j)
                        logs=root/a.part/'logs';logs.mkdir(parents=True,exist_ok=True)
                        task=logs/(j['key']+'.job.json');store.atomic(task,{'plan':str(pp),'plan_sha256':store.sha(pp),'unit':j['key']},immutable=True)
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0',
                             'APERTURE_BATCH':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1',
                             'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
                        log=logs/(j['key']+'.log');print(f'E10/{a.part}: {j["key"]} -> {log}',flush=True)
                        with log.open('a') as f:
                            r=subprocess.run([sys.executable,str(REPO/'scripts/run_aperture_retention.py'),'internal','--task',str(task)],
                                             cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
                        store.atomic(logs/(j['key']+'.exit.json'),{'returncode':r.returncode})
                        if r.returncode:raise RuntimeError(f'E10 stopped; see {log}')
                        attempts.append(j['key'])
                result={'completed_requested_units':attempts,'full_scope':'run summarize; no benchmark improvement is presumed'}
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (OSError,ValueError,RuntimeError,KeyError,ImportError) as e:
        print('E10 blocked: '+str(e),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
