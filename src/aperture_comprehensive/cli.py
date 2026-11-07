"""Prepare, audit, fit, evaluate and resume complete comparison units."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from aperture_table.cli import gpu_batches
from vigor_handoff.protocol import atomic_json
from .protocol import MODELS, ARMS, make_jobs, read_roles, source_signature, write_locked


def lock_plan(config):
    jobs=make_jobs(config)
    plan=dict(config=config,source=source_signature(),jobs=jobs,states=len(jobs),
              trainable_fits=sum(j['arm']!='frozen' for j in jobs),query_labels_used_for_selection=False)
    write_locked(Path(config['run_root'])/'plan.json',plan)
    return plan


def select_jobs(jobs,models=None,datasets=None,seeds=None):
    for field,values in [('model_key',models),('domain_key',datasets),('support_seed',seeds)]:
        if values and not set(values)<={j[field] for j in jobs}:raise ValueError(f'unknown {field} filter')
    selected=[j for j in jobs if (not models or j['model_key'] in models)
              and (not datasets or j['domain_key'] in datasets) and (not seeds or j['support_seed'] in seeds)]
    if not selected:raise ValueError('empty job selection')
    return selected


def child(config_path,root,job,action,gpu,selection=None):
    name=job['job_id']+'--'+action
    filename=Path(root)/'queue'/(name+'.json');write_locked(filename,job)
    logs=Path(root)/'logs';logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    cmd=[sys.executable,'-m','aperture_comprehensive.cli','--config',str(Path(config_path).resolve()),
         '_worker','--action',action,'--job',str(filename.resolve())]
    if selection:cmd+=['--selection',str(Path(selection).resolve())]
    with (logs/(name+'.log')).open('a') as stream:
        result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed: {logs/(name+".log")}')
    print(f'OK {name}',flush=True)


def canonical_audits(all_jobs, selected):
    keys=list(dict.fromkeys((j['model_key'],j['domain_key']) for j in selected))
    return [next(j for j in all_jobs if (j['model_key'],j['domain_key'])==key
                 and j['arm']=='aperture') for key in keys]


def run(config,args):
    from .worker import seal_unit,compare_repeat
    all_jobs=lock_plan(config)['jobs']
    jobs=select_jobs(all_jobs,args.models,args.datasets,args.seeds)
    root=Path(config['run_root']);gpus=args.gpus.split(',')
    if not gpus or any(not g.isdigit() for g in gpus) or len(set(gpus))!=len(gpus):
        raise ValueError('use unique comma-separated local GPU IDs')
    # Every selected model/task is audited. The canonical repeat uses a fixed reference task.
    models=list(dict.fromkeys(j['model_key'] for j in jobs))
    ref_domain=config.get('repeat_reference',config['datasets'][0]['key'])
    refs=[next(j for j in all_jobs if j['model_key']==m and j['domain_key']==ref_domain
               and j['support_seed']==config['support_seeds'][0] and j['arm']=='aperture') for m in models]
    audit_jobs=canonical_audits(all_jobs,jobs+(refs if args.command=='run' else []))
    gpu_batches(audit_jobs,gpus,lambda j,g:child(args.config,root,j,'audit',g))
    if args.command=='audit':return
    if args.command=='run':
        def repeat(job,gpu):
            replica=deepcopy(job);replica['job_id']+='--repeat';replica['output']=str(root/'repeat'/job['model_key']/'state')
            child(args.config,root,job,'fit',gpu);child(args.config,root,replica,'fit',gpu)
            compare_repeat(job,replica['output'],root/'repeat'/job['model_key']/'gate.json')
        gpu_batches(refs,gpus,repeat)
    if args.command=='smoke':
        # Smoke uses the actual fixed recipe and budget; no query images are scored.
        jobs=[j for j in jobs if j['support_seed']==min(x['support_seed'] for x in jobs)
              and (not args.arms or j['arm'] in args.arms)]
    def fit_and_replay(job,gpu):
        child(args.config,root,job,'fit',gpu)
        if args.command=='smoke':child(args.config,root,job,'replay',gpu)
    gpu_batches(jobs,gpus,fit_and_replay)
    if args.command in ('smoke','fit'):return
    units={}
    for job in jobs:units.setdefault(job['unit_id'],[]).append(job)
    for unit,group in units.items():seal_unit(group,root/'selections'/(unit+'.json'))
    gpu_batches(jobs,gpus,lambda j,g:child(args.config,root,j,'evaluate',g,root/'selections'/(j['unit_id']+'.json')))


def parser():
    p=argparse.ArgumentParser(description='Six-backbone, five-arm comprehensive suite; fixed plans and support-only gates.')
    p.add_argument('--config',default='configs/aperture_comprehensive_v1.json')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','prepare','check','status','summarize'):sub.add_parser(name)
    for name in ('audit','smoke','fit','run'):
        q=sub.add_parser(name);q.add_argument('--gpus',default='0,1')
        q.add_argument('--models',nargs='+',choices=list(MODELS));q.add_argument('--datasets',nargs='+')
        q.add_argument('--seeds',nargs='+',type=int)
        if name=='smoke':q.add_argument('--arms',nargs='+',choices=ARMS)
    q=sub.add_parser('_worker');q.add_argument('--action',choices=['audit','fit','replay','evaluate'],required=True)
    q.add_argument('--job',required=True);q.add_argument('--selection')
    return p


def main(argv=None):
    args=parser().parse_args(argv);config=json.loads(Path(args.config).read_text());root=Path(config['run_root'])
    if args.command=='plan':
        plan=lock_plan(config);print(json.dumps({k:plan[k] for k in ('states','trainable_fits','source')},indent=2))
    elif args.command=='prepare':
        from .data import prepare
        lock_plan(config);prepare(config)
    elif args.command=='check':
        from .worker import check_assets
        jobs=lock_plan(config)['jobs'];seen=set()
        for job in jobs:
            key=(job['domain_key'],job['support_seed'])
            if key not in seen:read_roles(job,images=True);seen.add(key)
        for model in MODELS:
            check_assets(next(j for j in jobs if j['model_key']==model),root/'assets',refresh=True)
        atomic_json(root/'preflight.json',dict(passed=True,source=source_signature(),states=len(jobs),partitions=len(seen)))
        print(f'PASS: {len(jobs)} states, {len(seen)} partitions, all six model snapshots')
    elif args.command in ('run','fit','audit','smoke'):run(config,args)
    elif args.command=='status':
        from .reporting import status
        print(json.dumps(status(config),indent=2))
    elif args.command=='summarize':
        from .reporting import export
        print(json.dumps(export(config),indent=2))
    elif args.command=='_worker':
        job=json.loads(Path(args.job).read_text())
        if args.action=='audit':
            from .audit import execute_audit
            print(execute_audit(job,root))
        elif args.action=='fit':
            from .worker import execute_fit
            print(execute_fit(job,root))
        elif args.action=='replay':
            from .worker import execute_replay
            print(execute_replay(job,root))
        else:
            from .worker import execute_evaluate
            print(execute_evaluate(job,root,args.selection))


if __name__=='__main__':main()
