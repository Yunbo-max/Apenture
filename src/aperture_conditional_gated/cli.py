"""Conditional-gated CG-v1: independent cases, explicit execution, no legacy relabeling."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from aperture_rebuttal_cases import store
from .plan import CASES,REVIEW_MAP,freeze,source_files

REPO=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG='configs/aperture_conditional_gated_v1.json'
_PRECHECKED_MODELS=set()


def filter_units(units,unit=None,model=None,domain=None,seed=None):
    selected=[u for u in units if (unit is None or u['key']==unit) and (model is None or u['model_key']==model)
              and (domain is None or u['domain']==domain) and (seed is None or u['support_seed']==seed)]
    if not selected:raise ValueError('No registered comparison units match; do not invent a task/model')
    return selected


def preflight(unit,plan):
    from aperture_rebuttal_cases.data import load_roles
    from aperture_comprehensive.worker import check_assets
    data=load_roles(unit,images=True)
    check_assets(unit,Path(unit['output']).parents[2]/'assets',refresh=unit['model_path'] not in _PRECHECKED_MODELS)
    _PRECHECKED_MODELS.add(unit['model_path'])
    if any(v['engine']=='loreft' for v in unit['variants']):
        try:from pyreft import LoreftIntervention
        except ImportError as e:raise RuntimeError('E02 requires official pyreft; install in the isolated experiment environment') from e
    receipt={'source':plan['source_files'],'unit':unit['key'],'manifests':{k:store.sha(unit[k]) for k in data},
             'counts':{k:len(v) for k,v in data.items()},'no_gpu_forward':True}
    store.atomic(Path(unit['output'])/'preflight.json',receipt,immutable=True)
    return receipt


def task_reference(plan, unit, variant, stage, repo):
    path=Path(plan['settings']['run_root'])/plan['case']/'plan.json'
    return {'stage':stage,'unit_key':unit['key'],'variant_name':variant['name'] if variant else None,
            'plan_path':str(path),'plan_sha256':store.sha(path),'repo':str(repo)}


def subprocess_stage(stage,unit,variant,plan,repo,gpu):
    folder=Path(unit['output']);logs=folder/'logs';logs.mkdir(parents=True,exist_ok=True)
    tag=stage+('--'+variant['name'] if variant else '')
    task=task_reference(plan,unit,variant,stage,repo)
    path=logs/(tag+'.job.json');store.atomic(path,task,immutable=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1',
         'APERTURE_BATCH':str(plan['settings']['execution_batch']),
         'PYTHONPATH':str(repo/'src')+os.pathsep+os.environ.get('PYTHONPATH','')}
    command=[sys.executable,str(repo/'scripts/run_aperture_gated.py'),'internal','--task',str(path)]
    print(f"{unit['key']} {tag}",flush=True)
    with (logs/(tag+'.log')).open('a',encoding='utf-8') as stream:
        done=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
    store.atomic(logs/(tag+'.exit.json'),{'command':command,'returncode':done.returncode})
    if done.returncode:raise RuntimeError(f'Case stopped at {tag}; see {logs/(tag+".log")}')


def execute(plan,units,stage,repo,gpu):
    from .engine import seal_selection
    for u in units:
        with store.lock(Path(u['output'])/'unit.lock'):
            # A normal failed process releases these OS locks automatically.
            preflight(u,plan)
            if stage in ('audit','fit','run'):
                subprocess_stage('audit',u,None,plan,repo,gpu)
            if stage in ('fit','run') and u['case']!='CG00':
                for v in u['variants']:subprocess_stage('fit',u,v,plan,repo,gpu)
                seal_selection(u,plan)
            if stage in ('evaluate','run') and u['case']!='CG00':
                selected=store.read(Path(u['output'])/'selection.json')['evaluate']
                for v in u['variants']:
                    if v['name'] in selected:subprocess_stage('evaluate',u,v,plan,repo,gpu)
    return {'status':'requested_stages_completed','case':plan['case'],'units':len(units),
            'scientific_case_complete':'run summarize to validate full declared scope'}


def internal(task):
    from . import engine
    t=store.read(task)
    if store.sha(t['plan_path'])!=t['plan_sha256']:raise ValueError('Frozen job plan changed')
    p=store.read(t['plan_path'])
    u=next(u for u in p['units'] if u['key']==t['unit_key'])
    v=next((v for v in u['variants'] if v['name']==t['variant_name']),None)
    if t['stage']!='audit' and v is None:raise ValueError('Unknown registered variant')
    # Revalidate frozen source before executing a serialized job.
    if p['source_files']!=source_files(t['repo']):raise ValueError('Source changed after case was frozen')
    if t['stage']=='audit':engine.audit(u,p)
    elif t['stage']=='fit':engine.fit(u,v,p)
    elif t['stage']=='evaluate':engine.evaluate(u,v,p,t['repo'])
    else:raise ValueError('Unknown internal stage')
    return {'stage':t['stage'],'unit':u['key'],'status':'complete'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['list','prepare','plan','check','audit','fit','evaluate','run','status','summarize','internal'])
    parser.add_argument('--case',choices=sorted(CASES))
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--unit');parser.add_argument('--model');parser.add_argument('--domain');parser.add_argument('--seed',type=int)
    parser.add_argument('--gpu',default='0');parser.add_argument('--execute',action='store_true');parser.add_argument('--task')
    args=parser.parse_args(argv)
    try:
        if args.action=='list':
            print(json.dumps({'cases':CASES,'reviewer_mapping':REVIEW_MAP},indent=2));return 0
        if args.action=='internal':
            if not args.task:raise ValueError('Internal job path required')
            print(json.dumps(internal(args.task),indent=2));return 0
        if not args.case:raise ValueError('Specify exactly one --case; no implicit all-case launch')
        if not args.gpu.isdigit():raise ValueError('Use one nonnegative GPU index per invocation')
        cfg=store.read(REPO/args.config)
        for field in ('base_config','run_root','prepared_root','medical_data_root','textvqa_train_json','textvqa_val_json','textvqa_images'):
            cfg[field]=str((REPO/cfg[field]).resolve())
        if args.action=='prepare':
            if not args.execute:result={'status':'dry_run','case':args.case,'execute_required':True}
            elif args.case=='CG03':
                from aperture_rebuttal_cases.data import prepare_support
                result=prepare_support(cfg,REPO)
            elif args.case=='CG07':
                from aperture_rebuttal_cases.data import prepare_vqa
                result=prepare_vqa(cfg,REPO)
            else:
                raise ValueError('This case reuses the comprehensive prepared manifests; run its native prepare first. No data is silently regenerated.')
            print(json.dumps(result,indent=2));return 0
        plan=freeze(args.case,cfg,REPO)
        os.environ['APERTURE_BATCH']=str(plan['settings']['execution_batch'])
        if args.action=='plan':
            result={'case':args.case,'units':len(plan['units']),'fits':sum(len(u['variants']) for u in plan['units']),
                    'unit_ids':[u['key'] for u in plan['units']],'state':'planned_not_run'}
        elif args.case=='CG09':
            if args.action not in ('run','summarize','status'):raise ValueError('CG09 consumes sealed CG01 evidence')
            if not args.execute:result={'status':'dry_run','source':'complete predeclared CG01 Qwen2.5 four-BRIGHT/three-seed panel','execute_required':True}
            else:
                from .report import analyze
                source=freeze('CG01',cfg,REPO);scope=CASES['CG09']
                us=[u for u in source['units'] if u['model_key'] in scope['models'] and u['domain'] in scope['datasets'] and u['support_seed'] in scope['seeds']]
                if len(us)!=12:raise ValueError('CG09 requires all four BRIGHT domains and three seeds on Qwen2.5')
                result=analyze(source,Path(cfg['run_root'])/'CG09/report',units=us)
        elif args.action in ('status','summarize'):
            from .report import coverage,summarize
            if args.case=='CG00':
                status=coverage([Path(u['output'])/'audit' for u in plan['units']])
                store.atomic(Path(cfg['run_root'])/'CG00/report/audit_coverage.json',status)
                result=status
                if args.action=='summarize' and not status['complete']:raise ValueError('Incomplete native audit population')
            elif args.action=='summarize':result=summarize(plan)
            else:
                result={'case':args.case,'units':{u['key']:{'fits':coverage([Path(u['output'])/'fits'/v['name'] for v in u['variants']]),
                  'selection_sealed':(Path(u['output'])/'selection_receipt/DONE.json').exists()} for u in plan['units']}}
        else:
            us=filter_units(plan['units'],args.unit,args.model,args.domain,args.seed)
            if args.action=='check':result={'checked':[preflight(u,plan)['unit'] for u in us]}
            elif not args.execute:result={'status':'dry_run','case':args.case,'stage':args.action,'units':[u['key'] for u in us],'execute_required':True}
            else:
                if args.case=='CG00' and args.action not in ('run','audit'):raise ValueError('CG00 is support-only audit, not fitting/query evaluation')
                with store.lock(Path(cfg['run_root'])/f'.gpu-{args.gpu}.lock'):
                    result=execute(plan,us,args.action,REPO,args.gpu)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print('Conditional-gated case blocked: '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
