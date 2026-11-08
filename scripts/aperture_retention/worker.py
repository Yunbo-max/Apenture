"""Evaluation-only E10 workers; old training states are read-only."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np
from aperture_rebuttal_cases import store
from . import protocol,metrics,text


def public_visual(row):
    out={k:row[k] for k in ('sample_id','group_id','question','candidate_labels','image','pre_image','post_image','bbox_xyxy') if k in row}
    # Native score's label fields are output metadata only. Supply a dummy label.
    out.update(label_id=0,label=out['candidate_labels'][0])
    return out


def validate_predictions(predictions,rows,complete=True):
    if complete and len(predictions)!=len(rows):raise ValueError('Incomplete predictions')
    if len(predictions)>len(rows):raise ValueError('Too many predictions')
    for p,r in zip(predictions,rows):
        if any(p.get(k)!=r[k] for k in ('sample_id','group_id','label_id')):raise ValueError('Prediction population/labels differ')
        count=len(r.get('candidate_labels',r.get('candidates',[])))
        if len(p['scores'])!=count or not p.get('prompt_contract_hash') or not np.isfinite(p['scores']).all():raise ValueError('Invalid prediction contract')


def state_bytes(folder):
    folder=Path(folder)
    return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and
               (p.name in ('controller.pt','deployment.json') or 'adapter' in p.relative_to(folder).parts))


def collect(folder,rows,scorer):
    """Resume an exact prefix; never truncate a corrupt tail or invent missing scores."""
    folder=Path(folder);folder.parent.mkdir(parents=True,exist_ok=True)
    saved=store.rows(folder) if folder.exists() else []
    validate_predictions(saved,rows,complete=False)
    with folder.open('a',encoding='utf-8') as f:
        for r in rows[len(saved):]:
            values=scorer(r)
            out={**values,**{k:r[k] for k in ('sample_id','group_id','label_id')}}
            validate_predictions([out],[r])
            f.write(json.dumps(out,allow_nan=False)+'\n');f.flush();saved.append(out)
    validate_predictions(saved,rows)
    return saved


def visual_score(backend,row,controller):
    out=backend.score(public_visual(row),controller)
    return {'scores':out['mean_log_scores'],'prompt_contract_hash':out['prompt_contract_hash'],
            'image_tokens':out.get('image_tokens',out.get('visual_tokens'))}


@contextmanager
def source_batch(count):
    old=os.environ.get('APERTURE_BATCH');os.environ['APERTURE_BATCH']=str(count)
    try:yield
    finally:
        if old is None:os.environ.pop('APERTURE_BATCH',None)
        else:os.environ['APERTURE_BATCH']=old


def verify_pixels(rows):
    from aperture_comprehensive.data import content_hash
    for r in rows:
        if not r.get('content_hash') or r['content_hash']!=content_hash(r):raise ValueError('Query pixels changed or no content hash')


def load_queries(plan,sources,domain):
    unit=sources['frozen']['unit']
    from aperture_rebuttal_cases.data import load_roles
    roles=load_roles(unit,images=True)
    q=store.rows(plan['queries'][domain]['path'])
    if len({r['sample_id'] for r in q})!=len(q):raise ValueError('Duplicate query IDs')
    base=store.read(plan['settings']['base_config']);spec=next(d for d in base['datasets'] if d['key']==domain)
    if spec.get('query_count') is not None and len(q)!=spec['query_count']:raise ValueError('Wrong full query count')
    verify_pixels(q);protocol.no_overlap(roles['support']+roles['calibration'],q)
    if domain==unit['domain'] and store.digest(q)!=store.digest(roles['query']):raise ValueError('Source-task query differs from original split')
    return q


def check_job(plan,job):
    protocol.verify_plan(plan)
    domains=job.get('available_domains',[job['domain']]);sources={}
    for domain in domains:
        ss=protocol.resolve_sources(plan,job,domain);sources[domain]=ss
        if set(ss)!=(set(job['methods'])|{'frozen'}):raise ValueError('Incomplete declared method set')
        # load_roles checks exact counts, group isolation, manifests and pixels.
        from aperture_rebuttal_cases.data import load_roles
        for s in ss.values():load_roles(s['unit'],images=True)
        if plan['part']=='bank':
            for name in job['methods']:
                s=ss[name];u=s['unit'];root=Path(u['output']);ev=root/'evaluation'/s['variant']['name']
                selection=store.read(root/'selection.json')
                store.verify(root/'selection_receipt',{'unit':u['key'],'plan':s['source_plan_digest']})
                if selection!=store.read(root/'selection_receipt/selection.json') or selection.get('query_metrics_used') is not False:
                    raise ValueError('Invalid original bank selection')
                if s['variant']['name'] not in selection['evaluate'] or selection['fit_seals'][s['variant']['name']]!=s['seal']:
                    raise ValueError('Bank variant/selection mismatch')
                store.verify(ev,{'fit_seal':s['seal'],'selection':store.sha(root/'selection.json'),'identity':s['binding']})
                s['reference_evaluation']=str(ev);s['reference_seal']=store.sha(ev/'DONE.json')
    if plan['part']=='visual':
        for d in job['targets']:load_queries(plan,sources[job['domain']],d)
    elif plan['part']=='bank':
        for d in domains:load_queries(plan,sources[d],d)
    else:
        provenance=store.read(plan['queries']['_provenance']['path'])
        if provenance.get('datasets')!=text.DATASETS:raise ValueError('Text dataset revision/spec changed')
        for d in job['targets']:
            q=store.rows(plan['queries'][d]['path']);text.check_count(d,q)
            if provenance['files'][d]!=store.sha(plan['queries'][d]['path']):raise ValueError('Text preparation receipt changed')
    return sources


def compact_binding(plan,job,sources):
    return {'plan':store.digest(plan),'job':job,'source_fits':{
        d:{name:{k:s[k] for k in ('folder','seal','source_plan_digest','binding')} for name,s in ss.items()}
        for d,ss in sources.items()},'old_query_temperature_used':False,'query_batch_size':1}


def restore_replay(source,backend,controller):
    from aperture_rebuttal_cases.engine import score,compare_replay
    from aperture_rebuttal_cases.data import load_roles
    roles=load_roles(source['unit'],images=True)
    with source_batch(source['stats'].get('batch_size',1)):
        got=score(backend,roles['calibration'],controller)
    delta=compare_replay(store.rows(Path(source['folder'])/'calibration.jsonl'),got)
    return {'max_abs_calibration_score_difference':delta,'tolerance':.005,'source_batch':source['stats'].get('batch_size',1)}


def _active(plan,job,sources,folder):
    import torch
    from aperture_rebuttal_cases.adapters import loaded
    domain=job['domain'];ss=sources[domain];results={};replays={};identity_checks={}
    # Source-domain gain is measured under the same restored states, even in text part.
    from aperture_rebuttal_cases.data import load_roles
    own=load_roles(ss['frozen']['unit'],images=True)['query']
    querymap={domain:own}
    if plan['part']=='visual':
        querymap.update({d:load_queries(plan,ss,d) for d in job['targets']})
    else:querymap.update({d:store.rows(plan['queries'][d]['path']) for d in job['targets']})
    for name in job['methods']:
        source=ss[name]
        with loaded(source['unit'],source['variant'],Path(source['folder'])) as (model,backend,controller):
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            if controller is not None:
                for p in controller.parameters():p.requires_grad_(False)
            replays[name]=restore_replay(source,backend,controller)
            original=deepcopy(backend.job)
            for target,rows in querymap.items():
                is_text=target in text.DATASETS
                if is_text:
                    scorer=lambda row:text.score_text(model,backend.processor,text.public_row(row),controller)
                else:
                    backend.job=protocol.eval_unit(original,target)
                    scorer=lambda row:visual_score(backend,row,controller)
                predicted=collect(folder/'predictions'/target/(name+'.jsonl'),rows,scorer)
                results.setdefault(target,{})[name]=predicted
                if is_text and source['variant']['engine']=='lora':
                    # This is a separate restoration control, never the active LoRA result.
                    with model.disable_adapter():
                        disabled=collect(folder/'predictions'/target/(name+'-disabled.jsonl'),rows,
                            lambda row:text.score_text(model,backend.processor,text.public_row(row),None))
                    results[target][name+'-disabled']=disabled
            backend.job=original
        # Disk states must not change as a side-effect of evaluation.
        store.verify(source['folder'],source['binding'])
        if store.sha(Path(source['folder'])/'DONE.json')!=source['seal']:raise ValueError('Source changed during evaluation')
    comparisons={target:{name:metrics.compare(allrows['frozen'],rows) for name,rows in allrows.items() if name!='frozen'}
                 for target,allrows in results.items()}
    if plan['part']=='text':
        for target in text.DATASETS:
            for name in ['aperture',*[n+'-disabled' for n in ('lora1','lora4','lora-r4')]]:
                c=comparisons[target][name]
                identity_checks[target+'/'+name]={
                    'passed':c['max_abs_score_difference']<=.005 and c['prediction_disagreement']==0,
                    'max_abs_score_difference':c['max_abs_score_difference'],'prediction_disagreement':c['prediction_disagreement'],
                    'note':'Measured identity check, not a guarantee inferred from frozen base weights'}
    return {'part':plan['part'],'unit':job['key'],'new_domain':domain,'comparisons':comparisons,
            'replay':replays,'identity_checks':identity_checks,'all_identity_checks_passed':all(v['passed'] for v in identity_checks.values()),
            'state_bytes':{k:state_bytes(v['folder']) if k!='frozen' else 0 for k,v in ss.items()},
            'training_performed':False,'old_task_selection':False,'claim_scope':'active-adapter retention, not sequential forgetting'}


def _bank(plan,job,sources,folder):
    from aperture_rebuttal_cases.adapters import loaded
    from aperture_rebuttal_cases.engine import compare_replay
    result=[];totals={m:0 for m in job['methods']}
    for domain in job['available_domains']:
        ss=sources[domain];query=load_queries(plan,ss,domain)
        for name in job['methods']:
            s=ss[name]
            with loaded(s['unit'],s['variant'],Path(s['folder'])) as (model,backend,controller):
                model.eval();replay=restore_replay(s,backend,controller)
                prediction=collect(folder/'predictions'/domain/(name+'.jsonl'),query,lambda r:visual_score(backend,r,controller))
            reference=store.rows(Path(s['reference_evaluation'])/'predictions.jsonl')
            validate_predictions([{**r,'scores':r['mean_log_scores']} for r in reference],query)
            ref=[dict(r,scores=r['mean_log_scores']) for r in reference]
            comparison=metrics.compare(ref,prediction)
            totals[name]+=state_bytes(s['folder'])
            store.verify(s['folder'],s['binding'])
            result.append({'domain':domain,'method':name,'replay':replay,**comparison,
                           'identity_passed':comparison['max_abs_score_difference']<=.005 and comparison['prediction_disagreement']==0})
    return {'part':'bank','unit':job['key'],'stage':job['stage'],'available_domains':job['available_domains'],
            'retrievals':result,'cumulative_adaptation_bytes':totals,'shared_base_counted':False,
            'all_identity_checks_passed':all(r['identity_passed'] for r in result),
            'task_identity_given_to_all_methods':True,'training_performed':False,
            'chronology':'replay of independently trained archived modules, NOT sequential shared-adapter learning'}


def run(plan,job):
    from aperture_rebuttal_cases.engine import setup
    setup();sources=check_job(plan,job);binding=compact_binding(plan,job,sources)
    folder=Path(plan['settings']['run_root'])/plan['part']/'units'/job['key']
    with store.lock(folder.parent/(folder.name+'.lock')):
        if (folder/'DONE.json').exists():
            store.verify(folder,binding);return store.read(folder/'result.json')
        store.atomic(folder/'binding.json',binding,immutable=True)
        result=_bank(plan,job,sources,folder) if plan['part']=='bank' else _active(plan,job,sources,folder)
        protocol.verify_plan(plan)
        store.atomic(folder/'result.json',result);store.seal(folder,binding)
        return {'unit':job['key'],'status':'complete','all_identity_checks_passed':result['all_identity_checks_passed']}
