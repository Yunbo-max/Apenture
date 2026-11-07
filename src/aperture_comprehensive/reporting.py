"""Complete-only, prediction-recomputed exports for the comprehensive matrix."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from vigor_handoff.protocol import atomic_json, file_hash, digest, read_jsonl
from aperture_final.numeric import metrics
from .protocol import make_jobs, read_roles, source_signature
from .worker import verified_record, validate_rows, verify_selection, verify_repeat, identity


def status(config):
    jobs=make_jobs(config);counts={'planned':len(jobs),'fit_complete':0,'evaluation_complete':0,'invalid':[]}
    coverage=defaultdict(lambda:{'planned':0,'fits':0,'evaluations':0})
    for job in jobs:
        cell=coverage[job['model_key']+' / '+job['domain_key']];cell['planned']+=1
        for phase,field,short in [('fit','fit_complete','fits'),('evaluation','evaluation_complete','evaluations')]:
            folder=Path(job['output'])/phase
            if not (folder/'DONE.json').exists():continue
            try:verified_record(folder)
            except (ValueError,KeyError,OSError) as exc:counts['invalid'].append({'job':job['job_id'],'phase':phase,'error':str(exc)})
            else:counts[field]+=1;cell[short]+=1
    counts['coverage']=dict(coverage)
    counts['complete']=counts['evaluation_complete']==len(jobs) and not counts['invalid']
    return counts


def export(config):
    root=Path(config['run_root']);out=root/'exports';out.mkdir(parents=True,exist_ok=True)
    # Revoke stale complete exports before any verification that could fail.
    for name in ('results.csv','summary.csv','provenance.json'):(out/name).unlink(missing_ok=True)
    progress=status(config);atomic_json(root/'status.json',progress)
    if not progress['complete']:raise ValueError('incomplete or invalid matrix; no scientific aggregate exported')
    records=[];query_ids={};seals={}
    for job in make_jobs(config):
        roles=read_roles(job,images=True)
        audit=root/'audits'/job['model_key']/job['domain_key'];verified_record(audit)
        if not json.loads((audit/'audit.json').read_text())['passed']:raise ValueError('failed audit')
        verify_repeat(job,root);selection=verify_selection(job,root/'selections'/(job['unit_id']+'.json'))
        fit=Path(job['output'])/'fit';evaluation=Path(job['output'])/'evaluation'
        fit_done=verified_record(fit);verified_record(evaluation)
        current=identity(job,root/'assets');saved=json.loads((fit/'identity.json').read_text())['identity']
        if any(saved.get(k)!=v for k,v in current.items()):raise ValueError('stale fitted state')
        if saved['audit_seal']!=file_hash(audit/'DONE.json'):raise ValueError('audit seal changed')
        eval_identity=json.loads((evaluation/'identity.json').read_text())['identity']
        if eval_identity != {'fit_seal':digest(fit_done),
                             'selection':file_hash(root/'selections'/(job['unit_id']+'.json')),
                             'identity':current}:
            raise ValueError('evaluation belongs to another fit or selection')
        rows=read_jsonl(evaluation/'predictions.jsonl');validate_rows(rows,roles['query'])
        ids=[r['sample_id'] for r in rows]
        if job['domain_key'] in query_ids and query_ids[job['domain_key']]!=ids:raise ValueError('queries differ across models/seeds/arms')
        query_ids[job['domain_key']]=ids
        from .numeric import score_arrays
        scores,labels=score_arrays(rows,job['classes'])
        cal=json.loads((fit/'calibration.json').read_text());stats=json.loads((fit/'fitting.json').read_text())
        recorded=json.loads((evaluation/'evaluation.json').read_text())
        for view,t,bias in [('raw',1.,None),('temperature',cal['temperature']['temperature'],None)]+([('bias',1.,cal['bias']['values'])] if job['arm']=='frozen' else []):
            measured=metrics(scores,labels,t,bias=bias)
            if measured!=recorded[view]:raise ValueError('saved metrics disagree with predictions')
            records.append(dict(model=job['model_key'],model_id=job['model_id'],dataset=job['domain_key'],
                seed=job['support_seed'],arm=job['arm'],view=view,**measured,
                optimizer_updates=stats['optimizer_updates'],fit_visits=stats['fitting_visits'],
                basis_visits=stats['basis_visits'],state_bytes=stats['serialized_state_bytes'],
                fit_seconds=stats['fit_wall_seconds'],image_size=job['image_size']))
        seals[job['job_id']]={'fit':file_hash(fit/'DONE.json'),'evaluation':file_hash(evaluation/'DONE.json')}
    fields=list(dict.fromkeys(k for r in records for k in r))
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    groups=defaultdict(list)
    for row in records:groups[(row['model'],row['dataset'],row['arm'],row['view'])].append(row)
    summary=[]
    for key,rows in groups.items():
        row=dict(zip(('model','dataset','arm','view'),key));row['seeds']=len(rows)
        for metric in ('accuracy','macro_f1','nll','brier','ece'):
            if metric not in rows[0]:continue
            values=[r[metric] for r in rows];row[metric+'_mean']=float(np.mean(values))
            row[metric+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(row)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    provenance={'complete':True,'source':source_signature(),'states':len(seals),'seals':seals,
                'query_ids':{k:digest(v) for k,v in query_ids.items()},
                'note':'Q2 classes denote candidate positions. Seed SD is descriptive, not patient-level uncertainty.'}
    atomic_json(out/'provenance.json',provenance)
    return {'complete':True,'states':len(seals),'records':len(records),'output':str(out)}
