"""Calibration-only selection and complete-case, paired descriptive reports."""
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from .store import atomic, digest, read, rows, sha, verify


def select(records):
    groups=defaultdict(list)
    for r in records:
        if not math.isfinite(r['calibration_nll']):raise ValueError('Nonfinite calibration score')
        groups[r['family']].append(r)
    return {k:min(v,key=lambda r:(r['calibration_nll'],r['name']))['name'] for k,v in groups.items()}


def coverage(paths):
    missing=[];invalid=[];completed=0
    for path in paths:
        p=Path(path)
        if not (p/'DONE.json').exists():missing.append(str(p));continue
        try:verify(p);completed+=1
        except (ValueError,OSError,KeyError) as exc:invalid.append({'path':str(p),'error':str(exc)})
    return {'complete':not missing and not invalid,'planned':len(paths),'completed':completed,'missing':missing,'invalid':invalid}


def metrics(data):
    if not data:raise ValueError('Empty prediction population')
    p=[np.asarray(r['probabilities'],float) for r in data];ys=[r['label_id'] for r in data]
    if any(not np.isfinite(a).all() or (a<0).any() or abs(a.sum()-1)>1e-6 for a in p):raise ValueError('Invalid probabilities')
    logps=[]
    for row,a,y in zip(data,p,ys):
        if type(y) is not int or not 0<=y<len(a):raise ValueError('Invalid label index')
        if 'mean_log_scores' in row:
            z=np.asarray(row['mean_log_scores'],float)
            if z.shape!=a.shape or not np.isfinite(z).all():raise ValueError('Invalid scores')
            z=z-z.max();lp=z-np.log(np.exp(z).sum())
            if not np.allclose(np.exp(lp),a,rtol=1e-7,atol=1e-8):raise ValueError('Probability/score mismatch')
        else:
            if (a<=0).any():raise ValueError('Stable raw scores required when probabilities underflow')
            lp=np.log(a)
        logps.append(lp)
        if 'prediction' in row and row['prediction']!=int(a.argmax()):raise ValueError('Prediction/score mismatch')
    pred=[int(a.argmax()) for a in p];classes=range(max(len(a) for a in p))
    f1=[];per_class={}
    for c in classes:
        tp=sum(y==c and z==c for y,z in zip(ys,pred));fp=sum(y!=c and z==c for y,z in zip(ys,pred));fn=sum(y==c and z!=c for y,z in zip(ys,pred))
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        per_class[str(c)]={'count':sum(y==c for y in ys),'recall':tp/(tp+fn) if tp+fn else None,'f1':f1[-1]}
    n=len(data)
    return {'n':n,'accuracy':sum(y==z for y,z in zip(ys,pred))/n,'macro_f1':sum(f1)/len(f1),
            'nll':float(np.mean([-lp[y] for lp,y in zip(logps,ys)])),
            'brier':float(np.mean([sum((a-np.eye(len(a))[y])**2) for a,y in zip(p,ys)])),
            'per_class':per_class}


def paired_accuracy(left,right,replicates=2000):
    if [r['sample_id'] for r in left]!=[r['sample_id'] for r in right]:raise ValueError('Paired IDs/order differ')
    grouped=defaultdict(list)
    for a,b in zip(left,right):
        if a['label_id']!=b['label_id'] or a['group_id']!=b['group_id']:raise ValueError('Paired labels/groups differ')
        grouped[a['group_id']].append(float(b['prediction']==b['label_id'])-float(a['prediction']==a['label_id']))
    values=list(grouped.values());s=np.array([sum(v) for v in values]);n=np.array([len(v) for v in values])
    rng=np.random.default_rng(0);boot=[]
    for _ in range(replicates):
        ix=rng.integers(0,len(values),len(values));boot.append(float(s[ix].sum()/n[ix].sum()))
    return {'difference':float(s.sum()/n.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'groups':len(values),'unit':'query groups; conditional on fitted support seed',
            'few_clusters_warning':len(values)<20}


def error_examples(left,right,limit=5):
    if [r['sample_id'] for r in left]!=[r['sample_id'] for r in right]:raise ValueError('Paired IDs differ')
    result={'fixed':[],'regressed':[],'both_wrong':[],'both_correct':[]}
    for a,b in zip(left,right):
        x=a['prediction']==a['label_id'];y=b['prediction']==b['label_id']
        name='both_correct' if x and y else 'regressed' if x else 'fixed' if y else 'both_wrong'
        result[name].append({'sample_id':a['sample_id'],'group_id':a['group_id'],'label_id':a['label_id'],
                             'left':a['prediction'],'right':b['prediction']})
    return {k:sorted(v,key=lambda r:digest(['qualitative-v1',r['sample_id']]))[:limit] for k,v in result.items()}


def expected(unit):
    folder=Path(unit['output']);selection=read(folder/'selection.json')
    return list(selection['evaluate'])


def summarize(plan):
    import csv
    out=Path(plan['settings']['run_root'])/plan['case']/'report'
    out.mkdir(parents=True,exist_ok=True)
    for name in ('results.json','results.csv','summary.json'):(out/name).unlink(missing_ok=True)
    store_state={'case':plan['case'],'status':'checking','units':{}}
    atomic(out/'coverage.json',store_state)
    result={'case':plan['case'],'status':'incomplete','units':{},'method_changed':False}
    complete=True;flat=[];queries={}
    for u in plan['units']:
        folder=Path(u['output'])
        if not (folder/'selection.json').is_file():
            failures=[read(p) for p in (folder/'logs').glob('*.exit.json') if read(p).get('returncode')]
            result['units'][u['key']]={'status':'failed' if failures else 'missing_selection','failures':failures};complete=False;continue
        names=expected(u);paths=[folder/'evaluation'/n for n in names];state=coverage(paths)
        result['units'][u['key']]=state
        if not state['complete']:complete=False;continue
        verify(folder/'selection_receipt',{'unit':u['key'],'plan':digest(plan)})
        selected=read(folder/'selection.json')
        if read(folder/'selection_receipt'/'selection.json')!=selected:raise ValueError('Selection changed')
        variants={v['name']:v for v in u['variants']}
        for name,h in selected['fit_seals'].items():
            fit=folder/'fits'/name;saved=verify(fit)['binding']
            if sha(fit/'DONE.json')!=h or saved['unit']!=u or saved['variant']!=variants[name] or saved['plan']!=digest(plan):
                raise ValueError('Selection refers to an incompatible fitted state')
            for role,h in saved['manifests'].items():
                if sha(u[role])!=h:raise ValueError('Prepared population changed after fit')
        evidence=[]
        for name in names:
            fit=folder/'fits'/name;record=read(folder/'evaluation'/name/'metrics.json')
            bound=read(folder/'evaluation'/name/'DONE.json')['binding']
            if bound['fit_seal']!=sha(fit/'DONE.json') or bound['selection']!=sha(folder/'selection.json'):
                raise ValueError('Evaluation fit/selection mismatch')
            predictions=rows(folder/'evaluation'/name/'predictions.jsonl')
            ids=[r['sample_id'] for r in predictions]
            if u['domain'] in queries and ids!=queries[u['domain']]:raise ValueError('Query population differs across model/seed/budget')
            queries[u['domain']]=ids
            if u['task_type']=='classification':
                if metrics(predictions)!=record['raw']:raise ValueError('Metrics disagree with sealed predictions')
                scaled=[]
                for r in predictions:
                    z=np.asarray(r['mean_log_scores'],float);z=(z-z.max())/record['temperature'];a=np.exp(z);a/=a.sum()
                    scaled.append({**r,'mean_log_scores':z.tolist(),'probabilities':a.tolist()})
                if metrics(scaled)!=record['temperature_scaled']:raise ValueError('Scaled metrics disagree with predictions')
                views=[('raw',record['raw']),('temperature_scaled',record['temperature_scaled'])]
            else:
                from .data import evaluator
                repo=Path(plan['settings']['base_config']).parents[1]
                refs=rows(u['query'])
                if ids!=[r['sample_id'] for r in refs]:raise ValueError('VQA question IDs differ')
                value=evaluator(plan['settings'],repo).TextVQAAccuracyEvaluator().eval_pred_list([
                    {'pred_answer':r['pred_answer'],'gt_answers':q['answers']} for r,q in zip(predictions,refs)])
                if value!=record['vqa_accuracy']:raise ValueError('VQA score differs from official evaluator')
                views=[('free_generation',record)]
            resources=read(fit/'fit.json');evidence.append({'variant':name,**record,'resources':resources})
            label=variants[name]['family'] if u['evaluate_policy']=='selected' else name
            for view,m in views:
                flat.append({'model':u['model_key'],'domain':u['domain'],'total_labels':u['total_labels'],
                    'seed':u['support_seed'],'method':label,'variant':name,'view':view,
                    **{k:m.get(k) for k in ('accuracy','macro_f1','nll','brier','vqa_accuracy')},
                    'state_bytes':resources['serialized_state_bytes'],'fit_seconds':resources['fit_seconds_including_extraction']})
        result['units'][u['key']]={'status':'complete','results':evidence,'calibration_selection':selected['selected']}
    atomic(out/'coverage.json',result)
    if not complete:raise ValueError('Incomplete case; coverage written, no scientific aggregate')
    result['status']='complete'
    groups=defaultdict(list)
    for r in flat:groups[(r['model'],r['domain'],r['total_labels'],r['method'],r['view'])].append(r)
    summary=[]
    for key,rr in groups.items():
        item=dict(zip(('model','domain','total_labels','method','view'),key));item['support_seeds']=len(rr)
        if len({r['seed'] for r in rr})!=len(rr):raise ValueError('Duplicate support seed in aggregate')
        for metric in ('accuracy','macro_f1','nll','brier','vqa_accuracy'):
            vals=[r[metric] for r in rr if r[metric] is not None]
            if vals:item[metric]={'mean':float(np.mean(vals)),'sample_sd':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
        summary.append(item)
    atomic(out/'results.json',result)
    atomic(out/'summary.json',{'rows':summary,'uncertainty':'support-seed SD is descriptive, not query/patient CI'})
    if flat:
        with (out/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    return {'status':'complete','case':plan['case'],'units':len(plan['units']),'output':str(out)}


def analyze(source_plan,out,units=None):
    """All domains, all matched seeds; examples selected by ID hash, not gain size."""
    report={}
    for u in (source_plan['units'] if units is None else units):
        folder=Path(u['output']);names=expected(u)
        verify(folder/'selection_receipt',{'unit':u['key'],'plan':digest(source_plan)})
        selection=read(folder/'selection.json')
        if selection!=read(folder/'selection_receipt'/'selection.json'):raise ValueError('Selection changed')
        for name in names:
            fit=folder/'fits'/name;verify(fit)
            bound=verify(folder/'evaluation'/name)['binding']
            if selection['fit_seals'][name]!=sha(fit/'DONE.json') or bound['fit_seal']!=sha(fit/'DONE.json'):
                raise ValueError('Analysis fit/evaluation mismatch')
        if 'aperture' not in names:raise ValueError('E08 expects original E01 APERTURE')
        right=rows(folder/'evaluation/aperture/predictions.jsonl')
        comparisons={}
        for baseline in ('frozen','lora1','lora4','random'):
            left=rows(folder/'evaluation'/baseline/'predictions.jsonl')
            comparisons[baseline]={'paired_accuracy':paired_accuracy(left,right),
                                   'baseline_metrics':metrics(left),'aperture_metrics':metrics(right),
                                   'examples':error_examples(left,right)}
        report[u['key']]=comparisons
    atomic(Path(out)/'domain_analysis.json',{'units':report,'causal_claim':False,
           'example_selection':'fixed hash order within four outcome categories','private_ids':True})
    return {'units':len(report),'output':str(out)}
