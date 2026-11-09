#!/usr/bin/env python3
"""Read-only diagnosis of sealed E01/E02 fits referenced by an E10 unit.

No model loading, GPU forward, query scoring, tuning, or modification of old
artifacts. Run this standalone script from any checkout; --source-repo refers
to the ORIGINAL running checkout. Reports contain aggregate class/group counts,
not medical images, sample IDs, patient IDs, or per-example predictions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_fit(source):
    folder=Path(source['folder']).resolve();seal=folder/'DONE.json'
    if sha(seal)!=source['seal']:raise ValueError('Source fit seal changed')
    record=read(seal)
    if record['binding']!=source['binding'] or not record.get('files'):
        raise ValueError('Source fit identity changed or empty seal')
    for name,expected in record['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha(path)!=expected:
            raise ValueError('Source fit artifact changed: '+name)
    return folder


def checked_rows(path,manifest):
    data=rows(path)
    if len(data)!=len(manifest):raise ValueError('Incomplete support/calibration predictions')
    if len({r['sample_id'] for r in data})!=len(data):raise ValueError('Duplicate prediction IDs')
    for p,r in zip(data,manifest):
        if p['sample_id']!=r['sample_id'] or p['label_id']!=r['label_id']:
            raise ValueError('Support/calibration population or labels changed')
        if p.get('group_id',r['group_id'])!=r['group_id']:raise ValueError('Group identity changed')
        scores=p.get('mean_log_scores')
        if not isinstance(scores,list) or len(scores)!=len(r['candidate_labels']):
            raise ValueError('Prediction candidate count mismatch')
        if not scores or not all(math.isfinite(x) for x in scores):raise ValueError('Nonfinite scores')
        if type(r['label_id']) is not int or not 0<=r['label_id']<len(scores):raise ValueError('Invalid class index')
        if r['candidate_labels'][r['label_id']]!=r['label']:raise ValueError('Manifest class mapping mismatch')
        if p.get('probabilities') is not None:
            offset=max(scores);weights=[math.exp(x-offset) for x in scores];total=sum(weights)
            values=p['probabilities']
            if len(values)!=len(scores) or any(not math.isfinite(x) for x in values):raise ValueError('Invalid probabilities')
            if any(abs(x-w/total)>1e-7+1e-7*abs(w/total) for x,w in zip(values,weights)):
                raise ValueError('Probabilities disagree with raw scores')
    return data


def class_summary(data):
    if not data:raise ValueError('Empty support/calibration predictions')
    classes=max(len(r['mean_log_scores']) for r in data)
    matrix=[[0]*classes for _ in range(classes)]
    nll=[];absolute=[];margins=[];conf=[]
    for r in data:
        s=r['mean_log_scores'];y=r['label_id'];prediction=max(range(len(s)),key=s.__getitem__)
        matrix[y][prediction]+=1
        mx=max(s);logsum=mx+math.log(sum(math.exp(x-mx) for x in s))
        nll.append(logsum-s[y]);absolute.append(-s[y])
        margins.append(s[y]-max(x for i,x in enumerate(s) if i!=y))
        conf.append(math.exp(s[prediction]-logsum))
    support_counts=[sum(row) for row in matrix]
    predicted=[sum(row[c] for row in matrix) for c in range(classes)]
    f1=[2*matrix[c][c]/(support_counts[c]+predicted[c]) if support_counts[c]+predicted[c] else 0.
        for c in range(classes)]
    return {'count':len(data),'accuracy':sum(matrix[i][i] for i in range(classes))/len(data),
        'macro_f1':sum(f1)/classes,'candidate_nll':sum(nll)/len(nll),
        'correct_answer_sequence_nll':sum(absolute)/len(absolute),
        'mean_correct_candidate_margin':sum(margins)/len(margins),'mean_confidence':sum(conf)/len(conf),
        'class_counts':support_counts,'prediction_counts':predicted,'confusion_matrix':matrix,
        'class_recall':[matrix[c][c]/n if n else None for c,n in enumerate(support_counts)],
        'scoring':'raw sequence-sum candidate scores, T=1; not training-history loss'}


def transitions(base,adapted):
    if [(r['sample_id'],r['label_id']) for r in base]!=[(r['sample_id'],r['label_id']) for r in adapted]:
        raise ValueError('Cross-method prediction population mismatch')
    lost=fixed=old=disagree=0
    for a,b in zip(base,adapted):
        p=max(range(len(a['mean_log_scores'])),key=a['mean_log_scores'].__getitem__)
        q=max(range(len(b['mean_log_scores'])),key=b['mean_log_scores'].__getitem__)
        ok=p==a['label_id'];new=q==b['label_id']
        old+=ok;lost+=ok and not new;fixed+=not ok and new;disagree+=p!=q
    return {'original_correct':old,'correct_to_wrong':lost,'wrong_to_correct':fixed,
            'lost_original_correct_fraction':lost/old if old else None,
            'prediction_disagreement':disagree/len(base)}


def controller_summary(path):
    import torch  # optional, CPU and weights_only; never load the backbone
    payload=torch.load(path,weights_only=True,map_location='cpu')
    result={};alpha=float(payload['alpha'])
    for key,raw in payload['raw'].items():
        raw=raw.detach().double();norm=float(torch.linalg.vector_norm(raw))
        if raw.ndim!=2:raise ValueError('This audit only interprets original Full controllers')
        a=alpha*raw/(1+norm);u=payload['bases'][key].detach().double()
        result[key]={'rank':raw.shape[0],'basis_shape':list(u.shape),'raw_frobenius':norm,
            'operator_frobenius':float(torch.linalg.matrix_norm(a)),
            'operator_spectral':float(torch.linalg.matrix_norm(a,ord=2)),
            'basis_orthogonality_max_error':float((u.T@u-torch.eye(u.shape[1],dtype=u.dtype)).abs().max())}
    return {'alpha':alpha,'sites':result,'note':'operator norms, not an output-KL or semantic guarantee'}


def split_summary(manifests):
    summary={};sets={}
    for role,rs in manifests.items():
        if len({r['sample_id'] for r in rs})!=len(rs):raise ValueError('Duplicate manifest IDs')
        summary[role]={'count':len(rs),'group_count':len({r['group_id'] for r in rs})}
        sets[role]={key:{r[key] for r in rs if r.get(key)} for key in ('sample_id','group_id','content_hash')}
    overlaps={}
    for a,b in [('support','calibration'),('support','query'),('calibration','query')]:
        overlaps[a+'__'+b]={key:len(sets[a][key]&sets[b][key]) for key in sets[a]}
    return {'roles':summary,'overlap_counts':overlaps,
            'note':'fit/calibration group overlap may be declared in some historical non-medical protocols; inspect the frozen protocol'}


def audit_retention(unit_dir,source_repo=None,inspect_controller=False):
    unit_dir=Path(unit_dir).resolve()
    e10plan=read(unit_dir.parent.parent/'plan.json')
    e10binding=read(unit_dir/'binding.json')
    repo=Path(source_repo or e10plan['repo']).resolve()
    plans={};snapshots=[];protected=[str(Path(e10plan['settings']['run_root']).resolve())]
    for item in e10plan['source_plans']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen source plan changed')
        plan=read(item['path']);plans[digest(plan)]=plan
        differences=[]
        for name,expected in plan['source_files'].items():
            path=(repo/name).resolve()
            if not path.is_relative_to(repo) or not path.is_file() or sha(path)!=expected:differences.append(name)
        snapshots.append({'case':plan['case'],'plan_digest':digest(plan),'matching':not differences,
            'mismatches':differences,'engine_sha256_expected':plan['source_files'].get('src/aperture_rebuttal_cases/engine.py')})
        protected.append(str(Path(plan['settings']['run_root']).resolve()))
    report={'read_only':True,'query_metrics_read':False,'new_training':False,
        'e10_complete':(unit_dir/'DONE.json').exists(),'source_snapshots':snapshots,
        'domains':{},'protected_roots':sorted(set(protected)),
        'interpretation':'Support/calibration diagnosis only. No automatic tuning, rerunning, or attribution of query regressions.'}
    for domain,sources in e10binding['source_fits'].items():
        group={'fits':{},'transitions':{}};cache={}
        for name,source in sources.items():
            if source['source_plan_digest'] not in plans:raise ValueError('Unknown sealed source plan')
            folder=verify_fit(source);binding=source['binding'];unit=binding['unit'];variant=binding['variant']
            manifests={}
            for role in ('support','calibration','query'):
                if sha(unit[role])!=binding['manifests'][role]:raise ValueError('Source manifest changed')
                manifests[role]=rows(unit[role])
            stats=read(folder/'fit.json');deployment=read(folder/'deployment.json')
            if stats.get('query_forwards')!=0:raise ValueError('Fit does not establish zero query forwards')
            settings={k:variant.get(k,unit.get(k)) for k in
                ('engine','rank','alpha','lr','steps','l2','layers','mask','loss_reduction','objective','lora_rank','lora_lr','lora_passes')}
            count=len(manifests['support']);penalty=settings['l2']
            effective=(penalty/count if settings['loss_reduction']=='sum' else penalty) if penalty is not None and settings['engine']=='aperture' else None
            item={'seal_verified':True,'settings':settings,'split':split_summary(manifests),
                'fit_stats':{'recorded_batch_size':stats.get('batch_size'),
                    'frozen_batch_size':unit.get('execution_batch_size'),
                    'batch_bound_in_plan':unit.get('execution_batch_size') is not None,
                    'optimized_scalars':stats.get('optimized_scalars'),
                    'optimizer_updates':stats.get('optimizer_updates'),
                    'fit_seconds_including_extraction':stats.get('fit_seconds_including_extraction'),
                    'history':stats.get('history',[])},
                'mean_objective_relative_l2':effective,
                'regularization_note':'Algebraic loss ratio, NOT an Adam step-size estimate',
                'temperature_saved_but_not_applied':deployment.get('temperature')}
            cache[name]={}
            for role,filename in [('support','support_final.jsonl'),('calibration','calibration.jsonl')]:
                data=checked_rows(folder/filename,manifests[role]);cache[name][role]=data;item[role]=class_summary(data)
            model_cfg=Path(unit['model_path'])/'config.json'
            if model_cfg.exists():
                cfg=read(model_cfg);text=cfg.get('text_config',cfg.get('llm_config',cfg));types=text.get('layer_types',[])
                item['decoder_layout']={'depth':text.get('num_hidden_layers'),
                    'selected':[{'index':i,'type':types[i] if i<len(types) else 'not_recorded'} for i in unit['layers']]}
            if inspect_controller and (folder/'controller.pt').exists():item['controller']=controller_summary(folder/'controller.pt')
            # Detect a concurrent mutation during the read; never repair artifacts.
            verify_fit(source)
            group['fits'][name]=item
        if 'frozen' in cache and 'aperture' in cache:
            group['transitions']={r:transitions(cache['frozen'][r],cache['aperture'][r]) for r in ('support','calibration')}
        report['domains'][domain]=group
    return report


def save_report(report,path):
    path=Path(path).expanduser().resolve()
    if any(path==Path(root) or path.is_relative_to(root) for root in report['protected_roots']):
        raise ValueError('Report must be outside sealed training and retention roots')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retention-unit',required=True,type=Path)
    parser.add_argument('--source-repo',type=Path)
    parser.add_argument('--inspect-controller',action='store_true')
    parser.add_argument('--output',type=Path,help='New file OUTSIDE the sealed run roots; default stdout')
    args=parser.parse_args(argv)
    try:
        result=audit_retention(args.retention_unit,args.source_repo,args.inspect_controller)
        if args.output:save_report(result,args.output)
        else:print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return 0
    except (OSError,ValueError,KeyError,ImportError) as exc:
        parser.exit(2,'Read-only audit stopped: '+str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
