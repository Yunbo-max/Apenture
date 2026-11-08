"""Complete-scope reports; never omit failed identities or fill missing cells."""
from pathlib import Path
import csv
import statistics
from collections import defaultdict
from aperture_rebuttal_cases import store


def coverage(plan):
    missing=[];invalid=[];complete=[]
    root=Path(plan['settings']['run_root'])/plan['part']/'units'
    for j in plan['jobs']:
        p=root/j['key']
        if not (p/'DONE.json').exists():missing.append(j['key']);continue
        try:
            receipt=store.verify(p)
            if receipt['binding'].get('plan')!=store.digest(plan):raise ValueError('Result is from another plan')
        except (ValueError,OSError,KeyError) as e:invalid.append({'unit':j['key'],'reason':str(e)})
        else:complete.append(j['key'])
    return {'complete':not missing and not invalid,'completed':len(complete),'planned':len(plan['jobs']),'missing':missing,'invalid':invalid}


def flatten_visual(unit,comparisons,new_domain):
    rows=[]
    for target,arms in comparisons.items():
        for method,values in arms.items():
            rows.append({'unit':unit,'target':target,'role':'new_domain' if target==new_domain else 'retention',
                         'method':method,**values})
    return rows


def aggregate(plan):
    root=Path(plan['settings']['run_root'])/plan['part'];out=root/'report';out.mkdir(parents=True,exist_ok=True)
    for n in ('results.json','results.csv','summary.json'):(out/n).unlink(missing_ok=True)
    status=coverage(plan);store.atomic(out/'coverage.json',status)
    if not status['complete']:raise ValueError('Incomplete/invalid E10 population; no scientific aggregate')
    # Recheck original model/fit/selection identities as well as E10 outputs.
    from .worker import check_job,compact_binding
    rows=[];identities=[];groups=defaultdict(list)
    for j in plan['jobs']:
        sources=check_job(plan,j);folder=root/'units'/j['key']
        store.verify(folder,compact_binding(plan,j,sources));r=store.read(folder/'result.json')
        if plan['part']=='bank':
            for x in r['retrievals']:
                row={'unit':j['key'],'model':j['model'],'seed':j['seed'],'stage':j['stage'],**x,
                     'cumulative_adaptation_bytes':r['cumulative_adaptation_bytes'][x['method']]}
                rows.append(row)
        else:
            for row in flatten_visual(j['key'],r['comparisons'],j['domain']):
                row.update(model=j['model'],seed=j['seed'],source_domain=j['domain'],
                           state_bytes=r['state_bytes'].get(row['method'],0))
                rows.append(row)
                groups[(j['model'],j['domain'],row['target'],row['method'])].append(row)
        identities.append({'unit':j['key'],'passed':r['all_identity_checks_passed']})
    summary=[]
    for key,rr in groups.items():
        if len(rr)!=3 or {r['seed'] for r in rr}!={0,1,2}:raise ValueError('Incomplete support seed population')
        item=dict(zip(('model','source_domain','target','method'),key));item['support_seeds']=3
        for metric in ('accuracy_delta','macro_f1_delta','nll_change','retention_drop','lost_original_correct_fraction'):
            vals=[r[metric] for r in rr if r[metric] is not None]
            item[metric]={'mean':statistics.mean(vals) if vals else None,'sample_sd':statistics.stdev(vals) if len(vals)>1 else None,
                          'n_defined':len(vals),'uncertainty_unit':'support_seed_not_query_count'}
        summary.append(item)
    store.atomic(out/'results.json',{'complete':True,'rows':rows,'identity_checks':identities,
         'all_identity_checks_passed':all(i['passed'] for i in identities),'claim':'No universal no-forgetting claim; bank has oracle task identity'})
    store.atomic(out/'summary.json',{'seed_summary':summary,'part':plan['part'],
        'notes':['Query-group intervals are conditional on support seed.','No non-inferiority test or no-forgetting guarantee is inferred from nonsignificance.','Compare new-domain gain and retention jointly.']})
    flat=[]
    for row in rows:flat.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
    fields=list(dict.fromkeys(k for r in flat for k in r))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    return {'complete':True,'rows':len(rows),'output':str(out),'all_identity_checks_passed':all(i['passed'] for i in identities)}
