"""Fixed E10 scope and source-plan/adapter provenance. Does not modify training."""
from copy import deepcopy
from pathlib import Path
from aperture_rebuttal_cases import store

sha=store.sha
MODELS=('q25','g34')
DOMAINS=('camelyon17-hospital_2','bright-hawaii-wildfire','bright-libya-flood')
BRIGHT=('bright-hawaii-wildfire','bright-libya-flood','bright-noto-earthquake','bright-turkey-earthquake')
METHODS=('frozen','aperture','lora1','lora4','lora-r4')
TARGETS={
 'camelyon17-hospital_2':('camelyon17-hospital_0','camelyon17-hospital_1','camelyon17-hospital_3','camelyon17-hospital_4','pathmnist'),
 'bright-hawaii-wildfire':BRIGHT[1:],
 'bright-libya-flood':(BRIGHT[0],BRIGHT[2],BRIGHT[3]),
}
PARTS=('visual','text','bank')


def job_specs(part):
    if part not in PARTS:raise ValueError('Unknown E10 part')
    jobs=[]
    for model in MODELS:
        for seed in (0,1,2):
            if part=='bank':
                for stage in range(1,5):
                    jobs.append(dict(key=f'{model}--bank--s{seed}--t{stage}',model=model,seed=seed,
                        stage=stage,domain='bank',available_domains=list(BRIGHT[:stage]),methods=['aperture','lora1','lora4']))
            else:
                for domain in DOMAINS:
                    n=16 if domain.startswith('camelyon') else 24
                    jobs.append(dict(key=f'{model}--{domain}--n{n}--s{seed}',model=model,seed=seed,domain=domain,
                                     targets=[domain,*TARGETS[domain]] if part=='visual' else ['arc_challenge','boolq'],methods=list(METHODS)))
    return jobs


def select_jobs(jobs,unit=None,model=None,domain=None,seed=None):
    out=[j for j in jobs if (unit is None or j['key']==unit) and (model is None or j['model']==model)
         and (domain is None or j['domain']==domain) and (seed is None or j['seed']==seed)]
    if not out:raise ValueError('No declared E10 jobs match')
    return out


def verify_source(files,repo):
    repo=Path(repo).resolve()
    for name,h in files.items():
        p=(repo/name).resolve()
        if not p.is_relative_to(repo) or not p.is_file() or sha(p)!=h:raise ValueError(f'Original source snapshot changed: {name}')
    expected={s for s in files if s.startswith('src/') and s.endswith('.py')}
    current={str(p.relative_to(repo)) for p in (repo/'src').rglob('*.py')}
    if current!=expected:raise ValueError('Original source file set changed; use source-compatible checkout')


def safe_root(repo,root,old_roots):
    repo=Path(repo).resolve();root=Path(root).resolve()
    def overlap(a,b):return a==b or a in b.parents or b in a.parents
    if root==repo or root in repo.parents:raise ValueError('Unsafe output root')
    for p in [repo/x for x in ('src','scripts','configs','data','paper','reports','results','.git')]+list(map(Path,old_roots)):
        if overlap(root,p.resolve()):raise ValueError('Output overlaps source/training data or results')


def eval_unit(unit,domain):
    d=deepcopy(unit);paired=domain.startswith('bright-')
    if paired!=(unit['kind']=='paired'):raise ValueError('Cross-modality mask changes are forbidden')
    d.update(domain=domain,domain_key=domain,classes=3 if paired else 9 if domain=='pathmnist' else 2)
    return d


def no_overlap(training,query):
    for key in ('sample_id','group_id','content_hash'):
        a={r[key] for r in training if r.get(key)};b={r[key] for r in query if r.get(key)}
        if a&b:raise ValueError(f'Retention query overlaps adaptation data: {key}')


def own_sources(repo):
    repo=Path(repo)
    ps=[repo/'scripts/run_aperture_retention.py',*(repo/'scripts/aperture_retention').glob('*.py')]
    return {str(p.relative_to(repo)):sha(p) for p in sorted(ps)}


def resolve_config(repo,path):
    cfg=store.read(path);repo=Path(repo)
    if cfg.get('protocol')!='aperture-retention-v1':raise ValueError('Wrong E10 protocol')
    for k in ('run_root','text_root','base_config'):
        cfg[k]=str((repo/cfg[k]).resolve())
    cfg['source_plans']=[str((repo/p).resolve()) for p in cfg['source_plans']]
    if not cfg['source_plans'] or len(set(cfg['source_plans']))!=len(cfg['source_plans']):raise ValueError('Unique source plans required')
    return cfg


def freeze(repo,cfg,part):
    repo=Path(repo).resolve();plans=[];old=[]
    for path in cfg['source_plans']:
        p=store.read(path)
        if p['case'] not in ('E01','E02'):raise ValueError('E10 only reuses declared E01/E02 sources')
        verify_source(p['source_files'],repo)
        plans.append({'path':path,'sha256':sha(path),'case':p['case']})
        old.append(Path(p['settings']['run_root']))
    safe_root(repo,cfg['run_root'],old)
    base=store.read(cfg['base_config'])
    expected={x for vals in TARGETS.values() for x in vals}|set(DOMAINS)
    if not expected<={d['key'] for d in base['datasets']}:raise ValueError('Declared retention task missing from base config')
    missing=[]
    required={(m,d,s,'E01') for m in MODELS for d in (BRIGHT if part=='bank' else DOMAINS) for s in (0,1,2)}
    if part!='bank':required|={(m,d,s,'E02') for m in MODELS for d in DOMAINS for s in (0,1,2)}
    available={(u['model_key'],u['domain'],u['support_seed'],p['case']) for source in plans
               for p in [store.read(source['path'])] for u in p['units']}
    if required-available:raise ValueError(f'Missing declared source plan units: {sorted(required-available)}')
    prepared=(repo/base['prepared_root']).resolve()
    queries={}
    if part in ('visual','bank'):
        for domain in sorted(set(BRIGHT) if part=='bank' else expected):
            p=prepared/domain/'query.jsonl';queries[domain]={'path':str(p),'sha256':sha(p)}
    else:
        from .text import DATASETS
        for name in DATASETS:
            p=Path(cfg['text_root'])/(name+'.jsonl');queries[name]={'path':str(p),'sha256':sha(p)}
        p=Path(cfg['text_root'])/'provenance.json';queries['_provenance']={'path':str(p),'sha256':sha(p)}
    plan={'protocol':cfg['protocol'],'part':part,'settings':cfg,'repo':str(repo),'source_plans':plans,
          'evaluator_sources':own_sources(repo),'base_config_sha256':sha(cfg['base_config']),
          'queries':queries,'jobs':job_specs(part),'training':False,'selection_on_retention':False,
          'claim':'capability retention or explicit-task module retrieval, NOT continual-learning guarantee'}
    folder=Path(cfg['run_root'])/part
    if folder.exists() and any(folder.iterdir()) and not (folder/'plan.json').exists():raise ValueError('Nonempty unbound E10 root')
    store.atomic(folder/'plan.json',plan,immutable=True)
    return plan


def verify_plan(plan):
    if plan['evaluator_sources']!=own_sources(plan['repo']):raise ValueError('E10 evaluator source changed')
    if sha(plan['settings']['base_config'])!=plan['base_config_sha256']:raise ValueError('Base config changed')
    for s in plan['source_plans']:
        if sha(s['path'])!=s['sha256']:raise ValueError('Source plan changed')
        verify_source(store.read(s['path'])['source_files'],plan['repo'])
    for q in plan['queries'].values():
        if sha(q['path'])!=q['sha256']:raise ValueError('Frozen query/provenance changed')


def locate(plan,job,case,domain,required):
    """Exactly one complete unit, never choose by scores or combine partial fits."""
    candidates=[]
    for sp in plan['source_plans']:
        if sp['case']!=case:continue
        p=store.read(sp['path'])
        for u in p['units']:
            if (u['model_key'],u['domain'],u['support_seed'])!=(job['model'],domain,job['seed']):continue
            if all((Path(u['output'])/'fits'/v['name']/'DONE.json').exists() for v in u['variants'] if required(v)):
                candidates.append((p,u))
    if len(candidates)!=1:raise ValueError(f'Need exactly one complete {case}/{job["model"]}/{domain}/s{job["seed"]}; found {len(candidates)}')
    return candidates[0]


def validate_variant(label,variant):
    if label=='aperture':
        required=dict(engine='aperture',rank=16,layers=[14,27],alpha=3.,steps=4,lr=.05,l2=.001,
                      objective='token_mean',sites=['K','V'],mode='full',basis_mode='covariance')
    elif label in ('lora1','lora4'):
        required=dict(engine='lora',lora_rank=16,lora_alpha=32,lora_passes=1 if label=='lora1' else 4)
    elif label=='lora-r4':required=dict(engine='lora',lora_rank=4,lora_alpha=8)
    elif label=='frozen':required=dict(engine='frozen')
    else:raise ValueError('Unknown retention method')
    if any(variant.get(k)!=v for k,v in required.items()):
        raise ValueError('Source variant differs from predeclared method; do not relabel a tuned method')


def resolve_sources(plan,job,domain):
    from aperture_rebuttal_cases.engine import binding
    p,u=locate(plan,job,'E01',domain,lambda v:v['name'] in ('frozen','aperture','lora1','lora4'))
    store.verify(Path(u['output'])/'audit',binding(u,p))
    result={}
    def add(label,sp,su,v):
        validate_variant(label,v)
        if store.read(Path(su['output'])/'audit/audit.json').get('passed') is not True:
            raise ValueError('Source audit not passed')
        folder=Path(su['output'])/'fits'/v['name'];expected=binding(su,sp,v)
        store.verify(folder,expected)
        if store.read(folder/'fit.json').get('query_forwards')!=0:
            raise ValueError('Source fitting does not establish zero query forwards')
        result[label]={'unit':su,'variant':v,'folder':str(folder),'seal':sha(folder/'DONE.json'),
                       'source_plan_digest':store.digest(sp),'binding':expected,'stats':store.read(folder/'fit.json')}
    for v in u['variants']:
        if v['name'] in job['methods'] or v['name']=='frozen':add(v['name'],p,u,v)
    if 'lora-r4' in job['methods']:
        p2,u2=locate(plan,job,'E02',domain,lambda v:v['family']=='lora-r4')
        for key in ('support','calibration','query'):
            if sha(u[key])!=sha(u2[key]):raise ValueError('LoRA-r4 and primary arms use different data')
        if binding(u,p)['model_assets']!=binding(u2,p2)['model_assets']:raise ValueError('Backbones differ across arms')
        store.verify(Path(u2['output'])/'audit',binding(u2,p2))
        selection=store.read(Path(u2['output'])/'selection.json')
        recdir=Path(u2['output'])/'selection_receipt'
        store.verify(recdir,{'unit':u2['key'],'plan':store.digest(p2)})
        if selection!=store.read(recdir/'selection.json') or selection.get('query_metrics_used') is not False:
            raise ValueError('Invalid support-only selection')
        vv=[v for v in u2['variants'] if v['family']=='lora-r4'];ranks=[]
        for v in vv:
            folder=Path(u2['output'])/'fits'/v['name'];store.verify(folder,binding(u2,p2,v))
            if selection['fit_seals'][v['name']]!=sha(folder/'DONE.json'):raise ValueError('Selection fit changed')
            import math
            score=store.read(folder/'fit.json')['calibration_nll']
            if not math.isfinite(score):raise ValueError('Invalid calibration score')
            ranks.append((score,v['name'],v))
        _,name,v=min(ranks,key=lambda x:x[:2])
        if name!=selection['selected']['lora-r4']:raise ValueError('Selection not calibration-optimal under frozen tie rule')
        add('lora-r4',p2,u2,v)
    return result
