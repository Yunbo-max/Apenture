"""Predeclared cases, reviewer coverage and independent comparison units."""
from copy import deepcopy
from pathlib import Path
from .store import atomic, digest, read, sources

MODELS = ('q25', 'q34', 'g34', 'internvl3', 'phi')
FIXED = dict(rank=16, layers=[14,27], alpha=3., steps=4, lr=.05, l2=.001,
             objective='token_mean', optimization_seed=0, basis_seed=0,
             weight_dtype='bfloat16', sites=['K','V'], mode='full', basis_mode='covariance')
MODEL_INFO = {
 'q25': ('Qwen/Qwen2.5-VL-7B-Instruct','qwen2_5_vl',448),
 'q34': ('Qwen/Qwen3-VL-4B-Instruct','qwen3_vl',448),
 'g34': ('google/gemma-3-4b-it','gemma3',896),
 'internvl3': ('OpenGVLab/InternVL3-8B-Instruct','internvl_chat',224),
 'phi': ('microsoft/Phi-3.5-vision-instruct','phi3_v',448)}
PANEL = ('bright-hawaii-wildfire','bright-libya-flood','camelyon17-hospital_2')
CASES = {
 'E01': {'title':'Five-model complete comparison','kind':'classification','models':list(MODELS),'seeds':[0,1,2],'datasets':'all','evaluate':'all'},
 'E02': {'title':'Stronger baselines and equal candidate-count tuning','kind':'classification','models':['q25','g34'],'seeds':[0,1,2],'datasets':list(PANEL),'evaluate':'selected'},
 'E03': {'title':'Nested 16/32/64-label hospital curves','kind':'support','models':['q25','g34'],'seeds':[0,1,2],'datasets':['hospital_0','hospital_2'],'budgets':[16,32,64],'evaluate':'selected'},
 'E04': {'title':'Basis, mixing, bound and K/V ablations','kind':'classification','models':['q25'],'seeds':[0,1,2],'datasets':list(PANEL),'evaluate':'all'},
 'E05': {'title':'Original-method alpha/LR sensitivity','kind':'classification','models':['q25','g34'],'seeds':[0,1,2],'datasets':list(PANEL),'evaluate':'all'},
 'E06': {'title':'Storage, fitting cost and steady-state latency','kind':'classification','models':['q25'],'seeds':[0],'datasets':['bright-hawaii-wildfire','camelyon17-hospital_2'],'evaluate':'all'},
 'E07': {'title':'TextVQA free short-answer generation','kind':'vqa','models':['q25','g34'],'seeds':[0,1,2],'datasets':['textvqa'],'evaluate':'selected'},
 'E08': {'title':'Paired domain/class/error analysis','kind':'analysis','models':['q25'],'seeds':[0,1,2],'datasets':['bright-hawaii-wildfire','bright-libya-flood','bright-noto-earthquake','bright-turkey-earthquake'],'evaluate':'none'},
 'E09': {'title':'Method/site/mask/restore evidence and writing checklist','kind':'audit','models':list(MODELS),'seeds':[0],'datasets':list(PANEL),'evaluate':'none'},
}
REVIEW_MAP = {
 'R1.1': dict(cases=['E01','E06'],writing='Align contribution with storage/performance trade-off; no universal superiority.',limit='Empirical gains remain unmeasured in this delivery.'),
 'R1.2': dict(cases=['E01','E08'],writing='Separate APERTURE-minus-Frozen from APERTURE-minus-LoRA; retain regressions.',limit='Performance is not presumed to improve.'),
 'R1.3': dict(cases=['E02','E04'],writing='Explain contribution relative to representation adaptation; sensitivity is not semantics.',limit='More experiments alone do not establish novelty.'),
 'R1.4': dict(cases=['E08','E04'],writing='Discuss all four BRIGHT events and deterministic positive/negative examples.',limit='Qualitative associations are not causal explanations.'),
 'R2.1': dict(cases=['E02'],writing='Distinguish approaches, training budgets and ablations; disclose VLM baseline placements.',limit='Support/data and tuning candidates match, not exact GPU compute.'),
 'R2.2': dict(cases=['E03','E02'],writing='Report nested larger support and longer/tuned LoRA, including all label access.',limit='New hospital curve is a separate protocol, not replacement historical results.'),
 'R2.3': dict(cases=['E07'],writing='Add short-answer free VQA; explicitly narrow claims on reasoning/dialogue.',limit='Complex reasoning and multi-turn dialogue are not established by TextVQA.'),
 'R2.4': dict(cases=['E06'],writing='Acknowledge fitting may be slower; include extraction and complete state bytes.',limit='No speed advantage is assumed.'),
 'R3.1': dict(cases=['E01','E03'],writing='Define supervised few-shot target-domain adaptation, not novel-class meta-learning.',limit='This is primarily a problem-definition correction.'),
 'R3.2': dict(cases=['E09'],writing='Specify decoder layer indices, projection outputs, masks, dimensions and fused-QKV routing.',limit='Audits verify implementation, not scientific performance.'),
 'R3.3': dict(cases=['E09'],writing='Define x=(image,question), answer y, support N, image-token mask M; add question to overview.',limit='Requires manuscript/figure editing, not just a new result table.'),
 'R3.4': dict(cases=['E04','E09'],writing='Explain signed supervised controller fitting after gradient-second-moment basis estimation.',limit='Neither sensitive directions nor these ablations prove causal semantic discovery.'),
 'R3.5': dict(cases=['E09'],writing='Add a support->basis->controller->fresh-query overview and notation table.',limit='Writing checklist is not an automatically revised manuscript.'),
}


def variants(case, kind):
    lr = 1e-4 if kind=='paired' else 2e-4
    out=[]
    def add(name, family=None, engine='aperture', **extra):
        out.append({**deepcopy(FIXED),'name':name,'family':family or name,'engine':engine,
                    'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,
                    'lora_lr':lr,'lora_passes':4,'lora_accumulation':3 if kind=='paired' else 1, **extra})
    if case in ('E08','E09'): return []
    add('frozen',engine='frozen')
    if case in ('E01','E06'):
        add('lora1',engine='lora',lora_passes=1);add('lora4',engine='lora')
        if case=='E01':add('random',basis_mode='random')
        if case=='E06':add('lora16',engine='lora',lora_passes=16)
        add('aperture')
    elif case in ('E02','E03'):
        families=[('lora-r4','lora',4),('lora-r16','lora',16)]
        if case=='E02':families=[('lora-r1','lora',1),*families,('dora-r4','dora',4),('loreft-r4','loreft',4)]
        for family,engine,rank in families:
            for rate in ((1e-4,1e-3) if engine=='loreft' else (lr/4,lr)):
                for passes in (4,16):
                    add(f'{family}-lr{rate:g}-p{passes}',family,engine,lora_rank=rank,lora_alpha=2*rank,
                        lora_lr=rate,lora_passes=passes)
        for alpha in (1.,3.):
            for rate in (.01,.05):add(f'aperture-a{alpha:g}-lr{rate:g}','aperture',alpha=alpha,lr=rate)
    elif case=='E04':
        add('aperture');add('random',basis_mode='random');add('pca',basis_mode='activation_pca')
        add('diagonal-matched',mode='diagonal_matched');add('unbounded',mode='unbounded')
        add('K-only',sites=['K']);add('V-only',sites=['V'])
    elif case=='E05':
        for alpha in (1.,3.):
            for rate in (.01,.05):add(f'aperture-a{alpha:g}-lr{rate:g}','aperture',alpha=alpha,lr=rate)
    elif case=='E07':
        add('lora1','lora',engine='lora',lora_passes=1);add('lora4','lora',engine='lora');add('aperture')
    return out


def build_units(case, base, root, repo, prepared=None):
    if case not in CASES: raise ValueError('Unknown case')
    root,repo=Path(root).resolve(),Path(repo).resolve(); spec=CASES[case]
    keys=[d['key'] for d in base['datasets']]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate dataset keys')
    import re
    if any(not re.fullmatch(r'[a-zA-Z0-9_-]+',k) for k in keys):raise ValueError('Unsafe dataset key')
    prepared=Path(prepared or root.parent/'prepared').resolve()
    models={m['key']:m for m in base['models']}
    if any(m not in models for m in spec['models']): raise ValueError('Missing declared model; no substitution')
    if spec['kind']=='analysis': return []
    if spec['kind']=='support':
        datasets=[{'key':h,'kind':'single','classes':2,'query_count':500,'budget':n}
                  for h in spec['datasets'] for n in spec['budgets']]
    elif spec['kind']=='vqa':datasets=[{'key':'textvqa','kind':'single','classes':1,'query_count':5000,'budget':64}]
    else:
        datasets=[d for d in base['datasets'] if spec['datasets']=='all' or d['key'] in spec['datasets']]
    units=[]
    for model in spec['models']:
        ident,family,size=MODEL_INFO[model]
        for d in datasets:
            for seed in spec['seeds']:
                budget=d.get('budget',8*d['classes']); key=f"{model}--{d['key']}--n{budget}--s{seed}"
                if spec['kind']=='support':
                    path=prepared/'support'/d['key']/f'support_{budget}'/f'seed_{seed}'
                    roles={r:str(path/f'{"fit" if r=="support" else r}.jsonl') for r in ('support','calibration','query')}
                elif spec['kind']=='vqa':
                    path=prepared/'textvqa'/f'seed_{seed}'
                    roles={r:str(path/f'{"fit" if r=="support" else r}.jsonl') for r in ('support','calibration','query')}
                else:
                    path=repo/base['prepared_root']/d['key']
                    roles={'support':str(path/f'seed_{seed}'/'fit.jsonl'),
                           'calibration':str(path/f'seed_{seed}'/'calibration.jsonl'), 'query':str(path/'query.jsonl')}
                units.append({**deepcopy(FIXED),**roles,'case':case,'key':key,'model_key':model,
                    'model_id':ident,'family':family,'model_path':str((repo/models[model]['path']).resolve()),
                    'domain':d['key'],'domain_key':d['key'],'kind':d['kind'],'classes':d['classes'],
                    'query_count':d.get('query_count'),'support_seed':seed,'total_labels':budget,
                    'fit_count':budget*3//4,'cal_count':budget//4,'image_size':size,
                    'loss_reduction':'mean' if d['kind']=='paired' else 'sum',
                    'mask':'post' if d['kind']=='paired' else 'all_visual',
                    'group_disjoint':spec['kind'] in ('support','vqa') or bool(d.get('fit_cal_group_disjoint',False)),
                    'output':str(root/case/'units'/key),'variants':variants(case,d['kind']),
                    'evaluate_policy':spec['evaluate'],'task_type':'vqa' if spec['kind']=='vqa' else 'classification'})
    return units


def freeze(case, cfg, repo):
    repo=Path(repo).resolve(); base=read(repo/cfg['base_config'])
    root=(repo/cfg['run_root']).resolve(); prepared=(repo/cfg['prepared_root']).resolve()
    spec=CASES[case]
    if spec['kind'] in ('classification','audit') and spec['datasets']!='all':
        if not set(spec['datasets']) <= {d['key'] for d in base['datasets']}:raise ValueError('Declared panel task missing; no silent shrinkage')
    for protected in ('src','scripts','configs','data','paper','reports','results','.git'):
        if root==repo or root in repo.parents or root.is_relative_to(repo/protected) or (repo/protected).is_relative_to(root):
            raise ValueError('Output root overlaps protected repository content')
    old=(repo/base['run_root']).resolve()
    if root==old or root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('Output overlaps historical run root')
    plan={'version':'aperture-reviewer-cases-v1','case':case,'settings':cfg,'base_config':base,
          'source_files':sources(repo),'units':build_units(case,base,root,repo,prepared),
          'review_map':REVIEW_MAP,'method_changed':False,'state':'planned_not_run'}
    folder=root/case; marker=folder/'plan.json'
    if folder.exists() and any(folder.iterdir()) and not marker.exists():raise ValueError('Nonempty unbound case root')
    atomic(marker,plan,immutable=True)
    return plan
