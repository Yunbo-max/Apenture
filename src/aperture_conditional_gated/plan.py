"""A new, versioned CG registry. Legacy E plans and results are never relabeled."""
from copy import deepcopy
from pathlib import Path
import os
from aperture_rebuttal_cases import plan as legacy
from aperture_rebuttal_cases import store

VERSION='aperture-conditional-gated-v1'
BASE_COMMIT='43cb0ebb2d1dc4d29bb1ab5fb8eba75d4d47dd88'
MAP={f'CG{i:02}':f'E{i:02}' for i in range(1,8)}
MAP.update(CG00='E09',CG08='E02',CG09='E08')
CASES={k:deepcopy(legacy.CASES[v]) for k,v in MAP.items()}
CASES['CG00']['title']='Native support-only conditional-gate audit'
CASES['CG01']['title']='Five-model gated primary, static Full and LoRA controls'
CASES['CG02']['title']='Matched six-candidate selection: gated/static/LoRA/DoRA'
CASES['CG04']['title']='Gate, amplitude, basis and K/V mechanism controls'
CASES['CG05']['title']='Alpha 1/2/3 x LR .01/.05, static and gated'
CASES['CG06']['title']='Complete gated state and total adaptation costs'
CASES['CG08'].update(title='Independent 1/2/4/8/16-support-pass budget diagnostics',evaluate='all')
CASES['CG09']['title']='Read-only paired error analysis of CG01 BRIGHT panel'
REVIEW_MAP={
 'qmC7.framing':{'cases':['CG01','CG06'],'writing':'Report gated revision separately; measured performance/storage trade-off, no universal superiority.'},
 'qmC7.mixed_results':{'cases':['CG01','CG05','CG09'],'writing':'All domains/seeds including regressions; gains versus Frozen are not gains versus LoRA.'},
 'qmC7.novelty':{'cases':['CG02','CG04'],'writing':'Explain conditional gate versus static Full, fixed .5 and learned scalar; no semantic/causal discovery claim.'},
 'qmC7.qualitative':{'cases':['CG09'],'writing':'Fixed-hash positive AND negative cases; associations are not causal explanations.'},
 'cJAm.baselines':{'cases':['CG02'],'writing':'LoRA rank1/4/16 and DoRA with equal six-candidate counts. Historical frozen-identical LoReFT is not accepted as valid evidence.'},
 'cJAm.support_size':{'cases':['CG03','CG08'],'writing':'16/32/64 total labels remain separate. Support passes are not equal optimizer steps or compute.'},
 'cJAm.open_ended':{'cases':['CG07'],'writing':'TextVQA free generation without answer choices. Complex reasoning/dialogue remains untested.'},
 'cJAm.cost':{'cases':['CG06'],'writing':'Include U, gates, extraction, fits, search costs and merged/unmerged LoRA timing if valid; no speed claim assumed.'},
 'SXxo.problem':{'cases':['CG01','CG03'],'writing':'Define supervised few-shot target-domain adaptation rather than novel-class episodic meta-learning.'},
 'SXxo.KV_sites':{'cases':['CG00'],'writing':'Specify projection outputs, decoder layers, widths, masks and placement before model-native operations.'},
 'SXxo.symbols':{'cases':['CG00'],'writing':'x=(image,question), y=answer, N=fit examples, m=visual token mask; include question in overview.'},
 'SXxo.basis_gate':{'cases':['CG00','CG04'],'writing':'Support gradients estimate U; answer loss trains R,w,b. Gate uses current visual state, no query label.'},
 'SXxo.overview':{'cases':['CG00'],'writing':'Add fit -> basis -> gated-controller -> fresh-query overview. Code is not a rewritten manuscript.'},
 'additional.overfitting':{'cases':['CG08'],'writing':'Compare final fit/calibration classification at each budget; no guaranteed anti-overfitting or no-forgetting claim.'}}


def variants(case,kind):
    if case not in CASES:raise ValueError('Unknown gated case')
    if case in ('CG00','CG09'):return []
    out=[];lr=1e-4 if kind=='paired' else 2e-4
    def add(name,family=None,engine='conditional_gated',**kw):
        out.append({**deepcopy(legacy.FIXED),'name':name,'family':family or name,'engine':engine,
                    'gate_mode':'conditional','gate_eps':1e-6,'gate_init':.5,
                    'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,'lora_lr':lr,
                    'lora_passes':4,'lora_accumulation':3 if kind=='paired' else 1,**kw})
    add('frozen',engine='frozen')
    if case=='CG01':
        add('static-full',engine='aperture');add('conditional-gated')
        add('random-gated',basis_mode='random');add('lora1',engine='lora',lora_passes=1);add('lora4',engine='lora')
    elif case in ('CG02','CG03'):
        ranks=[1,4,16] if case=='CG02' else [4,16]
        families=[(f'lora-r{r}','lora',r) for r in ranks]
        if case=='CG02':families.append(('dora-r4','dora',4))
        for family,engine,rank in families:
            for rate in (lr/4,lr,lr*4):
                for passes in (4,16):
                    add(f'{family}-lr{rate:g}-p{passes}',family,engine,lora_rank=rank,lora_alpha=2*rank,lora_lr=rate,lora_passes=passes)
        for family,engine in [('static-full','aperture'),('conditional-gated','conditional_gated')]:
            for alpha in (1.,2.,3.):
                for rate in (.01,.05):
                    add(f'{family}-a{alpha:g}-lr{rate:g}',family,engine,alpha=alpha,lr=rate)
    elif case=='CG04':
        add('static-full',engine='aperture');add('conditional-gated')
        add('fixed-half',gate_mode='half');add('learned-scalar',gate_mode='scalar')
        add('random-gated',basis_mode='random');add('pca-gated',basis_mode='activation_pca')
        add('K-gated',sites=['K']);add('V-gated',sites=['V'])
    elif case=='CG05':
        for family,engine in [('static-full','aperture'),('conditional-gated','conditional_gated')]:
            for alpha in (1.,2.,3.):
                for rate in (.01,.05):add(f'{family}-a{alpha:g}-lr{rate:g}',family,engine,alpha=alpha,lr=rate)
    elif case=='CG06':
        add('static-full',engine='aperture');add('conditional-gated')
        for passes in (1,4,16):add(f'lora{passes}',engine='lora',lora_passes=passes)
    elif case=='CG07':
        add('static-full',engine='aperture');add('conditional-gated')
        for passes in (1,4):add(f'lora{passes}','lora',engine='lora',lora_passes=passes)
    elif case=='CG08':
        for passes in (1,2,4,8,16):
            add(f'static-full-p{passes}','static-full','aperture',steps=passes)
            add(f'conditional-gated-p{passes}','conditional-gated',steps=passes)
            add(f'lora-r16-p{passes}','lora-r16','lora',lora_passes=passes)
    return out


def source_files(repo):
    repo=Path(repo);result=store.sources(repo)
    p=repo/'scripts/run_aperture_gated.py'
    if p.is_file():result[str(p.relative_to(repo))]=store.sha(p)
    return result


def build_units(case,base,root,repo,prepared=None,batch=1):
    if type(batch) is not int or batch<1:raise ValueError('batch must be positive integer')
    if case not in CASES:raise ValueError('Unknown gated case')
    spec=CASES[case]
    if spec['kind'] in ('classification','audit') and spec['datasets']!='all':
        if not set(spec['datasets']) <= {d['key'] for d in base['datasets']}:raise ValueError('Missing declared panel; no silent shrinkage')
    units=legacy.build_units(MAP[case],base,root,repo,prepared)
    for u in units:
        u.update(case=case,output=str(Path(root).resolve()/case/'units'/u['key']),
                 variants=variants(case,u['kind']),evaluate_policy=spec['evaluate'],batch_size=batch,execution_batch_size=batch,
                 method_version=VERSION,measure_latency=case=='CG06',
                 diagnostic_budget_policy='independent fits from original base; not a resumed trajectory' if case=='CG08' else None)
    return units


def freeze(case,cfg,repo):
    repo=Path(repo).resolve();cfg=deepcopy(cfg)
    for field in ('base_config','run_root','prepared_root','medical_data_root','textvqa_train_json','textvqa_val_json','textvqa_images'):
        if field in cfg:cfg[field]=str((repo/cfg[field]).resolve())
    if cfg.get('method_version')!=VERSION:raise ValueError('Explicit CG-v1 method_version required')
    batch=cfg.get('execution_batch',1)
    if type(batch) is not int or batch<1:raise ValueError('Invalid execution_batch')
    if 'APERTURE_BATCH' in os.environ and os.environ['APERTURE_BATCH']!=str(batch):
        raise ValueError('APERTURE_BATCH differs from explicit execution_batch')
    root=Path(cfg['run_root']);base=store.read(cfg['base_config'])
    for protected in ('src','scripts','configs','data','paper','reports','results','.git'):
        if root==repo or root in repo.parents or root.is_relative_to(repo/protected) or (repo/protected).is_relative_to(root):
            raise ValueError('Output overlaps protected source/data')
    old=(repo/base['run_root']).resolve()
    if root==old or root.is_relative_to(old) or old.is_relative_to(root):raise ValueError('Output overlaps legacy run root')
    if 'conditional_gated' not in root.name and 'conditional-gated' not in root.name:
        raise ValueError('Use a new root name containing conditional_gated; never reuse an E run root')
    p={'version':VERSION,'case':case,'settings':cfg,'base_config':base,
       'source_files':source_files(repo),'units':build_units(case,base,root,repo,cfg['prepared_root'],batch),
       'review_map':REVIEW_MAP,'method_changed':True,'base_commit':BASE_COMMIT,
       'prior_best_reproduced':False,'query_status':'previously inspected data; exploratory follow-up unless independently locked',
       'state':'planned_not_run'}
    folder=root/case;marker=folder/'plan.json'
    if folder.exists() and any(folder.iterdir()) and not marker.exists():raise ValueError('Nonempty unbound case root')
    store.atomic(marker,p,immutable=True)
    return p
