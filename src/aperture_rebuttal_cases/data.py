"""Label-budget-preserving preparation and leakage checks, independent of outcomes."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from .store import atomic, digest, rows, sha, write_rows

EVALUATOR_BLOB = '1be4fc7acc8920d6e8224509b808821c209eec02'


def ordered(rs, salt):
    return sorted(rs, key=lambda r: digest([salt,r['sample_id']]))


def nested(fit_pool, calibration_pool, per_class, seed, classes):
    if type(per_class) is not int or per_class < 4 or per_class % 4: raise ValueError('per_class must divide into 3:1 roles')
    result={}
    for role,pool,count in [('support',fit_pool,3*per_class//4),('calibration',calibration_pool,per_class//4)]:
        chosen=[]
        for c in range(classes):
            available=ordered([r for r in pool if r['label_id']==c],f'{role}:{seed}:{c}')
            if len(available)<count:raise ValueError('Insufficient non-query pool; never borrow query examples')
            chosen.extend(deepcopy(available[:count]))
        result[role]=sorted(chosen,key=lambda r:r['sample_id'])
    if {r['sample_id'] for r in result['support']}&{r['sample_id'] for r in result['calibration']}:
        raise ValueError('Fit/calibration overlap')
    return result


def validate(roles, group_disjoint, *, vqa=False):
    sets={}
    for role,rs in roles.items():
        if not rs:raise ValueError(f'Empty {role}')
        ids=[r['sample_id'] for r in rs]
        if len(set(ids))!=len(ids):raise ValueError(f'Duplicate IDs in {role}')
        if any(not r.get('group_id') for r in rs):raise ValueError('Missing group identity')
        inputs=[digest([r.get('content_hash',r.get('image')),r['question'],r.get('candidate_labels')]) for r in rs]
        if not vqa and len(set(inputs))!=len(inputs):raise ValueError('Duplicate image/question/candidate input')
        sets[role]={'ids':set(ids),'groups':{r['group_id'] for r in rs},
                    'pixels':{r.get('content_hash',r.get('image',str((r.get('pre_image'),r.get('post_image'))))) for r in rs},
                    'inputs':set(inputs)}
        if not vqa:
            for r in rs:
                cs=r['candidate_labels'];y=r['label_id']
                if type(y) is not int or not 0<=y<len(cs) or len(cs)!=len(set(cs)) or cs[y]!=r['label']:
                    raise ValueError('Invalid candidate/label contract')
    for a,b in [('support','calibration'),('support','query'),('calibration','query')]:
        for key in ('ids','inputs'):
            if sets[a][key]&sets[b][key]:raise ValueError(f'{key} overlap between {a} and {b}')
        if group_disjoint or b=='query':
            for key in ('groups','pixels'):
                if sets[a][key]&sets[b][key]:raise ValueError(f'{key} leakage between {a} and {b}')


def load_roles(unit, *, images=False):
    data={k:rows(unit[k]) for k in ('support','calibration','query')}
    vqa=unit['task_type']=='vqa'
    validate(data,unit['group_disjoint'],vqa=vqa)
    for role,n in [('support',unit['fit_count']),('calibration',unit['cal_count']),('query',unit['query_count'])]:
        if n is not None and len(data[role])!=n:raise ValueError(f'{unit["key"]}: wrong {role} count')
    if not vqa:
        for role,n in [('support',unit['fit_count']),('calibration',unit['cal_count'])]:
            if Counter(r['label_id'] for r in data[role])!=Counter({i:n//unit['classes'] for i in range(unit['classes'])}):
                raise ValueError('Class budget changed')
    if images:
        from aperture_comprehensive.data import content_hash
        for rs in data.values():
            for r in rs:
                if not r.get('content_hash') or content_hash(r)!=r['content_hash']:raise ValueError('Missing or changed pixel fingerprint')
    return data


def prepare_support(cfg, repo):
    """Reuse the official-metadata patient partitioner; no arbitrary new thresholds."""
    from aperture_medical_multimodel.data import prepare
    from aperture_comprehensive.data import content_hash
    root=Path(repo)/cfg['prepared_root']/'support'
    spec={'protocol':'aperture_rebuttal_support_v1','data_root':str(Path(repo)/cfg['medical_data_root']),
          'prepared_root':str(root),'hospitals':[0,2],'support_seeds':[0,1,2],
          'primary_support':16,'curve_supports':[32,64],'query_count':500,'split_seed':cfg['split_seed']}
    # Native preparation is immutable. Put pixel-augmented manifests into a
    # separate tree rather than rewriting the native partitioner's receipts.
    spec['prepared_root']=str(root.parent/'support_native')
    receipt=prepare(spec)
    for h in (0,2):
        for seed in (0,1,2):
            previous=None
            for budget in (16,32,64):
                rel=Path(f'hospital_{h}')/f'support_{budget}'/f'seed_{seed}'
                unit={}
                for role in ('fit','calibration','query'):
                    rs=rows(Path(spec['prepared_root'])/rel/f'{role}.jsonl')
                    for r in rs:r['content_hash']=content_hash(r)
                    write_rows(root/rel/f'{role}.jsonl',rs);unit[role]=rs
                if previous:
                    for role in ('fit','calibration'):
                        if not {r['sample_id'] for r in previous[role]}<={r['sample_id'] for r in unit[role]}:
                            raise ValueError('Support curve is not nested')
                    if previous['query']!=unit['query']:raise ValueError('Query population changed across budget')
                previous=unit
    atomic(root/'preparation.json',receipt,immutable=True)
    return {'prepared':str(root),'budget_total_labels':[16,32,64],'hospitals':[0,2]}


def evaluator_path(cfg, repo):return Path(repo)/cfg['prepared_root']/'evaluators/m4c_evaluators.py'


def evaluator(cfg, repo):
    path=evaluator_path(cfg,repo);data=path.read_bytes()
    git_sha=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    if git_sha!=EVALUATOR_BLOB:raise ValueError('Official evaluator source fingerprint changed')
    spec=importlib.util.spec_from_file_location('aperture_official_textvqa',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def prepare_vqa(cfg, repo):
    """Official JSON + local images; use all 5,000 validation questions."""
    import base64
    from urllib.request import urlopen
    from aperture_comprehensive.data import content_hash
    repo=Path(repo);root=repo/cfg['prepared_root']/'textvqa';path=evaluator_path(cfg,repo)
    if not path.exists():
        # Pinned Git blob, not mutable main; the evaluator is loaded only after
        # verifying the exact upstream bytes. No model is downloaded here.
        url='https://api.github.com/repos/facebookresearch/mmf/git/blobs/'+EVALUATOR_BLOB
        with urlopen(url,timeout=60) as response:blob=json.load(response)
        data=base64.b64decode(blob['content'])
        if hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=EVALUATOR_BLOB:raise ValueError('Evaluator download checksum failed')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    official=evaluator(cfg,repo);normalize=official.EvalAIAnswerProcessor()
    inputs={role:repo/cfg['textvqa_'+role+'_json'] for role in ('train','val')}
    source={str(p):sha(p) for p in inputs.values()}
    def convert(p):
        output=[]
        for r in read_json(p)['data']:
            answers=r['answers']
            if len(answers)!=10:raise ValueError('TextVQA requires all ten human reference answers')
            counts=Counter(normalize(a) for a in answers)
            target=min(counts,key=lambda a:(-counts[a],a))
            image=repo/cfg['textvqa_images']/(str(r['image_id'])+'.jpg')
            row={'sample_id':str(r['question_id']),'group_id':str(r['image_id']),'image':str(image.resolve()),
                 'question':r['question'],'answers':answers,'label':target,'label_id':0,'candidate_labels':[target]}
            row['content_hash']=content_hash(row);output.append(row)
        return output
    train=convert(inputs['train']);query=convert(inputs['val'])
    if len(query)!=5000:raise ValueError('Use complete official 5,000-question validation, not a subset')
    qgroups={r['group_id'] for r in query};qpixels={r['content_hash'] for r in query}
    eligible=[r for r in train if r['group_id'] not in qgroups and r['content_hash'] not in qpixels]
    for seed in (0,1,2):
        seen=set();chosen=[]
        for r in ordered(eligible,f'textvqa-support:{seed}'):
            if r['group_id'] not in seen:
                chosen.append(r);seen.add(r['group_id'])
            if len(chosen)==64:break
        if len(chosen)!=64:raise ValueError('Not enough independent support images')
        unit={'support':chosen[:48],'calibration':chosen[48:],'query':query}
        validate(unit,True,vqa=True)
        for role,rs in unit.items():write_rows(root/f'seed_{seed}'/f'{"fit" if role=="support" else role}.jsonl',rs)
    atomic(root/'preparation.json',{'sources':source,'evaluator_git_blob':EVALUATOR_BLOB,'query_count':5000,
        'fit':48,'calibration':16,'image_disjoint':True,'external_ocr_used':False},immutable=True)
    return {'prepared':str(root),'query_count':5000}


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
