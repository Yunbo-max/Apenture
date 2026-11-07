"""Data-only preparation; model outcomes never influence these partitions."""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from PIL import Image
from vigor_handoff.protocol import resolved_manifest, digest, file_hash
from .protocol import PROTOCOL, make_jobs, write_locked, _save_rows, validate_roles


def q2_row(folder, domain, task):
    folder = Path(folder)
    question = (folder/f'{task}_question.txt').read_text().strip()
    answer = (folder/f'{task}_answer.txt').read_text().strip()
    match = re.search(r'pick one from\s+([\w, ]+)\)', question, re.I)
    if not match:
        raise ValueError(f'missing explicit Q2 options: {folder}/{task}')
    original_options = [s.strip().lower() for s in match[1].split(',')]
    options = list(dict.fromkeys(original_options))
    prefix = {'pick_point':'Picking Point: ', 'ending_tile':'Ending tile: '}[task]
    value = answer.split(':')[-1].strip().lower()
    if len(original_options)!=4 or len(options)<2 or value not in options:
        raise ValueError(f'invalid Q2 answer/options: {folder}/{task}')
    candidates = [prefix+x for x in options]
    return dict(sample_id=f'q2-{domain}-{task}-{folder.name}', group_id=folder.name,
                image=str((folder/'image.png').resolve()), question=question,
                candidate_labels=candidates, label=prefix+value,label_id=options.index(value),
                dataset='manipbench-q2', metadata={'task':task,'class_meaning':'candidate_position',
                                                  'original_options':original_options})


def ordered(rows, salt):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['sample_id']}".encode()).hexdigest())


def balanced(rows, per_class, classes, salt):
    result = []
    for label in range(classes):
        candidates = ordered([r for r in rows if r['label_id']==label], salt)
        if len(candidates)<per_class:
            raise ValueError(f'insufficient class {label} examples: {len(candidates)} < {per_class}')
        result.extend(candidates[:per_class])
    return sorted(result,key=lambda r:r['sample_id'])


def shared_query_support(pool, query, classes, seed, key):
    query_groups={r['group_id'] for r in query}
    eligible=[r for r in pool if r['group_id'] not in query_groups]
    return balanced(eligible,8,classes,f'{key}|support|{seed}')


def guardian_pool(data):
    """Reconstruct all task rows against already materialized official composites."""
    raw=[json.loads(s) for s in Path(data['metadata']).read_text().splitlines() if s.strip()]
    split=data['domain'];rows=[]
    for index,r in enumerate(raw):
        ident=r.get('episode_id',r.get('taskvar',index))
        image=Path(data['composites'])/f'{index:05d}_{ident}.jpg'
        if not image.is_file():raise FileNotFoundError(image)
        question=('The top half of the image is the observation before the manipulation and the '
                  'bottom half is the observation after it. Given the task instruction and the '
                  'before/after visual evidence, determine whether execution succeeded. '
                  'Task instruction: '+r.get('task_instruction','')+'.')
        if r.get('detailed_subtask_name'):question+=' The verified subtask is: '+r['detailed_subtask_name']+'.'
        question+=' Answer exactly one label: success or failure.'
        label='success' if int(r['execution_reward'])==1 else 'failure'
        rows.append(dict(sample_id=f'{split}-{ident}-{index}',group_id=str(ident),
                         image=str(image.resolve()),question=question,candidate_labels=['success','failure'],
                         label=label,label_id=0 if label=='success' else 1,dataset='guardian-execution',
                         metadata={'source_split':split,'taskvar':r.get('taskvar'),'episode_id':r.get('episode_id')}))
    return rows


def pool_split(rows, classes, query_count, seed, key):
    groups = sorted({r.get('group_id',r.get('tile_id')) for r in rows},
                    key=lambda g: hashlib.sha256(f'comprehensive-query-v1|{key}|{g}'.encode()).hexdigest())
    if None in groups:
        raise ValueError('pool requires explicit groups')
    # Deterministic, label-stratified allocation before any model outcomes.
    # Reserve support capacity while filling the query quota; a fixed fraction
    # cannot satisfy small/imbalanced pools even when a valid split exists.
    remaining=Counter(r['label_id'] for r in rows);query_counts=Counter()
    grouped=defaultdict(list)
    for row in rows:grouped[row.get('group_id',row.get('tile_id'))].append(row)
    query_groups=set();quota=query_count//classes
    for group in groups:
        counts=Counter(r['label_id'] for r in grouped[group])
        if not any(query_counts[c]<quota and counts[c] for c in range(classes)):continue
        if any(remaining[c]-counts[c]<8 for c in range(classes)):continue
        query_groups.add(group);query_counts.update(counts);remaining.subtract(counts)
        if all(query_counts[c]>=quota for c in range(classes)):break
    query_pool = [r for r in rows if r.get('group_id',r.get('tile_id')) in query_groups]
    support_pool = [r for r in rows if r.get('group_id',r.get('tile_id')) not in query_groups]
    if query_count % classes:
        raise ValueError('balanced query count must divide by class count')
    query = balanced(query_pool,query_count//classes,classes,f'{key}|query')
    support = balanced(support_pool,8,classes,f'{key}|support|{seed}')
    return support, query


@lru_cache(maxsize=12)
def _raster(path):
    from eventttt.vision import load_image
    return load_image(path)


def materialize(row, kind, folder):
    row = deepcopy(row)
    if kind=='paired':
        from eventttt.vision import crop_pair
        from eventttt.prompts import question_for
        from eventttt.schemas import DAMAGE_LABELS
        row['candidate_labels']=list(DAMAGE_LABELS)
        row['question']=row.get('question') or question_for()
        row['group_id']=row['tile_id']
        pre,post=crop_pair(_raster(row['pre_image']),_raster(row['post_image']),row.get('bbox_xyxy'),size=448)
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        name=hashlib.sha256(row['sample_id'].encode()).hexdigest()
        for key,im in [('pre_image',pre),('post_image',post)]:
            path=folder/f'{name}-{key}.png'
            if path.exists():
                with Image.open(path) as old:
                    if not np.array_equal(np.asarray(old.convert('RGB')),np.asarray(im)):
                        raise ValueError('prepared crop changed')
            else: im.save(path)
            row[key]=str(path.resolve())
        row['metadata']={**row.get('metadata',{}),'source_bbox_xyxy':row.get('bbox_xyxy')}
        row['bbox_xyxy']=None
    else:
        row['image']=str(Path(row['image']).resolve())
    row['content_hash']=content_hash(row)
    return row


def content_hash(row):
    keys=('pre_image','post_image') if row.get('pre_image') else ('image',)
    hashes=[]
    for key in keys:
        with Image.open(row[key]) as im: a=np.asarray(im.convert('RGB'))
        hashes.append(hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest())
    return digest(hashes)


def prepare(config):
    make_jobs(config)
    pools={}; materialized={}
    for data in config['datasets']:
        key=data['key']; classes=data['classes']; base=Path(config['prepared_root'])/key
        for seed in config['support_seeds']:
            sources={}
            if 'source_fit' in data:
                paths={k:Path(data['source_'+k].format(seed=seed)) for k in ('fit','calibration','query')}
                fit,cal,query=(resolved_manifest(paths[k]) for k in ('fit','calibration','query'))
                sources={str(p.resolve()):file_hash(p) for p in paths.values()}
                disjoint=data.get('fit_cal_group_disjoint',True)
            else:
                if data.get('source')=='bright_pool':
                    path=data['pool']
                    if path not in pools:pools[path]=resolved_manifest(path)
                    pool=[r for r in pools[path] if r['event_id']==data['event']]
                    support,query=pool_split(pool,classes,data['query_count'],seed,key)
                    sources={str(Path(path).resolve()):file_hash(path)}
                elif data.get('source')=='manipbench_q2':
                    if key not in pools:
                        pools[key]=[q2_row(p,data['domain'],data['task']) for p in sorted(Path(data['root']).glob('question_*'))]
                    support,query=pool_split(pools[key],classes,data['query_count'],seed,key)
                    sources={'pool_content':digest(pools[key])}
                elif data.get('source')=='guardian_shared':
                    if key not in pools:pools[key]=guardian_pool(data)
                    query_path=Path(data['source_query'])
                    query=resolved_manifest(query_path)
                    support=shared_query_support(pools[key],query,classes,seed,key)
                    sources={str(Path(data['metadata']).resolve()):file_hash(data['metadata']),
                             str(query_path.resolve()):file_hash(query_path)}
                else:
                    paths=[Path(data['source_'+k].format(seed=seed)) for k in ('support','query')]
                    support,query=(resolved_manifest(p) for p in paths)
                    sources={str(p.resolve()):file_hash(p) for p in paths}
                cal=balanced(support,2,classes,f'{key}|calibration|{seed}')
                cal_ids={r['sample_id'] for r in cal};fit=[r for r in support if r['sample_id'] not in cal_ids]
                disjoint=False
            roles={}
            for role,rs in [('support',fit),('calibration',cal),('query',query)]:
                roles[role]=[]
                for row in rs:
                    cache=(key,digest(row))
                    if cache not in materialized:materialized[cache]=materialize(row,data['kind'],base/'images')
                    roles[role].append(materialized[cache])
            validate_roles(roles,classes,fit_cal_groups=disjoint,variable_candidates=data.get('variable_candidates',False))
            folder=base/f'seed_{seed}'
            record=dict(protocol=PROTOCOL,domain=key,support_seed=seed,
                        source_files=sources,content={k:digest(v) for k,v in roles.items()},
                        fit_cal_group_disjoint=disjoint,query_group_disjoint=True,
                        class_meaning=data.get('class_meaning','semantic_class'))
            if data.get('variable_candidates'):record['variable_candidates']=True
            write_locked(folder/'split.json',record)
            for name,rows in roles.items():
                path=base/'query.jsonl' if name=='query' else folder/('fit.jsonl' if name=='support' else 'calibration.jsonl')
                _save_rows(path,rows)
            print(f'Prepared {key} seed {seed}: {len(fit)} fit / {len(cal)} calibration / {len(query)} query',flush=True)
    jobs=make_jobs(config)
    from .protocol import read_roles
    seen=set()
    for job in jobs:
        key=(job['domain_key'],job['support_seed'])
        if key not in seen:read_roles(job);seen.add(key)
    return jobs
