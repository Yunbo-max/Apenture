"""Complete model/task/arm expansion, immutable plans and role validation."""
from collections import Counter
from pathlib import Path
import json
import re

from aperture_table.protocol import write_locked, _save_rows
from vigor_handoff.protocol import digest, file_hash, resolved_manifest

PROTOCOL = 'aperture_comprehensive_v1'
MODELS = {
    'q25': ('Qwen/Qwen2.5-VL-7B-Instruct', 'qwen2_5_vl', 'Qwen2_5_VLForConditionalGeneration', 448),
    'q34': ('Qwen/Qwen3-VL-4B-Instruct', 'qwen3_vl', 'Qwen3VLForConditionalGeneration', 448),
    'q38': ('Qwen/Qwen3-VL-8B-Instruct', 'qwen3_vl', 'Qwen3VLForConditionalGeneration', 448),
    'g34': ('google/gemma-3-4b-it', 'gemma3', 'Gemma3ForConditionalGeneration', 896),
    'internvl3': ('OpenGVLab/InternVL3-8B-Instruct', 'internvl_chat', 'AutoModel', 224),
    'phi': ('microsoft/Phi-3.5-vision-instruct', 'phi3_v', 'AutoModelForCausalLM', 448),
}
ARMS = ('frozen', 'lora1', 'lora4', 'random', 'aperture')


def source_signature():
    root = Path(__file__).resolve().parents[1]
    packages = ('aperture_comprehensive', 'aperture_table', 'aperture_medical',
                'aperture_final', 'vigor_handoff', 'visual_lens', 'eventttt')
    files = sorted(p for pkg in packages for p in (root/pkg).glob('*.py'))
    return digest([(str(p.relative_to(root)), file_hash(p)) for p in files])


def make_jobs(config):
    if config.get('protocol') != PROTOCOL:
        raise ValueError('wrong comprehensive protocol')
    seeds = config['support_seeds']
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('unique nonnegative support seeds required')
    models = config['models']; datasets = config['datasets']
    if {m['key'] for m in models} != set(MODELS) or len(models) != len(MODELS):
        raise ValueError('the comprehensive plan requires all six checkpoints')
    keys = [d['key'] for d in datasets]
    if not keys or len(set(keys)) != len(keys) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', k) for k in keys):
        raise ValueError('unique safe dataset keys required')
    jobs = []
    for model in models:
        key = model['key']; ident, family, _, size = MODELS[key]
        if model.get('id', ident) != ident or model.get('family', family) != family:
            raise ValueError('checkpoint substitution is forbidden')
        for data in datasets:
            if any(k in data for k in ('families','models','arms')):
                raise ValueError('dataset-specific model/arm exclusions are forbidden')
            if data['kind'] not in ('paired','single') or data['classes'] < 2:
                raise ValueError('invalid task kind or class count')
            classes = data['classes']; paired = data['kind'] == 'paired'
            for seed in seeds:
                unit = f"{key}--{data['key']}--s{seed}"
                folder = Path(config['prepared_root'])/data['key']/f'seed_{seed}'
                for arm in ARMS:
                    job_id = f'{unit}--{arm}'
                    jobs.append(dict(protocol=PROTOCOL, job_id=job_id, unit_id=unit,
                        model_key=key, model_id=ident, family=family,
                        model_path=str(Path(model['path']).resolve()), domain_key=data['key'],
                        domain=data['key'], kind=data['kind'], classes=classes,
                        query_count=data.get('query_count'), fit_count=6*classes, cal_count=2*classes,
                        total_labels=8*classes, support_seed=seed, arm=arm,
                        support=str(folder/'fit.jsonl'), calibration=str(folder/'calibration.jsonl'),
                        query=str(folder.parent/'query.jsonl'), split_record=str(folder/'split.json'),
                        output=str(Path(config['run_root'])/'states'/job_id),
                        rank=16, layers=[14,27], alpha=3., steps=4, lr=.05, l2=.001,
                        loss_reduction='mean' if paired else 'sum', mask='post' if paired else 'all_visual',
                        basis_mode='random' if arm=='random' else 'covariance', basis_seed=0,
                        lora_rank=16, lora_alpha=32, lora_dropout=.05,
                        lora_lr=1e-4 if paired else 2e-4, lora_accumulation=3 if paired else 1,
                        lora_passes=4 if arm=='lora4' else 1, optimization_seed=0,
                        temperature_penalty=.01, bias_l2=.001, image_size=size,
                        attention='sdpa_math' if family!='phi3_v' else 'eager', weight_dtype='bfloat16'))
    return jobs


def validate_roles(roles, classes, *, fit_cal_groups=False, variable_candidates=False):
    ids = {}; groups = {}; pixels = {}; inputs = {}
    for role, rows in roles.items():
        if not rows:
            raise ValueError(f'empty {role}')
        ids[role] = [r['sample_id'] for r in rows]
        groups[role] = {r['group_id'] for r in rows}
        pixels[role] = [r['content_hash'] for r in rows]
        inputs[role] = [digest([r['content_hash'],r.get('question',''),r['candidate_labels']]) for r in rows]
        if len(set(ids[role])) != len(rows) or len(set(inputs[role])) != len(rows):
            raise ValueError(f'duplicate IDs or complete image/question inputs in {role}')
        if not all(groups[role]):
            raise ValueError('missing group identity')
        for row in rows:
            labels = row['candidate_labels']; y = row['label_id']
            valid_count=2<=len(labels)<=classes if variable_candidates else len(labels)==classes
            if not valid_count or len(set(labels))!=len(labels) or type(y) is not int or not 0<=y<len(labels) or row['label']!=labels[y]:
                raise ValueError('candidate/label contract mismatch')
    for a,b in [('support','calibration'),('support','query'),('calibration','query')]:
        if set(ids[a])&set(ids[b]) or set(inputs[a])&set(inputs[b]):
            raise ValueError('sample or image content leakage across roles')
        if (b=='query' or fit_cal_groups) and set(pixels[a])&set(pixels[b]):
            raise ValueError('image content leakage across roles')
        if (b=='query' or fit_cal_groups) and groups[a]&groups[b]:
            raise ValueError('group leakage across roles')


def read_roles(job, images=False):
    from .data import content_hash
    record = json.loads(Path(job['split_record']).read_text())
    if (record['protocol'],record['domain'],record['support_seed']) != (PROTOCOL,job['domain_key'],job['support_seed']):
        raise ValueError('wrong split record')
    roles = {k:resolved_manifest(job[k]) for k in ('support','calibration','query')}
    validate_roles(roles,job['classes'],fit_cal_groups=record['fit_cal_group_disjoint'],
                   variable_candidates=record.get('variable_candidates',False))
    for role, rows in roles.items():
        expected = {'support':job['fit_count'],'calibration':job['cal_count'],'query':job['query_count']}[role]
        if (expected is not None and len(rows)!=expected) or digest(rows)!=record['content'][role]:
            raise ValueError('prepared rows/count changed')
        if role != 'query' and Counter(r['label_id'] for r in rows)!=Counter({i:(6 if role=='support' else 2) for i in range(job['classes'])}):
            raise ValueError('support class budget mismatch')
        if images:
            for row in rows:
                if content_hash(row)!=row['content_hash']:
                    raise ValueError('image content changed')
    return roles
