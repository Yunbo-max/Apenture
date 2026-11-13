#!/usr/bin/env python3
"""Read-only CG/E10 closure audit. Select hyperparameters from calibration only.

No GPU launches, no writes unless explicit --output (refuses overwrite), no
query-metric reads during selection, no modifications to frozen plans/seals.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

ALLOWED_SELECTION_CASES = ('CG05', 'CG08')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@lru_cache(maxsize=5000)
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified(folder):
    """Validate every file in an immutable DONE receipt (not just its presence)."""
    folder = Path(folder)
    try:
        receipt = read(folder / 'DONE.json')
        files = receipt.get('files')
        if not isinstance(files, dict) or not files:
            return False
        for rel, expected in files.items():
            path = (folder / rel).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file() or file_sha(str(path)) != expected:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def cg_plan(root, case):
    p = Path(root) / case / 'plan.json'
    plan = read(p)
    if plan.get('case') != case or plan.get('settings', {}).get('run_root') != str(Path(root).resolve()):
        raise ValueError(f'Frozen CG plan/root identity mismatch: {p}')
    return plan


def unit_path(root, case, unit):
    key = unit['key']
    if not key or '/' in key or '\\' in key or key in ('.', '..'):
        raise ValueError('Unsafe comparison unit')
    p = Path(root) / case / 'units' / key
    if Path(unit['output']).resolve() != p.resolve():
        raise ValueError(f'Unit output/root identity mismatch: {case}/{key}')
    return p


def split_signature(unit):
    """Read only hashes and declared protocol; never inspect query predictions."""
    keys = ('model_key', 'model_id', 'domain', 'total_labels', 'support_seed',
            'kind', 'classes', 'image_size', 'mask', 'layers', 'sites', 'rank')
    sig = {k: unit.get(k) for k in keys}
    sig.update({k + '_sha256': file_sha(str(Path(unit[k]).resolve())) for k in ('support', 'calibration', 'query')})
    return sig


def variant_family(variant):
    name = variant['name']
    if name.startswith('conditional-gated-'):
        return 'conditional-gated'
    if name.startswith('static-full-'):
        return 'static-full'
    return None


def select_candidates(root, cases=('CG05', 'CG08')):
    """Per-unit calibration-NLL best of the predeclared, completely sealed grid.

    CG05 alpha/LR and CG08 pass budgets are NOT a full Cartesian search. The
    union is eligible only if model, fit/calibration/query hashes & protocol agree.
    No query prediction, evaluation or query metric is loaded by this function.
    """
    root = Path(root).resolve()
    if not cases or len(set(cases)) != len(cases) or any(c not in ALLOWED_SELECTION_CASES for c in cases):
        raise ValueError(f'Only allowed candidate cases: {ALLOWED_SELECTION_CASES}')
    combined = {}
    for case in cases:
        plan = cg_plan(root, case)
        for unit in plan['units']:
            key = unit['key']
            entry = combined.setdefault(key, {'unit': key, 'model': unit['model_key'],
                                             'domain': unit['domain'], 'budget': unit['total_labels'],
                                             'seed': unit['support_seed'], 'signature': None,
                                             'families': defaultdict(list)})
            sig = split_signature(unit)
            if entry['signature'] is not None and entry['signature'] != sig:
                raise ValueError(f'Cross-case data manifests or protocol signature mismatch: {key}')
            entry['signature'] = sig
            folder = unit_path(root, case, unit)
            for var in unit['variants']:
                fam = variant_family(var)
                if fam is None:
                    continue
                fit = folder / 'fits' / var['name']
                row = {'case': case, 'variant': var['name'], 'family': fam,
                       'alpha': var['alpha'], 'lr': var['lr'], 'steps': var['steps'],
                       'fit_folder': str(fit), 'sealed': verified(fit)}
                if row['sealed']:
                    info = read(fit / 'fit.json')
                    value = info.get('calibration_nll')
                    if (info.get('name') != var['name'] or
                            not isinstance(value, (int, float)) or not math.isfinite(value)):
                        row['sealed'] = False
                    else:
                        row['calibration_nll'] = value
                        row['support_metrics'] = info.get('final_support_metrics')
                        row['calibration_metrics'] = info.get('final_calibration_metrics')
                        row['optimizer_updates'] = info.get('optimizer_updates')
                entry['families'][fam].append(row)
    result = []
    for key, unit in sorted(combined.items()):
        families = {}
        for fam, rows in sorted(unit['families'].items()):
            eligible = [r for r in rows if r['sealed']]
            missing = [r['case'] + ':' + r['variant'] for r in rows if not r['sealed']]
            # Do not advertise a partial-grid winner as "best".
            pick = min(eligible, key=lambda r: (r['calibration_nll'], r['case'], r['variant'])) if eligible and not missing else None
            families[fam] = {'status': 'complete' if not missing else 'incomplete',
                             'candidate_count': len(rows), 'sealed_count': len(eligible),
                             'missing': missing, 'selected': pick}
        result.append({k: v for k, v in unit.items() if k not in ('families', 'signature')} | {'families': families})
    return {'selection_rule': 'min raw calibration_nll per model/domain/seed/family; no query access',
            'case_sources': list(cases), 'search_space': 'CG05 alpha/LR (4 passes) UNION CG08 pass-count (fixed alpha/LR); not Cartesian',
            'calibration_selection_only': True, 'all_candidates_required': True,
            'units': result}


def cg_census(root, cases=('CG01', 'CG02', 'CG03', 'CG04', 'CG05', 'CG06', 'CG07', 'CG08')):
    root = Path(root).resolve()
    output = {}
    for case in cases:
        if not (root / case / 'plan.json').is_file():
            output[case] = {'status': 'missing_plan'}
            continue
        plan = cg_plan(root, case)
        status = {'planned_units': len(plan['units']), 'planned_fits': 0,
                  'sealed_fits': 0, 'sealed_evaluations': 0, 'missing': []}
        for unit in plan['units']:
            folder = unit_path(root, case, unit)
            status['planned_fits'] += len(unit['variants'])
            for var in unit['variants']:
                fit = folder / 'fits' / var['name']
                if verified(fit):
                    status['sealed_fits'] += 1
                else:
                    status['missing'].append({'unit': unit['key'], 'variant': var['name'], 'stage': 'fit'})
            receipt = folder / 'selection_receipt'
            if not verified(receipt):
                status['missing'].append({'unit': unit['key'], 'stage': 'selection'})
                continue
            selected = read(folder / 'selection.json').get('evaluate', [])
            for name in selected:
                if verified(folder / 'evaluation' / name):
                    status['sealed_evaluations'] += 1
                else:
                    status['missing'].append({'unit': unit['key'], 'variant': name, 'stage': 'evaluate'})
        status['complete'] = not status['missing']
        output[case] = status
    return {'root': str(root), 'cases': output,
            'note': 'Read-only status; no new training launched. Missing/invalid receipts are not DONE.'}


def audit_selected(root, cases=('CG02', 'CG03')):
    """Correct per-seed family selection before aggregating any query metrics.

    It rejects a wrong calibration-selected variant, tampered seal, missing
    query evaluation, or incomplete seed panel. It never *chooses* by query.
    """
    root=Path(root).resolve()
    if not cases or any(x not in ('CG02','CG03') for x in cases):
        raise ValueError('Only CG02/CG03 strong-baseline selection audits are allowed')
    unit_rows=[];issues=[];expected=defaultdict(set)
    for case in cases:
        plan=cg_plan(root,case)
        for unit in plan['units']:
            folder=unit_path(root,case,unit)
            selected_path=folder/'selection.json'
            families={v.get('family',v['name']) for v in unit['variants']}
            for family in families:
                expected[(case,unit['model_key'],unit['domain'],unit['total_labels'],family)].add(unit['support_seed'])
            if not selected_path.is_file() or not verified(folder/'selection_receipt'):
                issues.append({'case':case,'unit':unit['key'],'reason':'unsealed_selection'});continue
            receipt=read(folder/'selection_receipt'/'selection.json')
            selected=read(selected_path)
            if selected!=receipt:
                issues.append({'case':case,'unit':unit['key'],'reason':'selection_receipt_mismatch'});continue
            for family in sorted(families):
                choices=[v for v in unit['variants'] if v.get('family',v['name'])==family]
                records=[]
                for v in choices:
                    fit=folder/'fits'/v['name']
                    if not verified(fit):
                        records=[];break
                    info=read(fit/'fit.json')
                    val=info.get('calibration_nll')
                    if not isinstance(val,(int,float)) or not math.isfinite(val) or info.get('name')!=v['name']:
                        records=[];break
                    records.append((val,v['name']))
                if len(records)!=len(choices):
                    issues.append({'case':case,'unit':unit['key'],'family':family,'reason':'missing_or_invalid_candidate_fit'});continue
                correct=min(records,key=lambda x:(x[0],x[1]))[1]
                chosen=selected.get('selected',{}).get(family)
                if chosen!=correct:
                    issues.append({'case':case,'unit':unit['key'],'family':family,'reason':'calibration_selection_mismatch',
                                   'expected':correct,'recorded':chosen});continue
                evaluation=folder/'evaluation'/chosen
                if chosen not in selected.get('evaluate',[]) or not verified(evaluation):
                    issues.append({'case':case,'unit':unit['key'],'family':family,'reason':'missing_or_invalid_selected_evaluation'});continue
                raw=read(evaluation/'result.json').get('raw')
                if not isinstance(raw,dict) or not all(k in raw for k in ('accuracy','macro_f1','nll','n')):
                    issues.append({'case':case,'unit':unit['key'],'family':family,'reason':'missing_raw_metrics'});continue
                if unit.get('query_count') and raw['n']!=unit['query_count']:
                    issues.append({'case':case,'unit':unit['key'],'family':family,'reason':'query_population_size_mismatch'});continue
                unit_rows.append({'case':case,'model':unit['model_key'],'domain':unit['domain'],'budget':unit['total_labels'],
                                  'seed':unit['support_seed'],'family':family,'selected_variant':chosen,
                                  'calibration_nll':next(val for val,name in records if name==chosen),
                                  'accuracy':raw['accuracy'],'macro_f1':raw['macro_f1'],'nll':raw['nll'],'n':raw['n']})
    by=defaultdict(list)
    for row in unit_rows:
        key=(row['case'],row['model'],row['domain'],row['budget'],row['family'])
        by[key].append(row)
    summary=[]
    for key,rows in sorted(by.items()):
        required=expected[key]
        if {r['seed'] for r in rows}!=required or not required:
            issues.append({'case':key[0],'family':key[-1],'scope':'|'.join(map(str,key)),
                           'reason':'incomplete_independent_seed_panel','valid_seeds':[r['seed'] for r in rows],
                           'required_seeds':sorted(required)})
            continue
        item=dict(zip(('case','model','domain','budget','family'),key));item['seeds']=sorted(required)
        item['accuracy']=sum(r['accuracy'] for r in rows)/len(rows)
        item['macro_f1']=sum(r['macro_f1'] for r in rows)/len(rows)
        item['nll']=sum(r['nll'] for r in rows)/len(rows)
        item['selected_by_seed']={str(r['seed']):r['selected_variant'] for r in rows}
        summary.append(item)
    return {'status':'complete' if not issues else 'partial','selection_rule':'raw_calibration_nll',
            'query_used_for_selection':False,'unit_rows':unit_rows,'summary':summary,'issues':issues,
            'warning':'Do not average CG02/CG03 summary_raw variant rows; use per-seed selected family results.'}


def q34_diagnose(root):
    """Post-hoc query diagnosis only; these findings cannot select parameters."""
    root = Path(root).resolve()
    if not (root / 'CG01' / 'plan.json').is_file():
        return {'status': 'no_verified_pairs', 'units': []}
    plan = cg_plan(root, 'CG01')
    rows = []
    for u in plan['units']:
        if u['model_key'] != 'q34':
            continue
        folder = unit_path(root, 'CG01', u)
        a, b = folder / 'evaluation' / 'frozen', folder / 'evaluation' / 'conditional-gated'
        if not verified(a) or not verified(b):
            continue
        def predictions(p):
            return [json.loads(line) for line in (p / 'predictions.jsonl').read_text().splitlines() if line.strip()]
        left, right = predictions(a), predictions(b)
        if not left or len(left) != len(right) or [x['sample_id'] for x in left] != [x['sample_id'] for x in right]:
            raise ValueError(f'q34 query population mismatch: {u["key"]}')
        labels = Counter(str(x['prediction']) for x in right)
        score_delta = max(abs(float(s)-float(t)) for x, y in zip(left, right)
                          for s, t in zip(x['mean_log_scores'], y['mean_log_scores']))
        fit = folder / 'fits' / 'conditional-gated'
        gate = read(fit / 'gate_diagnostics.json') if verified(fit) else None
        rows.append({'unit': u['key'], 'samples': len(left), 'prediction_disagreement': sum(x['prediction']!=y['prediction'] for x,y in zip(left,right)),
                     'max_abs_candidate_log_score_change': score_delta, 'gated_prediction_counts': dict(labels),
                     'gate_diagnostics': gate, 'query_used_for_selection': False})
    return {'status': 'diagnosed' if rows else 'no_verified_pairs', 'units': rows,
            'note': 'Post-hoc query diagnostic only, never hyperparameter selection. Native K/V forward verification remains required.'}


def cost_audit(root):
    root = Path(root).resolve()
    if not (root/'CG06'/'plan.json').exists():return {'status':'missing_plan','rows':[]}
    plan = cg_plan(root, 'CG06');out=[]
    for unit in plan['units']:
        folder = unit_path(root, 'CG06', unit)
        for var in unit['variants']:
            fit = folder/'fits'/var['name']
            if not verified(fit):continue
            info = read(fit/'fit.json')
            out.append({'unit':unit['key'], 'variant':var['name'], 'engine':var['engine'],
                        'lora_rank':var.get('lora_rank') if var['engine']=='lora' else None,
                        'lora_passes':var.get('lora_passes') if var['engine']=='lora' else None,
                        'trainable_scalars':info.get('optimized_scalars'),
                        'saved_bytes':info.get('serialized_state_bytes'),
                        'fit_seconds':info.get('fit_seconds_including_extraction')})
    return {'status':'verified_fit_rows','rows':out,
            'warning':'lora4 means four training passes, NOT rank 4. Stored bytes and optimized scalars are different metrics.'}


def e10_census(root):
    root=Path(root).resolve();output={}
    for part in ('visual','text','bank'):
        plan_path=root/part/'plan.json'
        if not plan_path.exists():
            output[part]={'status':'missing_plan','valid':0,'missing':[],'invalid':[]};continue
        plan=read(plan_path)
        if plan.get('part')!=part or plan.get('settings',{}).get('run_root')!=str(root):
            raise ValueError('E10 part/root plan mismatch')
        missing=[];invalid=[];valid=[]
        plan_digest=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
        for job in plan['jobs']:
            folder=root/part/'units'/job['key']
            if not (folder/'DONE.json').exists():missing.append(job['key'])
            elif not verified(folder) or not (folder/'result.json').exists():invalid.append(job['key'])
            elif read(folder/'DONE.json').get('binding',{}).get('plan')!=plan_digest or read(folder/'DONE.json').get('binding',{}).get('job')!=job:
                invalid.append(job['key'])
            else:
                data=read(folder/'result.json')
                if data.get('all_identity_checks_passed') is not True:invalid.append(job['key'])
                else:valid.append(job['key'])
        output[part]={'status':'complete' if len(valid)==len(plan['jobs']) else 'partial',
                      'planned':len(plan['jobs']),'valid':len(valid),'missing':missing,'invalid':invalid}
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['census','select','strong','q34','cost','e10'])
    p.add_argument('--root',required=True,help='The EXACT frozen CG or E10 run root; no remapping')
    p.add_argument('--cases',nargs='+',choices=list(ALLOWED_SELECTION_CASES),default=list(ALLOWED_SELECTION_CASES))
    p.add_argument('--output',help='Optional new JSON report path; refuses existing outputs')
    args=p.parse_args(argv)
    try:
        funcs={'census':lambda:cg_census(args.root), 'select':lambda:select_candidates(args.root,args.cases),
               'strong':lambda:audit_selected(args.root), 'q34':lambda:q34_diagnose(args.root),'cost':lambda:cost_audit(args.root),
               'e10':lambda:e10_census(args.root)}
        result=funcs[args.action]()
        content=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.output:
            path=Path(args.output)
            with path.open('x',encoding='utf-8') as f:f.write(content)
        else:print(content,end='')
        return 0
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        print('FINALIZATION BLOCKED:',str(exc),file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())