"""Budget-aware, family-aware complete reports; raw and scaled scores stay separate."""
from pathlib import Path
from aperture_rebuttal_cases import report as legacy
from aperture_rebuttal_cases import store
from .plan import VERSION

metrics=legacy.metrics
select=legacy.select
coverage=legacy.coverage


def _query_population(unit, names):
    """Seals alone do not prove full coverage of the registered query manifest."""
    refs=store.rows(unit['query'])
    expected=[r['sample_id'] for r in refs]
    if not expected or len(set(expected))!=len(expected):
        raise ValueError('Empty or duplicate query manifest population')
    declared=unit.get('query_count')
    if declared is not None and len(expected)!=declared:
        raise ValueError('Query manifest size differs from declared population')
    if not names or len(set(names))!=len(names):
        raise ValueError('Empty or duplicate query evaluation selection')
    for name in names:
        folder=Path(unit['output'])/'evaluation'/name
        if not (folder/'DONE.json').is_file():
            continue # Legacy coverage reports the missing execution, never a success.
        store.verify(folder)
        predictions=store.rows(folder/'predictions.jsonl')
        if [r['sample_id'] for r in predictions]!=expected:
            raise ValueError('Query prediction population differs from registered manifest')
        keys=('label_id','group_id') if unit['task_type']=='classification' else ('group_id',)
        for prediction,reference in zip(predictions,refs):
            for key in keys:
                if key in reference and prediction.get(key)!=reference[key]:
                    raise ValueError('Query labels/groups differ from registered manifest')


def summarize(plan):
    if plan.get('version')!=VERSION or not plan.get('method_changed'):
        raise ValueError('Not a conditional-gated experiment plan')
    out=Path(plan['settings']['run_root'])/plan['case']/'report'
    products=('results.json','results.csv','summary.json','fit_calibration_diagnostics.json')
    for name in products:(out/name).unlink(missing_ok=True)
    store.atomic(out/'coverage.json',{'case':plan['case'],'status':'checking','units':{}})
    try:
        if not plan['units']:raise ValueError('Empty scientific comparison plan')
        keys=[u['key'] for u in plan['units']]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate comparison unit')
        for u in plan['units']:
            selection=Path(u['output'])/'selection.json'
            if selection.is_file():_query_population(u,store.read(selection)['evaluate'])
        result=legacy.summarize(plan)
        diagnostics=[]
        for u in plan['units']:
            selection=store.read(Path(u['output'])/'selection.json')
            for name in selection['evaluate']:
                fit=Path(u['output'])/'fits'/name;record=store.read(fit/'fit.json')
                diagnostics.append({'unit':u['key'],'model':u['model_key'],'domain':u['domain'],
                    'total_labels':u['total_labels'],'seed':u['support_seed'],'variant':name,
                    'fit':record.get('final_support_metrics'),'calibration':record.get('final_calibration_metrics'),
                    'optimizer_updates':record['optimizer_updates'],'support_visits':record['support_visits'],
                    'basis_visits':record['basis_visits'],'history':record['history'],
                    'gate':store.read(fit/'gate_diagnostics.json')})
        store.atomic(out/'fit_calibration_diagnostics.json',{'method_version':VERSION,'rows':diagnostics,
            'note':'final-state metrics; CG08 uses independent fits at declared budgets, not resumed trajectories'})
        return {**result,'method_version':VERSION}
    except Exception as exc:
        # Do not leave a previous success or a partly generated report looking current.
        for name in products:(out/name).unlink(missing_ok=True)
        coverage_path=out/'coverage.json'
        current=store.read(coverage_path) if coverage_path.exists() else {'case':plan['case']}
        current.update(status='incomplete_or_invalid',error=str(exc))
        store.atomic(coverage_path,current)
        raise
    finally:
        for name in ('coverage.json','results.json','summary.json'):
            path=out/name
            if path.exists():
                data=store.read(path)
                data.update(method_changed=True,method_version=VERSION,prior_best_reproduced=False,
                            query_status=plan['query_status'])
                store.atomic(path,data)


def analyze(source_plan,out,units=None):
    (Path(out)/'domain_analysis.json').unlink(missing_ok=True)
    report={}
    for u in source_plan['units'] if units is None else units:
        folder=Path(u['output']);store.verify(folder/'selection_receipt',{'unit':u['key'],'plan':store.digest(source_plan)})
        selection=store.read(folder/'selection.json')
        if selection!=store.read(folder/'selection_receipt/selection.json'):raise ValueError('Selection changed')
        names=selection['evaluate']
        required=('frozen','static-full','conditional-gated','random-gated','lora1','lora4')
        if not set(required)<=set(names):raise ValueError('Missing predeclared comparison arms')
        _query_population(u,required)
        predictions={}
        for name in required:
            fit=folder/'fits'/name;store.verify(fit)
            evaluated=folder/'evaluation'/name;bound=store.verify(evaluated)['binding']
            if selection['fit_seals'][name]!=store.sha(fit/'DONE.json') or bound['fit_seal']!=store.sha(fit/'DONE.json'):
                raise ValueError('Analysis source state mismatch')
            predictions[name]=store.rows(evaluated/'predictions.jsonl')
        right=predictions['conditional-gated'];comparisons={}
        for name in required:
            if name=='conditional-gated':continue
            left=predictions[name]
            comparisons[name]={'paired_accuracy':legacy.paired_accuracy(left,right),
                'baseline_metrics':metrics(left),'gated_metrics':metrics(right),
                'examples':legacy.error_examples(left,right)}
        report[u['key']]=comparisons
    store.atomic(Path(out)/'domain_analysis.json',{'method_version':VERSION,'units':report,'causal_claim':False,
        'selection':'fixed-hash positive/negative examples; IDs are private artifacts, not for public upload'})
    return {'units':len(report),'output':str(out)}
