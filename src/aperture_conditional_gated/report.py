"""Budget-aware, family-aware complete reports; raw and scaled scores stay separate."""
from pathlib import Path
from aperture_rebuttal_cases import report as legacy
from aperture_rebuttal_cases import store
from .plan import VERSION

metrics=legacy.metrics
select=legacy.select
coverage=legacy.coverage


def summarize(plan):
    if plan.get('version')!=VERSION or not plan.get('method_changed'):
        raise ValueError('Not a conditional-gated experiment plan')
    out=Path(plan['settings']['run_root'])/plan['case']/'report'
    try:result=legacy.summarize(plan)
    finally:
        # The inherited reporter writes protocol-independent raw measurements;
        # stamp the NEW method identity even for incomplete coverage.
        for name in ('coverage.json','results.json','summary.json'):
            path=out/name
            if path.exists():
                data=store.read(path)
                data.update(method_changed=True,method_version=VERSION,prior_best_reproduced=False,
                            query_status=plan['query_status'])
                store.atomic(path,data)
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
    store.atomic(out/'fit_calibration_diagnostics.json',{'rows':diagnostics,
        'note':'final-state metrics; CG08 uses independent fits at declared budgets, not resumed trajectories'})
    return {**result,'method_version':VERSION}


def analyze(source_plan,out,units=None):
    report={}
    for u in source_plan['units'] if units is None else units:
        folder=Path(u['output']);store.verify(folder/'selection_receipt',{'unit':u['key'],'plan':store.digest(source_plan)})
        selection=store.read(folder/'selection.json')
        if selection!=store.read(folder/'selection_receipt/selection.json'):raise ValueError('Selection changed')
        names=selection['evaluate']
        required=('frozen','static-full','conditional-gated','random-gated','lora1','lora4')
        if not set(required)<=set(names):raise ValueError('Missing predeclared comparison arms')
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
