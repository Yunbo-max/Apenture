"""Final regression checks on CG execution identity and complete reporting."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_conditional_gated import cli, engine, plan, report
from aperture_rebuttal_cases import store
from test_conditional_gated import make_base
from test_conditional_gated_integration import cli_fixture, fake_module, make_sealed_report


def test_explicit_batch_is_same_in_all_unit_fields(tmp_path, monkeypatch):
    monkeypatch.delenv('APERTURE_BATCH',raising=False)
    units=plan.build_units('CG02',make_base(),tmp_path/'runs/conditional_gated',tmp_path,batch=2)
    assert all(u['batch_size']==u['execution_batch_size']==2 for u in units)


def test_nondefault_batch_plan_idempotent_with_or_without_matching_environment(cli_fixture, monkeypatch, capsys):
    repo,cfg=cli_fixture
    cfg['execution_batch']=2
    store.atomic(repo/'configs/cg.json',cfg)
    args=['plan','--case','CG02','--config','configs/cg.json']
    assert cli.main(args)==0
    before=store.sha(repo/cfg['run_root']/'CG02/plan.json')
    assert cli.main(args)==0 # CLI set APERTURE_BATCH=2 after first freeze
    assert store.sha(repo/cfg['run_root']/'CG02/plan.json')==before
    monkeypatch.delenv('APERTURE_BATCH')
    assert cli.main(args)==0
    assert store.sha(repo/cfg['run_root']/'CG02/plan.json')==before


def test_runtime_binding_rejects_batch_different_from_frozen_plan(tmp_path,monkeypatch):
    fake_module(monkeypatch,'aperture_comprehensive.worker',check_assets=lambda *a,**kw:{})
    monkeypatch.setenv('APERTURE_BATCH','1')
    unit={'output':str(tmp_path/'CG01/units/u'),'batch_size':2,'execution_batch_size':2}
    for role in ('support','calibration','query'):
        p=tmp_path/(role+'.jsonl');store.write_rows(p,[{'sample_id':role}]);unit[role]=str(p)
    with pytest.raises(ValueError,match='batch'):
        engine.binding(unit,{'settings':{'execution_batch':2}})


def test_failed_summary_removes_stale_fit_calibration_report(cli_fixture):
    repo,cfg=cli_fixture
    p=plan.freeze('CG02',cfg,repo)
    out=repo/cfg['run_root']/'CG02/report';out.mkdir(parents=True)
    stale=out/'fit_calibration_diagnostics.json';store.atomic(stale,{'rows':[{'stale':True}]})
    with pytest.raises(ValueError):report.summarize(p)
    assert not stale.exists()


def test_complete_report_checks_prediction_ids_against_query_manifest(tmp_path):
    p=make_sealed_report(tmp_path)
    u=p['units'][0];variant=u['variants'][0]['name']
    ev=Path(u['output'])/'evaluation'/variant
    bound=store.verify(ev)['binding'];pred=store.rows(ev/'predictions.jsonl')[:1]
    # A consistent but partial producer output must NOT become a complete case.
    (ev/'predictions.jsonl').unlink();store.write_rows(ev/'predictions.jsonl',pred)
    metric={'raw':report.metrics(pred),'temperature_scaled':report.metrics(pred),'temperature':1.}
    store.atomic(ev/'metrics.json',metric)
    (ev/'DONE.json').unlink();store.seal(ev,bound)
    # Trim all other units too: equality between runs is not full population coverage.
    for other in p['units'][1:]:
        dest=Path(other['output'])/'evaluation'/other['variants'][0]['name']
        b=store.verify(dest)['binding']
        (dest/'predictions.jsonl').unlink();store.write_rows(dest/'predictions.jsonl',pred)
        store.atomic(dest/'metrics.json',metric)
        (dest/'DONE.json').unlink();store.seal(dest,b)
    with pytest.raises(ValueError,match='[Qq]uery|population'):
        report.summarize(p)


def test_empty_scientific_plan_cannot_report_complete(tmp_path):
    p={'version':plan.VERSION,'method_changed':True,'query_status':'synthetic','case':'CG02',
       'settings':{'run_root':str(tmp_path)},'units':[]}
    with pytest.raises(ValueError,match='[Ee]mpty|[Nn]o.*unit'):
        report.summarize(p)


@pytest.mark.parametrize('reduction',['mean','sum'])
def test_default_single_row_trainer_learns_gate_and_preserves_base(reduction,monkeypatch):
    import torch
    from aperture_conditional_gated import adapters
    from aperture_conditional_gated.controller import ConditionalGatedController
    from test_conditional_gated_integration import ToyModel,ToyBackend,toy_roles
    monkeypatch.setenv('APERTURE_BATCH','1')
    model=ToyModel();backend=ToyBackend(model)
    controller=ConditionalGatedController([(1,'K',model.proj)],{(1,'K'):torch.eye(8)[:,:3]})
    before=model.proj.weight.detach().clone()
    try:
        history=adapters.train_control(model,backend,controller,toy_roles()['support'],
            {'engine':'conditional_gated','steps':4,'lr':.05,'l2':.001,'loss_reduction':reduction})
        assert len(history)==4
        assert controller.raw['1:K'].detach().abs().sum()>0
        assert controller.gate_weight['1:K'].detach().abs().sum()>0
        assert controller._mask is None
        assert torch.equal(before,model.proj.weight)
        assert all(r['reduction']==reduction for r in history)
    finally:controller.close()


def test_empty_selection_cannot_report_complete(tmp_path):
    p=make_sealed_report(tmp_path);u=p['units'][0]
    store.atomic(Path(u['output'])/'selection.json',{'evaluate':[]})
    with pytest.raises(ValueError,match='[Ee]mpty'):
        report.summarize(p)


def test_diagnostic_failure_revokes_all_success_products(tmp_path):
    p=make_sealed_report(tmp_path)
    output=report.summarize(p)['output']
    u=p['units'][0];fit=Path(u['output'])/'fits'/u['variants'][0]['name']
    (fit/'gate_diagnostics.json').unlink()
    with pytest.raises((ValueError,OSError)):report.summarize(p)
    for name in ('summary.json','results.json','results.csv','fit_calibration_diagnostics.json'):
        assert not (Path(output)/name).exists()
    assert store.read(Path(output)/'coverage.json')['status']=='incomplete_or_invalid'


def test_free_generation_population_does_not_require_training_label_in_prediction(tmp_path):
    refs=[{'sample_id':'q1','group_id':'img1','label_id':0,'answers':['yes']*10}]
    predictions=[{'sample_id':'q1','group_id':'img1','pred_answer':'yes'}]
    query=tmp_path/'query.jsonl';store.write_rows(query,refs)
    folder=tmp_path/'u/evaluation/gated';store.write_rows(folder/'predictions.jsonl',predictions)
    store.seal(folder,{'synthetic':True})
    report._query_population({'query':str(query),'output':str(tmp_path/'u'),'task_type':'vqa','query_count':1},['gated'])


def test_classification_population_rejects_changed_labels(tmp_path):
    refs=[{'sample_id':'q1','group_id':'img1','label_id':0}]
    predictions=[{'sample_id':'q1','group_id':'img1','label_id':1}]
    query=tmp_path/'query.jsonl';store.write_rows(query,refs)
    folder=tmp_path/'u/evaluation/gated';store.write_rows(folder/'predictions.jsonl',predictions)
    store.seal(folder,{'synthetic':True})
    with pytest.raises(ValueError,match='labels/groups'):
        report._query_population({'query':str(query),'output':str(tmp_path/'u'),'task_type':'classification','query_count':1},['gated'])
