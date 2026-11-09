"""Synthetic sealed artifacts test read-only diagnosis; no model/medical data."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aperture_rebuttal_cases import store


def auditor():
    path=ROOT/'scripts/audit_aperture_fit.py'
    if not path.exists():pytest.fail('read-only fit auditor not implemented')
    spec=importlib.util.spec_from_file_location('fit_auditor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def sealed_run(tmp_path):
    repo=tmp_path/'repo';(repo/'src').mkdir(parents=True)
    (repo/'src/engine.py').write_text('# immutable source\n')
    roles={}
    for role in ('support','calibration','query'):
        rs=[dict(sample_id=f'{role}-{i}',group_id=f'private-patient-{role}-{i}',
                 content_hash=f'h-{role}-{i}',label_id=i,label=['normal','tumor'][i],
                 candidate_labels=['normal','tumor'],question='classify') for i in range(2)]
        path=repo/'data'/f'{role}.jsonl';store.write_rows(path,rs);roles[role]=str(path)
    unit={'key':'g34--toy--n4--s0','output':str(repo/'runs/source/E01/units/toy'),
          'model_key':'g34','model_path':str(repo/'model'),'domain':'toy','classes':2,
          'layers':[1,2],'mask':'all_visual','loss_reduction':'mean','rank':2,**roles}
    sp={'case':'E01','source_files':{'src/engine.py':store.sha(repo/'src/engine.py')},
        'units':[unit],'settings':{'run_root':str(repo/'runs/source')}}
    spath=repo/'runs/source/E01/plan.json';store.atomic(spath,sp)
    sources={}
    for name in ('frozen','aperture'):
        variant={'name':name,'engine':name,'alpha':3.,'lr':.05,'steps':4,'l2':.001,'rank':2}
        binding={'unit':unit,'variant':variant,'plan':store.digest(sp),
                 'manifests':{k:store.sha(v) for k,v in roles.items()},'environment':{'torch':'test'}}
        folder=Path(unit['output'])/'fits'/name
        store.atomic(folder/'fit.json',{'batch_size':16,'history':[{'data_loss_mean':2.}],
            'query_forwards':0,'optimized_scalars':16 if name=='aperture' else 0})
        store.atomic(folder/'deployment.json',{'variant':variant,'layers':[1,2],'mask':'all_visual'})
        for role,filename in [('support','support_final.jsonl'),('calibration','calibration.jsonl')]:
            predictions=[]
            for r in store.rows(roles[role]):
                s=[-.1,-3.] if name=='aperture' else ([-.1,-3.] if r['label_id']==0 else [-3.,-.1])
                predictions.append({k:r[k] for k in ('sample_id','group_id','label_id')}|{'mean_log_scores':s})
            store.write_rows(folder/filename,predictions)
        store.seal(folder,binding)
        sources[name]={'folder':str(folder),'seal':store.sha(folder/'DONE.json'),
                       'source_plan_digest':store.digest(sp),'binding':binding}
    eroot=repo/'runs/retention';runit=eroot/'visual/units/toy'
    store.atomic(eroot/'visual/plan.json',{'repo':str(repo),'source_plans':[{'path':str(spath),'sha256':store.sha(spath)}],
        'settings':{'run_root':str(eroot)}})
    store.atomic(runit/'binding.json',{'source_fits':{'toy':sources}})
    return repo,runit,sources


def snapshot(repo):
    return {str(p.relative_to(repo)):store.sha(p) for p in repo.rglob('*') if p.is_file()}


def test_audit_reports_fit_and_calibration_without_mutating_or_exposing_rows(sealed_run):
    repo,unit,_=sealed_run;before=snapshot(repo)
    out=auditor().audit_retention(unit,repo,False)
    assert out['read_only'] and not out['query_metrics_read']
    assert out['source_snapshots'][0]['matching']
    item=out['domains']['toy']['fits']['aperture']
    assert item['fit_stats']['recorded_batch_size']==16
    assert item['support']['prediction_counts']==[2,0]
    assert item['calibration']['accuracy']==.5
    assert out['domains']['toy']['transitions']['calibration']['correct_to_wrong']==1
    assert 'private-patient' not in json.dumps(out)
    assert snapshot(repo)==before


def test_tampered_source_fit_is_rejected(sealed_run):
    repo,unit,sources=sealed_run
    (Path(sources['aperture']['folder'])/'deployment.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        auditor().audit_retention(unit,repo,False)


def test_source_mismatch_is_reported_not_silently_treated_as_original(sealed_run):
    repo,unit,_=sealed_run
    (repo/'src/engine.py').write_text('# different runtime\n')
    out=auditor().audit_retention(unit,repo,False)
    assert not out['source_snapshots'][0]['matching']
    assert out['source_snapshots'][0]['mismatches']==['src/engine.py']


def test_output_refuses_training_and_retention_roots(sealed_run,tmp_path):
    repo,unit,sources=sealed_run
    module=auditor();out=module.audit_retention(unit,repo,False)
    for path in [unit/'debug.json',Path(sources['frozen']['folder'])/'debug.json']:
        with pytest.raises(ValueError,match='sealed'):
            module.save_report(out,path)
    target=tmp_path/'audit.json';module.save_report(out,target)
    assert target.exists()
    with pytest.raises(FileExistsError):module.save_report(out,target)


def test_inherited_controller_l2_is_not_misreported_as_lora_or_frozen_penalty(sealed_run):
    repo,unit,_=sealed_run
    out=auditor().audit_retention(unit,repo,False)
    fits=out['domains']['toy']['fits']
    assert fits['frozen']['mean_objective_relative_l2'] is None
    assert fits['aperture']['mean_objective_relative_l2']==.001
