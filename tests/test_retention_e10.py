"""Offline E10 contracts, not native-VLM performance or GPU qualification."""
import importlib
from pathlib import Path
import sys
import json
import pytest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))


def mod(name='protocol'):
    path=ROOT/'scripts/aperture_retention'/f'{name}.py'
    assert path.exists(), f'E10 {name} implementation missing'
    return importlib.import_module('aperture_retention.'+name)


def test_scope_and_no_implicit_continual_claim():
    m=mod()
    assert len(m.job_specs('visual'))==18
    assert len(m.job_specs('text'))==18
    assert len(m.job_specs('bank'))==24
    assert len(m.TARGETS['camelyon17-hospital_2'])==5
    assert m.METHODS==('frozen','aperture','lora1','lora4','lora-r4')
    assert 'q38' not in m.MODELS


def test_unit_filters_do_not_silently_fall_back():
    m=mod(); jobs=m.job_specs('visual')
    out=m.select_jobs(jobs,model='q25',domain='camelyon17-hospital_2',seed=0)
    assert len(out)==1
    with pytest.raises(ValueError,match='No'):
        m.select_jobs(jobs,model='missing')


def test_pair_metrics_sign_and_transitions():
    m=mod('metrics')
    rows=[dict(sample_id=str(i),group_id=str(i//2),label_id=y,scores=s) for i,(y,s) in enumerate([(0,[2.,0.]),(1,[0.,2.]),(0,[0.,2.]),(1,[2.,0.])])]
    new=[dict(r,scores=s) for r,s in zip(rows,[[0.,2.],[0.,2.],[2.,0.],[2.,0.]])]
    out=m.compare(rows,new,bootstrap=50)
    assert out['accuracy_delta']==0
    assert out['correct_to_wrong']==1 and out['wrong_to_correct']==1
    assert out['lost_original_correct_fraction']==.5
    assert out['retention_drop']==0
    assert out['ci_unit']=='query_group_conditional_on_support_seed'


def test_zero_original_correct_is_none_not_zero():
    m=mod('metrics');r=[dict(sample_id='a',group_id='g',label_id=0,scores=[0.,1.])]
    assert m.compare(r,r,bootstrap=0)['lost_original_correct_fraction'] is None


def test_population_order_label_nonfinite_checks():
    m=mod('metrics');r=[dict(sample_id='a',group_id='g',label_id=0,scores=[0.,1.])]
    for changed in [dict(r[0],sample_id='b'),dict(r[0],label_id=1),dict(r[0],scores=[float('nan'),1.])]:
        with pytest.raises(ValueError):m.compare(r,[changed],bootstrap=0)


def test_scalar_temperature_never_transferred():
    m=mod('metrics');r=[dict(sample_id='a',group_id='g',label_id=0,scores=[2.,0.],temperature=100.)]
    assert m.summarize(r)['nll']==pytest.approx(np.log1p(np.exp(-2)))


def test_source_file_unchanged_and_extra_src_rejected(tmp_path):
    m=mod();(tmp_path/'src').mkdir();(tmp_path/'src/a.py').write_text('x=1\n')
    files={'src/a.py':m.sha(tmp_path/'src/a.py')}
    m.verify_source(files,tmp_path)
    (tmp_path/'src/b.py').write_text('')
    with pytest.raises(ValueError,match='source'):
        m.verify_source(files,tmp_path)
    (tmp_path/'src/b.py').unlink();(tmp_path/'src/a.py').write_text('x=2\n')
    with pytest.raises(ValueError,match='source'):
        m.verify_source(files,tmp_path)


def test_overlap_reports_and_refuses(tmp_path):
    m=mod();train=[dict(sample_id='a',group_id='p',content_hash='x')]
    m.no_overlap(train,[dict(sample_id='b',group_id='q',content_hash='y')])
    for r in [dict(sample_id='a',group_id='q',content_hash='y'),dict(sample_id='b',group_id='p',content_hash='y'),dict(sample_id='b',group_id='q',content_hash='x')]:
        with pytest.raises(ValueError,match='overlap'):m.no_overlap(train,[r])


def test_retention_same_modality_contract():
    m=mod();src={'kind':'paired','mask':'post','classes':3,'domain':'bright-hawaii-wildfire'}
    target=m.eval_unit(src,'bright-libya-flood')
    assert target['mask']=='post' and target['classes']==3
    with pytest.raises(ValueError,match='modality'):
        m.eval_unit(src,'pathmnist')
    src={'kind':'single','mask':'all_visual','classes':2,'domain':'camelyon17-hospital_2'}
    assert m.eval_unit(src,'pathmnist')['classes']==9


def test_text_preparation_no_label_in_prompt():
    m=mod('text')
    arc=m.normalize('arc_challenge',[{'id':'x','question':'Which?', 'choices':{'label':['A','B'],'text':['foo','bar']},'answerKey':'B'}])
    assert arc[0]['label_id']==1 and arc[0]['candidates']==['foo','bar']
    assert 'answerKey' not in arc[0]['question']
    b=m.normalize('boolq',[dict(question='Is it?',passage='A fact.',answer=True)])
    assert b[0]['candidates']==['no','yes'] and b[0]['label_id']==1
    assert b[0]['question'].startswith('Passage: A fact.')


def test_text_duplicate_and_bad_key_fail():
    m=mod('text');r={'id':'x','question':'Q','choices':{'label':['A','B'],'text':['a','b']},'answerKey':'C'}
    with pytest.raises(ValueError):m.normalize('arc_challenge',[r])
    r['answerKey']='A'
    with pytest.raises(ValueError):m.normalize('arc_challenge',[r,r])


def test_full_text_counts():
    m=mod('text')
    assert m.DATASETS['arc_challenge']['count']==1172
    assert m.DATASETS['boolq']['count']==3270
    with pytest.raises(ValueError,match='population'):m.check_count('boolq',[{}])


def test_label_free_payload():
    m=mod('text');a={'sample_id':'s','question':'Q','candidates':['no','yes'],'label_id':1,'group_id':'s','answers':['yes']}
    assert m.public_row(a)=={'question':'Q','candidates':['no','yes']}


def test_no_image_text_zero_mask_and_summed_scores():
    import torch
    from types import SimpleNamespace
    m=mod('text')
    class Tok:
        def apply_chat_template(self,*a,**kw):return 'P:'
        def __call__(self,s,**kw):
            return {'input_ids':torch.tensor([[1,2]+[3]*len(s[2:])]),'attention_mask':torch.ones(1,len(s),dtype=torch.long)}
    class C:
        def __init__(self):self.masks=[];self.mask=None
        def set_mask(self,x):self.mask=x;self.masks.append(x.clone())
        def clear_mask(self):self.mask=None
    c=C()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.config=SimpleNamespace(image_token_id=99)
        def forward(self,input_ids,attention_mask,**kw):
            assert 'pixel_values' not in kw
            assert c.mask is not None and not c.mask.any()
            return SimpleNamespace(logits=torch.zeros(1,input_ids.shape[1],8))
    out=m.score_text(Model(),Tok(),{'question':'Q','candidates':['a','bbb'],'label_id':1},c)
    assert out['scores'][0]==pytest.approx(-np.log(8))
    assert out['scores'][1]==pytest.approx(-3*np.log(8))
    assert len(c.masks)==2 and c.mask is None


def test_zero_gate_restored_on_forward_error():
    import torch
    from types import SimpleNamespace
    m=mod('text')
    class T:
        def apply_chat_template(self,*a,**k):return 'P'
        def __call__(self,s,**k):return {'input_ids':torch.ones((1,len(s)),dtype=torch.long)}
    class C:
        def set_mask(self,x):self.mask=x
        def clear_mask(self):self.mask=None
    c=C()
    class M(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.config=SimpleNamespace()
        def forward(self,**kw):raise RuntimeError('fixture')
    with pytest.raises(RuntimeError,match='fixture'):m.score_text(M(),T(),{'question':'Q','candidates':['a','b']},c)
    assert c.mask is None


def test_plan_parts_and_method_sets():
    m=mod()
    bank=m.job_specs('bank')
    assert all(len(x['available_domains'])==x['stage'] for x in bank)
    assert all('lora-r4' not in x['methods'] for x in bank)
    assert len([x for x in bank if x['stage']==4])==6


def test_output_must_not_overlap_training(tmp_path):
    m=mod();repo=tmp_path/'repo';repo.mkdir();old=repo/'runs/old';old.mkdir(parents=True)
    for out in [repo,repo/'src/new',old,old/'child',repo/'runs']:
        with pytest.raises(ValueError):m.safe_root(repo,out,[old])
    m.safe_root(repo,repo/'runs/e10',[old])


def test_query_labels_are_not_sent_to_native_backend():
    m=mod('worker')
    row=dict(sample_id='a',group_id='p',image='i',question='Q',candidate_labels=['a','b'],label_id=1,label='b',metadata={'answer':'b'})
    public=m.public_visual(row)
    assert public['label_id']==0 and public['label']=='a' and 'metadata' not in public
    assert row['label_id']==1


def test_prediction_prefix_and_truth_validation():
    m=mod('worker')
    rows=[dict(sample_id='a',group_id='g',label_id=1,candidate_labels=['a','b'])]
    pred=[dict(sample_id='a',group_id='g',label_id=1,scores=[0.,1.],prompt_contract_hash='h')]
    m.validate_predictions(pred,rows)
    m.validate_predictions([],rows,complete=False)
    with pytest.raises(ValueError):m.validate_predictions([],rows)
    with pytest.raises(ValueError):m.validate_predictions([dict(pred[0],label_id=0)],rows)


def test_bank_storage_includes_bases_and_adapter_files(tmp_path):
    m=mod('worker');f=tmp_path/'fit';f.mkdir()
    (f/'controller.pt').write_bytes(b'base-and-R')
    (f/'deployment.json').write_text('{}')
    (f/'fit.json').write_text('not-state')
    assert m.state_bytes(f)==len(b'base-and-R')+2
    (f/'adapter').mkdir();(f/'adapter/config.json').write_text('123')
    assert m.state_bytes(f)==len(b'base-and-R')+5


def test_merged_report_refuses_partial_cells_and_keeps_missing(tmp_path):
    m=mod('report')
    plan={'part':'visual','jobs':[{'key':'a'},{'key':'b'}], 'settings':{'run_root':str(tmp_path)}}
    status=m.coverage(plan)
    assert status['complete'] is False and len(status['missing'])==2
    with pytest.raises(ValueError,match='Incomplete'):m.aggregate(plan)
    assert not (tmp_path/'visual/report/results.json').exists()


def test_method_report_keeps_gains_and_retention_separate():
    m=mod('report')
    results={'source':{'aperture':{'accuracy_delta':.2}},'old':{'aperture':{'accuracy_delta':-.1}}}
    flat=m.flatten_visual('u',results,'source')
    assert [x['role'] for x in flat]==['new_domain','retention']
    assert flat[1]['accuracy_delta']==-.1


def test_cli_list_and_dry_run_need_no_gpu():
    import subprocess
    p=ROOT/'scripts/run_aperture_retention.py'
    assert p.exists(),'E10 CLI missing'
    for args in [['--help'],['list']]:
        r=subprocess.run([sys.executable,str(p),*args],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
    r=subprocess.run([sys.executable,str(p),'run','--part','visual','--gpu','0,1'],capture_output=True,text=True)
    assert r.returncode!=0


def test_original_aperture_not_replaced_by_tuned_variant():
    m=mod()
    assert hasattr(m,'validate_variant'),'source-method identity validation missing'
    v=dict(engine='aperture',rank=16,layers=[14,27],alpha=3.,steps=4,lr=.05,l2=.001,
           objective='token_mean',sites=['K','V'],mode='full',basis_mode='covariance')
    m.validate_variant('aperture',v)
    for changes in [dict(alpha=1.),dict(mode='unbounded'),dict(objective='candidate_ce')]:
        with pytest.raises(ValueError):m.validate_variant('aperture',{**v,**changes})


def test_collect_resumes_exact_prefix_and_refuses_changed_labels(tmp_path):
    m=mod('worker');r=[dict(sample_id=str(i),group_id='g',label_id=0,candidate_labels=['a','b']) for i in range(3)]
    p=tmp_path/'pred.jsonl';calls=[]
    def score(row):calls.append(row['sample_id']);return {'scores':[1.,0.],'prompt_contract_hash':'h'}
    x=m.collect(p,r,score);assert len(x)==3 and len(calls)==3
    m.collect(p,r,score);assert len(calls)==3
    r[0]['label_id']=1
    with pytest.raises(ValueError):m.collect(p,r,score)


def test_readonly_sources_can_be_published_without_invalidation():
    m=mod();from aperture_rebuttal_cases.store import sources
    files=sources(ROOT)
    assert not any('aperture_retention' in k for k in files)
    assert all(not k.startswith('src/') for k in m.own_sources(ROOT))


def test_source_snapshot_required_even_for_empty_subset(tmp_path):
    m=mod();folder=tmp_path/'src';folder.mkdir();(folder/'x.py').write_text('x=0\n')
    with pytest.raises(ValueError,match='source'):m.verify_source({},tmp_path)


def test_plan_freezes_full_scope_and_detects_query_change(tmp_path):
    m=mod();from aperture_rebuttal_cases import store
    repo=tmp_path/'repo';(repo/'src').mkdir(parents=True);(repo/'src/x.py').write_text('x=1\n')
    (repo/'scripts/aperture_retention').mkdir(parents=True)
    (repo/'scripts/run_aperture_retention.py').write_text('# launcher')
    (repo/'scripts/aperture_retention/__init__.py').write_text('')
    source_files={'src/x.py':store.sha(repo/'src/x.py')};paths=[]
    for case in ('E01','E02'):
        units=[]
        for j in m.job_specs('visual'):
            units.append({'key':j['key'],'model_key':j['model'],'domain':j['domain'],'support_seed':j['seed']})
        p=repo/f'runs/old/{case}/plan.json';store.atomic(p,{'case':case,'source_files':source_files,'units':units,'settings':{'run_root':str(repo/'runs/old')}});paths.append(str(p))
    names=set(m.DOMAINS)|{x for ts in m.TARGETS.values() for x in ts}
    base={'datasets':[{'key':x} for x in sorted(names)],'prepared_root':'data/prepared'}
    bp=repo/'configs/base.json';store.atomic(bp,base)
    for x in names:
        q=repo/'data/prepared'/x/'query.jsonl';q.parent.mkdir(parents=True);q.write_text('{}\n')
    cfg={'protocol':'aperture-retention-v1','source_plans':paths,'run_root':str(repo/'runs/e10'),
         'base_config':str(bp),'text_root':str(repo/'data/text')}
    a=m.freeze(repo,cfg,'visual');b=m.freeze(repo,cfg,'visual');assert a==b and len(a['jobs'])==18
    m.verify_plan(a)
    (repo/'data/prepared/pathmnist/query.jsonl').write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match='changed'):m.verify_plan(a)
    with pytest.raises(ValueError,match='Frozen'):m.freeze(repo,cfg,'visual')


def test_cli_prepare_text_without_execute_is_dry_run():
    import subprocess
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_aperture_retention.py'),'prepare-text'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='dry_run'
