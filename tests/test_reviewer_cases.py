"""No-network regressions for case isolation and evidence correctness."""
from pathlib import Path
import importlib
import json
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def module(name):
    try:
        return importlib.import_module('aperture_rebuttal_cases.' + name)
    except ModuleNotFoundError:
        pytest.fail('Required reviewer-case module is not implemented: ' + name)


def test_catalog_covers_every_review_question_and_excludes_new_method():
    p = module('plan')
    assert set(p.CASES) == {f'E{i:02}' for i in range(1, 10)}
    assert set(p.REVIEW_MAP) == {'R1.1','R1.2','R1.3','R1.4','R2.1','R2.2','R2.3',
                                'R2.4','R3.1','R3.2','R3.3','R3.4','R3.5'}
    for q in p.REVIEW_MAP.values():
        assert set(q['cases']) <= set(p.CASES)
        assert q['writing']
    assert 'not established' in p.REVIEW_MAP['R2.3']['limit']
    assert p.FIXED['rank'] == 16 and p.FIXED['alpha'] == 3
    assert 'q38' not in p.MODELS


def test_variants_keep_original_and_have_real_baselines():
    p = module('plan')
    main = p.variants('E01', 'paired')
    assert {x['name'] for x in main} == {'frozen','lora1','lora4','random','aperture'}
    assert next(v for v in main if v['name']=='aperture')['lr'] == .05
    strong = p.variants('E02', 'single')
    assert {'lora-r1','lora-r4','lora-r16','dora-r4','loreft-r4','aperture'} <= {x['family'] for x in strong}
    for family in ('lora-r1','lora-r4','lora-r16','dora-r4','loreft-r4','aperture'):
        assert len([v for v in strong if v['family']==family]) == 4
    assert all(v.get('objective','token_mean') == 'token_mean' for v in strong)


def test_case_outputs_and_variant_ids_are_unique(tmp_path):
    p = module('plan')
    dataset = {'key':'bright-hawaii-wildfire','kind':'paired','classes':3,'query_count':300}
    base = {'prepared_root':str(tmp_path/'prepared'), 'models':[{'key':m,'path':f'models/{m}'} for m in p.MODELS],
            'datasets':[dataset], 'support_seeds':[0,1,2]}
    a = p.build_units('E01', base, tmp_path/'run', tmp_path)
    b = p.build_units('E02', base, tmp_path/'run', tmp_path)
    assert len(a)==15 and len(b)==6
    assert not {x['output'] for x in a}&{x['output'] for x in b}
    for u in a+b:
        ids = [v['name'] for v in u['variants']]
        assert len(ids)==len(set(ids))
        assert u['query_count']==300


def test_seal_rejects_tamper_and_traversal(tmp_path):
    m = module('store')
    (tmp_path/'a.txt').write_text('original')
    m.seal(tmp_path, {'id':'cell'})
    assert m.verify(tmp_path, {'id':'cell'})['binding']=={'id':'cell'}
    (tmp_path/'a.txt').write_text('altered')
    with pytest.raises(ValueError, match='changed'):
        m.verify(tmp_path, {'id':'cell'})
    (tmp_path/'DONE.json').write_text(json.dumps({'binding':{'id':'cell'},'files':{'../a':'x'}}))
    with pytest.raises(ValueError): m.verify(tmp_path, {'id':'cell'})


def test_lock_releases_on_exception_and_blocks_other_owner(tmp_path):
    m=module('store')
    with pytest.raises(RuntimeError):
        with m.lock(tmp_path/'owner.lock'):
            with pytest.raises(RuntimeError):
                with m.lock(tmp_path/'owner.lock'): pass
            raise RuntimeError('abort')
    with m.lock(tmp_path/'owner.lock'): pass


def test_calibration_selection_has_no_query_input_and_conservative_ties():
    m=module('report')
    records=[{'name':'a','family':'aperture','calibration_nll':.5},
             {'name':'b','family':'aperture','calibration_nll':.4},
             {'name':'c','family':'lora','calibration_nll':.4}]
    assert m.select(records)=={'aperture':'b','lora':'c'}
    records[0]['calibration_nll']=float('nan')
    with pytest.raises(ValueError):m.select(records)


def rows(n, group_prefix='g'):
    return [{'sample_id':str(i),'group_id':f'{group_prefix}{i//2}','content_hash':f'h{i}',
             'label_id':i%2,'label':['normal','tumor'][i%2],
             'candidate_labels':['normal','tumor'],'question':'Which?','image':f'{i}.png'} for i in range(n)]


def test_nested_support_does_not_move_calibration_to_fit():
    m=module('data')
    fit=rows(100,'f');cal=rows(100,'c')
    for r in cal:r['sample_id']='c'+r['sample_id'];r['content_hash']='c'+r['content_hash']
    a=m.nested(fit,cal,8,0,2);b=m.nested(fit,cal,16,0,2)
    assert {r['sample_id'] for r in a['support']} <= {r['sample_id'] for r in b['support']}
    assert {r['sample_id'] for r in a['calibration']} <= {r['sample_id'] for r in b['calibration']}
    assert len(a['support'])==12 and len(a['calibration'])==4
    assert not {r['sample_id'] for r in b['support']}&{r['sample_id'] for r in b['calibration']}


def test_leakage_and_insufficient_budget_fail():
    m=module('data')
    with pytest.raises(ValueError):m.nested(rows(4),rows(4),32,0,2)
    fit=rows(4);cal=rows(4,'c');q=rows(4,'q')
    with pytest.raises(ValueError):m.validate({'support':fit,'calibration':cal,'query':q},True)


def test_missing_state_is_never_imputed_as_zero(tmp_path):
    m=module('report')
    result=m.coverage([tmp_path/'missing-a', tmp_path/'missing-b'])
    assert result['completed']==0 and not result['complete']
    assert len(result['missing'])==2


def test_paired_cluster_bootstrap_matches_units_and_is_reproducible():
    m=module('report')
    a=[{'sample_id':str(i),'group_id':str(i//2),'label_id':0,'prediction':1} for i in range(6)]
    b=[{**r,'prediction':0} for r in a]
    x=m.paired_accuracy(a,b,replicates=40)
    assert x['difference']==1 and x['ci95']==[1.,1.] and x['groups']==3
    assert x==m.paired_accuracy(a,b,replicates=40)
    with pytest.raises(ValueError):m.paired_accuracy(a,b[:-1],replicates=40)


def test_error_examples_are_not_best_score_selected():
    m=module('report')
    a=[{'sample_id':str(i),'group_id':str(i),'label_id':0,'prediction':i%2} for i in range(20)]
    b=[{**r,'prediction':1-r['prediction']} for r in a]
    out=m.error_examples(a,b,limit=2)
    assert len(out['fixed'])==len(out['regressed'])==2
    assert out==m.error_examples(a,b,limit=2)


def test_raw_and_candidate_metrics_are_finite():
    m=module('report')
    x=[{'sample_id':'a','label_id':0,'probabilities':[.8,.2]},
       {'sample_id':'b','label_id':1,'probabilities':[.4,.6]}]
    y=m.metrics(x)
    assert y['accuracy']==1 and y['macro_f1']==1 and y['nll']>0
    with pytest.raises(ValueError):m.metrics([{**x[0],'probabilities':[float('nan'),0.]}])


def test_vqa_query_prompt_is_label_free_and_no_candidates():
    m=module('generation')
    row={'question':'What is written?','answers':['secret answer']*10,'label':'secret answer',
         'candidate_labels':['secret answer'],'image':'picture.png'}
    prompt=m.question(row)
    assert 'secret answer' not in prompt and 'What is written?' in prompt


def test_composition_and_loss_mix_are_not_enabled():
    p=module('plan')
    for case in p.CASES:
        for v in p.variants(case,'single'):
            assert not v.get('mix') and v.get('objective','token_mean')=='token_mean'


def test_filters_reject_empty_or_unknown_selection():
    m=module('cli')
    us=[{'key':'a','model_key':'q25','domain':'d','support_seed':0}]
    assert m.filter_units(us,None,'q25',None,None)==us
    with pytest.raises(ValueError):m.filter_units(us,None,'phi',None,None)
    with pytest.raises(ValueError):m.filter_units(us,'unknown',None,None,None)


def test_cli_list_does_not_require_models_or_data(capsys):
    m=module('cli')
    assert m.main(['list'])==0
    assert 'E07' in capsys.readouterr().out


def test_operator_initial_identity_and_matching_frobenius_bounds():
    import torch
    m=module('adapters')
    for mode,shape in [('full',(4,4)),('diagonal_matched',(4,)),('unbounded',(4,4))]:
        raw=torch.zeros(shape,requires_grad=True)
        a=m.ablation_operator(raw,3.,mode)
        assert torch.equal(a,torch.zeros(4,4))
        a.sum().backward();assert raw.grad.abs().sum()>0
        if mode!='unbounded':assert m.ablation_operator(torch.ones(shape)*100,3.,mode).norm()<3


def test_masked_ablation_preserves_unselected_rows_and_roundtrips():
    import torch
    m=module('adapters')
    target=torch.nn.Linear(6,6,bias=False);base=torch.eye(6)[:,:2]
    for mode in ('diagonal_matched','unbounded'):
        c=m.MaskedOperator([(0,'K',target)],{(0,'K'):base},3.,mode)
        try:
            c.set_mask(torch.tensor([[False,True,False]]))
            with torch.no_grad():c.raw['0:K'].fill_(.1)
            z=torch.randn(1,3,6);out=c._hook(0,'K')(target,None,z)
            assert torch.equal(out[:,[0,2]],z[:,[0,2]])
            assert not torch.equal(out[:,1],z[:,1])
            saved=c.payload()
            with torch.no_grad():c.raw['0:K'].zero_()
            c.restore(saved)
            assert torch.equal(c._hook(0,'K')(target,None,z),out)
        finally:c.close()


def test_evaluation_replay_fails_before_query_on_score_change():
    m=module('engine')
    before=[{'sample_id':'x','mean_log_scores':[-1.,-2.]}]
    after=[{'sample_id':'x','mean_log_scores':[-1.,-2.1]}]
    with pytest.raises(RuntimeError,match='replay'):m.compare_replay(before,after)
    assert m.compare_replay(before,before)==0


def test_baseline_family_must_not_overwrite_native_architecture():
    m=module('adapters')
    assert hasattr(m,'native_job'), 'Separate method family from architecture family'
    job=m.native_job({'family':'qwen2_5_vl','model_key':'q25'}, {'family':'dora-r4','engine':'dora'})
    assert job['family']=='qwen2_5_vl' and job['arm']=='frozen'


def test_plan_rejects_duplicate_output_ids(tmp_path):
    p=module('plan')
    d={'key':'bright-hawaii-wildfire','kind':'paired','classes':3,'query_count':300}
    base={'prepared_root':'data/prepared','models':[{'key':m,'path':m} for m in p.MODELS],
          'datasets':[d,d]}
    with pytest.raises(ValueError,match='duplicate'):
        p.build_units('E01',base,tmp_path/'run',tmp_path)


def test_generation_masks_only_image_prefix_and_never_reads_answers():
    import torch
    from types import SimpleNamespace
    m=module('generation')
    seen=[]
    class Model(torch.nn.Module):
        generation_config=SimpleNamespace(eos_token_id=2)
        def forward(self,input_ids,attention_mask,pixel_values,token_type_ids,use_cache=False,return_dict=True,logits_to_keep=1):
            assert use_cache is False
            seen.append(input_ids.clone())
            logits=torch.full((1,1,8),-10.);logits[0,0,5 if input_ids.shape[1]==3 else 2]=10.
            return SimpleNamespace(logits=logits)
    class Processor:
        eos_token_id=2
        def decode(self,ids,skip_special_tokens=True):return 'five' if ids==[5] else 'bad'
    class Backend:
        model=Model();base=model;processor=Processor()
        def batch(self,row,candidate):
            assert 'answers' not in row and row['label']=='*' and row['candidate_labels']==['*']
            return {'input_ids':torch.tensor([[1,7,3,6]]),'attention_mask':torch.ones(1,4,dtype=torch.long),
                    'token_type_ids':torch.tensor([[0,1,0,0]]),'pixel_values':torch.ones(1)},(3,4),torch.tensor([[False,True,False,False]]),{}
    class Controller:
        masks=[]
        def set_mask(self,x):self.masks.append(x.clone())
        def clear_mask(self):pass
    c=Controller()
    row={'sample_id':'1','group_id':'g','image':'x','question':'What text?',
         'answers':['hidden']*10,'label':'hidden'}
    out=m.generate(Backend(),row,c)
    assert out['pred_answer']=='five' and out['stopped_eos']
    assert [x.shape[1] for x in c.masks]==[3,4]
    assert all(int(x.sum())==1 for x in c.masks)
    assert seen[0].tolist()==[[1,7,3]]


def test_freezing_preserves_config_and_rejects_source_drift(tmp_path):
    p=module('plan');s=module('store')
    repo=tmp_path/'repo';(repo/'src/a').mkdir(parents=True);(repo/'src/a/x.py').write_text('x=1\n')
    base={'run_root':'runs/old','prepared_root':'data/old','models':[{'key':m,'path':f'models/{m}'} for m in p.MODELS],
          'datasets':[{'key':'d','kind':'single','classes':2,'query_count':8}]}
    s.atomic(repo/'configs/base.json',base)
    cfg={'base_config':'configs/base.json','run_root':'runs/new','prepared_root':'data/prepared/new'}
    first=p.freeze('E01',cfg,repo)
    assert p.freeze('E01',cfg,repo)==first
    (repo/'src/a/x.py').write_text('x=2\n')
    with pytest.raises(ValueError,match='changed'):p.freeze('E01',cfg,repo)


def test_stale_scientific_export_is_revoked_before_error(tmp_path):
    m=module('report');s=module('store')
    root=tmp_path/'run';u={'key':'a','output':str(root/'E01/units/a'),'variants':[]}
    plan={'case':'E01','settings':{'run_root':str(root)},'units':[u]}
    folder=Path(u['output']);s.atomic(folder/'selection.json',{'evaluate':[]})
    out=root/'E01/report/results.json';s.atomic(out,{'status':'complete','old':True})
    with pytest.raises((ValueError,OSError,KeyError)):m.summarize(plan)
    assert not out.exists()


def test_lora_and_dora_receive_declared_rank_not_hardcoded_16():
    m=module('adapters')
    assert hasattr(m,'lora_options')
    for rank in (1,4,16):
        v={'engine':'dora','lora_rank':rank,'lora_alpha':2*rank,'lora_dropout':.05}
        options=m.lora_options(v,['decoder.q_proj'])
        assert options['r']==rank and options['lora_alpha']==2*rank and options['use_dora'] is True


def test_serialized_stage_job_does_not_duplicate_full_matrix(tmp_path):
    m=module('cli')
    assert hasattr(m,'task_reference')
    cfg={'run_root':str(tmp_path)}
    plan={'settings':cfg,'case':'E01','units':[{'padding':'x'*10000}]*1000}
    (tmp_path/'E01').mkdir();(tmp_path/'E01/plan.json').write_text('{}')
    task=m.task_reference(plan,{'key':'q25--d--n16--s0'},None,'audit',tmp_path)
    assert len(json.dumps(task))<2048 and 'plan' not in task


def test_nll_is_unclipped_even_when_probability_underflows():
    m=module('report')
    data=[{'sample_id':'x','label_id':0,'mean_log_scores':[-1000.,0.],
           'probabilities':[0.,1.],'prediction':1}]
    assert m.metrics(data)['nll']==pytest.approx(1000.)


def test_prediction_field_must_agree_with_candidate_scores():
    m=module('report')
    data=[{'sample_id':'x','label_id':0,'mean_log_scores':[0.,-1000.],
           'probabilities':[1.,0.],'prediction':1}]
    with pytest.raises(ValueError):m.metrics(data)
