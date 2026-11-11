"""Synthetic integration: contracts and control flow, NOT native VLM evidence."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import json
import sys
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_conditional_gated import adapters, audit, cli, engine, plan, report
from aperture_rebuttal_cases import store, data


def fake_module(monkeypatch,name,**attrs):
    m=ModuleType(name);m.__dict__.update(attrs);monkeypatch.setitem(sys.modules,name,m);return m


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(8,8,bias=False)
        with torch.no_grad():self.proj.weight.copy_(torch.eye(8))
        self.proj.weight.requires_grad_(False)


class ToyBackend:
    def __init__(self,model,leak=False):
        self.model=model;self.processor=object();self.leak=leak;self.calls=[]
    def candidate(self,row,candidate,controller=None):
        if candidate not in row['candidate_labels']:raise ValueError('unknown candidate')
        self.calls.append(row['sample_id'])
        x=torch.arange(32,dtype=torch.float32).reshape(1,4,8)/20-.7
        if self.leak and candidate!=row['label']:x=x.flip(-1)
        mask=torch.tensor([[False,True,True,False]])
        if controller is not None:controller.set_mask(mask)
        z=self.model.proj(x);value=(z[mask]*torch.arange(1.,9.)).mean()
        probs=torch.log_softmax(torch.stack([value,-value]),0)
        idx=0 if candidate==row['label'] else 1
        return probs[idx:idx+1],mask,{'image_tokens':2,'prefix':'same'}
    def loss_and_mask(self,row,controller=None):
        p,m,_=self.candidate(row,row['label'],controller);return -p.mean(),m


def toy_roles(vqa=False):
    rows=[{'sample_id':str(i),'group_id':f'g{i}','label':f'a{i}',
           'candidate_labels':[f'a{i}'] if vqa else ['a0','a1'],
           'label_id':i,'question':'Look','image':'fake'} for i in (0,1)]
    return {'support':rows,'calibration':rows,'query':[]}


@pytest.mark.parametrize('vqa',[False,True])
def test_native_audit_contract_including_single_target_vqa(monkeypatch,tmp_path,vqa):
    model=ToyModel();backend=ToyBackend(model)
    @contextmanager
    def native(job):yield model,backend,None
    fake_module(monkeypatch,'aperture_comprehensive.backend',loaded=native,decoder_sites=lambda *a:[(1,'K',model.proj)])
    monkeypatch.setattr(engine,'setup',lambda:torch);monkeypatch.setattr(engine,'binding',lambda *a:{'test':'synthetic'})
    monkeypatch.setattr(audit,'load_roles',lambda *a,**kw:toy_roles(vqa))
    unit={'output':str(tmp_path/'unit'),'variants':[], 'layers':[1], 'rank':3,'alpha':3.,
          'task_type':'vqa' if vqa else 'classification','classes':2}
    audit.audit(unit,{})
    assert store.read(tmp_path/'unit/audit/audit.json')['passed']
    assert not model.proj._forward_hooks
    assert set(backend.calls)<= {'0','1'}


def test_audit_rejects_gate_dependence_on_candidate(monkeypatch,tmp_path):
    model=ToyModel();backend=ToyBackend(model,leak=True)
    @contextmanager
    def native(job):yield model,backend,None
    fake_module(monkeypatch,'aperture_comprehensive.backend',loaded=native,decoder_sites=lambda *a:[(1,'K',model.proj)])
    monkeypatch.setattr(engine,'setup',lambda:torch);monkeypatch.setattr(engine,'binding',lambda *a:{'x':1})
    monkeypatch.setattr(audit,'load_roles',lambda *a,**kw:toy_roles())
    unit={'output':str(tmp_path),'variants':[],'layers':[1],'rank':3,'alpha':3.,'task_type':'classification','classes':2}
    with pytest.raises(RuntimeError,match='depends on future candidate'):audit.audit(unit,{})
    assert not (tmp_path/'audit/DONE.json').exists()
    assert not model.proj._forward_hooks


def test_gated_loader_extracts_only_support_and_roundtrips(monkeypatch,tmp_path):
    model=ToyModel();backend=ToyBackend(model);calls=[]
    @contextmanager
    def native(job):
        assert job['family']=='toy' and job['arm']=='frozen'
        yield model,backend,None
    def extract(model,sites,support,*args):
        calls.extend(r['sample_id'] for r in support)
        return {(1,'K'):torch.eye(8)[:,:3]},{'fake':True}
    fake_module(monkeypatch,'aperture_comprehensive.backend',loaded=native,decoder_sites=lambda *a:[(1,'K',model.proj)])
    fake_module(monkeypatch,'vigor_handoff.core',extract_basis=extract)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(data,'load_roles',lambda *a,**kw:toy_roles())
    unit={'family':'toy','layers':[1],'task_type':'classification'}
    v={'engine':'conditional_gated','family':'conditional-gated','sites':['K'],'layers':[1],
       'rank':3,'basis_mode':'covariance','basis_seed':0,'alpha':3.,'gate_mode':'conditional','gate_eps':1e-6}
    with adapters.loaded(unit,v) as (_,b,c):
        assert b.processor is backend.processor
        with torch.no_grad():c.raw['1:K'].fill_(.1);c.gate_weight['1:K'].fill_(.2)
        old=b.candidate(toy_roles()['support'][0],'a0',c)[0].detach()
        torch.save(c.payload(),tmp_path/'controller.pt')
    with adapters.loaded(unit,v,tmp_path) as (_,b,c):
        now=b.candidate(toy_roles()['support'][0],'a0',c)[0].detach()
        assert torch.equal(old,now)
    assert calls==['0','1'] # restore does not re-extract from any labels
    payload=torch.load(tmp_path/'controller.pt',weights_only=True);payload.pop('format');torch.save(payload,tmp_path/'controller.pt')
    with pytest.raises(ValueError,match='Not a CG'):
        with adapters.loaded(unit,v,tmp_path):pass
    assert not model.proj._forward_hooks


@pytest.fixture
def cli_fixture(tmp_path,monkeypatch):
    from test_conditional_gated import make_base
    repo=tmp_path/'repo';(repo/'configs').mkdir(parents=True)
    store.atomic(repo/'configs/base.json',make_base())
    cfg={'method_version':plan.VERSION,'base_config':'configs/base.json','run_root':'runs/conditional_gated_fixture',
         'prepared_root':'data/prepared','medical_data_root':'data/raw','split_seed':271107,
         'textvqa_train_json':'data/train.json','textvqa_val_json':'data/val.json','textvqa_images':'data/images','execution_batch':1}
    store.atomic(repo/'configs/cg.json',cfg);monkeypatch.setattr(cli,'REPO',repo)
    monkeypatch.delenv('APERTURE_BATCH',raising=False)
    return repo,cfg


@pytest.mark.parametrize('case',[f'CG{i:02}' for i in range(10)])
def test_cli_each_case_plan_isolated(cli_fixture,capsys,case):
    repo,cfg=cli_fixture
    assert cli.main(['plan','--case',case,'--config','configs/cg.json'])==0
    result=json.loads(capsys.readouterr().out)
    saved=store.read(repo/cfg['run_root']/case/'plan.json')
    assert saved['method_changed'] and not saved['prior_best_reproduced']
    assert all(u['case']==case for u in saved['units'])


def test_cli_dry_run_does_not_load_model(cli_fixture,capsys,monkeypatch):
    monkeypatch.setattr(cli,'execute',lambda *a:pytest.fail('Dry run executed'))
    assert cli.main(['run','--case','CG02','--config','configs/cg.json','--unit','q25--camelyon17-hospital_2--n16--s0'])==0
    assert json.loads(capsys.readouterr().out)['status']=='dry_run'


def test_changed_plan_batch_or_root_is_rejected(cli_fixture,capsys,monkeypatch):
    repo,cfg=cli_fixture
    assert cli.main(['plan','--case','CG02','--config','configs/cg.json'])==0;capsys.readouterr()
    cfg['execution_batch']=2;store.atomic(repo/'configs/cg.json',cfg)
    assert cli.main(['plan','--case','CG02','--config','configs/cg.json'])==2
    cfg['run_root']='runs/old';cfg['execution_batch']=1;store.atomic(repo/'configs/cg.json',cfg)
    assert cli.main(['plan','--case','CG02','--config','configs/cg.json'])==2
    assert cli.main(['run','--case','CG02','--config','configs/cg.json','--gpu','0,1'])==2


def test_incomplete_summary_blocks_and_stamps_new_method(cli_fixture,capsys):
    repo,cfg=cli_fixture
    assert cli.main(['summarize','--case','CG02','--config','configs/cg.json'])==2
    state=store.read(repo/cfg['run_root']/'CG02/report/coverage.json')
    assert state['method_changed'] and state['method_version']==plan.VERSION
    assert not (repo/cfg['run_root']/'CG02/report/summary.json').exists()


def test_budget_filter_keeps_three_budgets(cli_fixture,capsys):
    assert cli.main(['run','--case','CG03','--config','configs/cg.json','--model','q25','--domain','hospital_2','--seed','0'])==0
    assert len(json.loads(capsys.readouterr().out)['units'])==3
    assert cli.main(['run','--case','CG03','--config','configs/cg.json','--unit','q25--hospital_2--n32--s0'])==0
    assert len(json.loads(capsys.readouterr().out)['units'])==1


def test_real_adam_updates_gate_and_raw_on_toy_support(monkeypatch):
    from aperture_conditional_gated.controller import ConditionalGatedController
    m=ToyModel();c=ConditionalGatedController([(1,'K',m.proj)],{(1,'K'):torch.eye(8)[:,:3]})
    b=ToyBackend(m)
    class Batched:
        def candidate_batch(self,rs,candidates,control):
            lps=[b.candidate(r,r['label'],control)[0].sum() for r in rs]
            return lps,None,{},[{'span':(4,5)} for _ in rs]
    monkeypatch.setenv('APERTURE_BATCH','2')
    before=m.proj.weight.detach().clone()
    history=engine.fit_controller_batched(m,Batched(),c,toy_roles()['support'],{'steps':4,'lr':.05,'l2':.001,'loss_reduction':'mean'})
    assert len(history)==4
    assert c.raw['1:K'].detach().abs().sum()>0
    assert c.gate_weight['1:K'].detach().abs().sum()>0
    assert torch.equal(before,m.proj.weight)
    assert c.operators()['1:K'].norm()<3
    c.close()


def make_sealed_report(tmp_path,vqa=False):
    import numpy as np
    root=tmp_path/'runs/conditional_gated_fixture';units=[]
    cfg={'run_root':str(root),'base_config':str(tmp_path/'configs/base.json')}
    for budget in (16,32):
        for seed in range(3):
            key=f'q25--d--n{budget}--s{seed}';folder=root/'CG03/units'/key
            roles={}
            for role in ('support','calibration','query'):
                path=tmp_path/f'{key}-{role}.jsonl'
                refs=[{'sample_id':str(i),'answers':['answer']*10} for i in range(2)]
                store.write_rows(path,refs);roles[role]=str(path)
            v={'name':f'conditional-gated-a{seed+1}','family':'conditional-gated'}
            units.append({**roles,'key':key,'output':str(folder),'model_key':'q25','domain':'d',
                          'total_labels':budget,'support_seed':seed,'task_type':'vqa' if vqa else 'classification',
                          'variants':[v],'evaluate_policy':'selected'})
    p={'version':plan.VERSION,'method_changed':True,'case':'CG03','query_status':'synthetic test data',
       'settings':cfg,'units':units}
    for u in units:
        rootu=Path(u['output']);v=u['variants'][0];fit=rootu/'fits'/v['name']
        store.atomic(fit/'fit.json',{'serialized_state_bytes':100,'fit_seconds_including_extraction':1.,
            'optimizer_updates':4,'support_visits':16,'basis_visits':4,'history':[],
            'final_support_metrics':{},'final_calibration_metrics':{}})
        store.atomic(fit/'gate_diagnostics.json',{})
        bound={'unit':u,'variant':v,'plan':store.digest(p),'manifests':{k:store.sha(u[k]) for k in ('support','calibration','query')}}
        store.seal(fit,bound)
        sel={'evaluate':[v['name']],'selected':{'conditional-gated':v['name']},'fit_seals':{v['name']:store.sha(fit/'DONE.json')}}
        store.atomic(rootu/'selection.json',sel);store.atomic(rootu/'selection_receipt/selection.json',sel)
        store.seal(rootu/'selection_receipt',{'unit':u['key'],'plan':store.digest(p)})
        dest=rootu/'evaluation'/v['name']
        if vqa:
            predictions=[{'sample_id':str(i),'pred_answer':'answer'} for i in range(2)]
            metric={'vqa_accuracy':.6,'n':2}
        else:
            scores=np.array([-1.,-2.]);probs=np.exp(scores-scores.max());probs/=probs.sum()
            predictions=[{'sample_id':str(i),'group_id':str(i),'label_id':i,'prediction':0,
                          'mean_log_scores':scores.tolist(),'probabilities':probs.tolist()} for i in range(2)]
            raw=report.metrics(predictions)
            scaled=[{**r,'mean_log_scores':[0.,-1.]} for r in predictions]
            metric={'raw':raw,'temperature_scaled':report.metrics(scaled),'temperature':1.}
        store.write_rows(dest/'predictions.jsonl',predictions);store.atomic(dest/'metrics.json',metric)
        store.seal(dest,{'fit_seal':store.sha(fit/'DONE.json'),'selection':store.sha(rootu/'selection.json')})
    return p


def test_complete_summary_keeps_budget_seed_family_and_views(tmp_path):
    p=make_sealed_report(tmp_path);result=report.summarize(p)
    assert result['status']=='complete'
    s=store.read(Path(result['output'])/'summary.json')
    assert s['method_changed'] and len(s['rows'])==4
    assert {r['total_labels'] for r in s['rows']}=={16,32}
    assert {r['view'] for r in s['rows']}=={'raw','temperature_scaled'}
    assert all(r['support_seeds']==3 and r['method']=='conditional-gated' for r in s['rows'])
    assert len(store.read(Path(result['output'])/'fit_calibration_diagnostics.json')['rows'])==6


def test_summary_retains_vqa_score(tmp_path,monkeypatch):
    p=make_sealed_report(tmp_path,vqa=True)
    class Eval:
        def eval_pred_list(self,data):return .6
    monkeypatch.setattr(data,'evaluator',lambda *a:SimpleNamespace(TextVQAAccuracyEvaluator=Eval))
    result=report.summarize(p);rows=store.read(Path(result['output'])/'summary.json')['rows']
    assert len(rows)==2 and all(r['vqa_accuracy']['mean']==.6 for r in rows)
    assert all(r['view']=='free_generation' for r in rows)


def test_summary_rejects_corruption_and_duplicate_seed(tmp_path):
    p=make_sealed_report(tmp_path)
    p['units'].append(p['units'][0])
    with pytest.raises(ValueError):report.summarize(p)
    p['units'].pop()
    u=p['units'][0];v=u['variants'][0]
    dest=Path(u['output'])/'evaluation'/v['name']/'predictions.jsonl'
    dest.write_text('corrupt\n')
    with pytest.raises(ValueError):report.summarize(p)


def test_prepare_local_config_resolves_old_assets_without_overwriting(tmp_path):
    import subprocess
    from test_conditional_gated import make_base
    old=tmp_path/'old';(old/'configs').mkdir(parents=True)
    store.atomic(old/'configs/aperture_comprehensive_v1.json',make_base())
    new=tmp_path/'new';(new/'configs').mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]
    cfg=json.loads((source/'configs/aperture_conditional_gated_v1.json').read_text())
    store.atomic(new/'configs/aperture_conditional_gated_v1.json',cfg)
    script=source/'scripts/prepare_aperture_gated_config.py'
    result=subprocess.run([sys.executable,str(script),'--source-repo',str(old),'--new-repo',str(new),
                           '--out','configs/cg.local.json'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    created=store.read(new/'configs/cg.local.json');base=store.read(created['base_config'])
    assert base['models'][0]['path']==str(old/'models/q25')
    assert base['prepared_root']==str(old/'data/old_prepared')
    assert created['execution_batch']==1 and 'conditional_gated' in created['run_root']
    again=subprocess.run([sys.executable,str(script),'--source-repo',str(old),'--new-repo',str(new),
                           '--out','configs/cg.local.json'],capture_output=True,text=True)
    assert again.returncode!=0


def test_synthetic_fit_select_fresh_restore_evaluate_pipeline(monkeypatch,tmp_path):
    """Runs actual new engine, torch optimizer and checkpoint; native assets mocked."""
    import numpy as np
    roles={}
    for role in ('support','calibration','query'):
        roles[role]=[{**r,'sample_id':role+r['sample_id'],'group_id':role+r['group_id']} for r in toy_roles()['support']]
    query_calls=[]
    class NativeModel(ToyModel):
        def gradient_checkpointing_disable(self):pass
    class NativeBackend(ToyBackend):
        def candidate(self,row,candidate,controller=None):
            if row['sample_id'].startswith('query'):query_calls.append(row['sample_id'])
            return super().candidate(row,candidate,controller)
        def score(self,row,controller=None):
            try:
                with torch.no_grad():scores=[float(self.candidate(row,c,controller)[0].sum()) for c in row['candidate_labels']]
                z=np.array(scores);p=np.exp(z-z.max());p/=p.sum()
                return {'sample_id':row['sample_id'],'label_id':row['label_id'],'mean_log_scores':scores,
                        'probabilities':p.tolist(),'prediction':int(p.argmax()),'prompt_contract_hash':'synthetic','image_tokens':2}
            finally:
                if controller:controller.clear_mask()
        def candidate_batch(self,rs,candidates=None,controller=None):
            cs=candidates or [r['label'] for r in rs]
            lps=[self.candidate(r,c,controller)[0].sum() for r,c in zip(rs,cs)]
            return lps,None,{'rows':[{'prefix':r['sample_id'],'image_tokens':2} for r in rs]},[{'span':(4,5)} for r in rs]
    @contextmanager
    def native(job):
        m=NativeModel();yield m,NativeBackend(m),None
    fake_module(monkeypatch,'aperture_comprehensive.backend',loaded=native,decoder_sites=lambda m,*a:[(1,'K',m.proj)])
    fake_module(monkeypatch,'vigor_handoff.core',extract_basis=lambda *a:({(1,'K'):torch.eye(8)[:,:3]},{}))
    fake_module(monkeypatch,'vigor_handoff.worker',train_lora=lambda *a:pytest.fail('Wrong trainer'))
    fake_module(monkeypatch,'aperture_comprehensive.numeric',score_arrays=lambda rs,c:(np.array([r['mean_log_scores'] for r in rs]),np.array([r['label_id'] for r in rs])))
    def probs(z):
        z=np.array(z);p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
    fake_module(monkeypatch,'aperture_final.numeric',fit_temperature=lambda *a,**kw:{'temperature':1.},probabilities=probs)
    monkeypatch.setattr(engine,'setup',lambda:torch)
    monkeypatch.setattr(engine,'binding',lambda *a:{'synthetic':True,'model_assets':{}})
    monkeypatch.setattr(engine,'load_roles',lambda *a,**kw:roles)
    monkeypatch.setattr(data,'load_roles',lambda *a,**kw:roles)
    for name in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'synthetic CPU')
    monkeypatch.setenv('APERTURE_BATCH','2')
    variant=next(v for v in plan.variants('CG01','single') if v['name']=='conditional-gated')
    variant.update(rank=3,layers=[1],sites=['K'])
    unit={'family':'toy','layers':[1],'rank':3,'task_type':'classification','classes':2,'model_id':'synthetic',
          'output':str(tmp_path/'units/u'),'loss_reduction':'mean','mask':'all_visual','image_size':8,
          'method_version':plan.VERSION,'total_labels':4,'key':'u','case':'CG01','variants':[variant],
          'evaluate_policy':'selected'}
    p={'case':'CG01','settings':{},'units':[unit]}
    aud=Path(unit['output'])/'audit';store.atomic(aud/'audit.json',{'synthetic':True});store.seal(aud,{'synthetic':True,'model_assets':{}})
    engine.fit(unit,variant,p)
    assert not query_calls
    fit=Path(unit['output'])/'fits'/variant['name']
    stats=store.read(fit/'fit.json')
    assert stats['optimized_scalars']==13 and stats['optimizer_updates']==4
    assert stats['final_support_metrics'] and store.read(fit/'gate_diagnostics.json')['support']
    engine.seal_selection(unit,p)
    engine.evaluate(unit,variant,p,tmp_path)
    ev=Path(unit['output'])/'evaluation'/variant['name'];store.verify(ev)
    result=store.read(ev/'metrics.json');assert result['calibration_replay_max_delta']==0
    assert len(query_calls)==4 # 2 examples x 2 candidates
    engine.evaluate(unit,variant,p,tmp_path)
    assert len(query_calls)==4 # sealed state reused, not recomputed
