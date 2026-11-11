"""CPU contracts; these tests do not claim native VLM or GPU performance."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aperture_conditional_gated.controller import ConditionalGatedController
from aperture_conditional_gated.plan import CASES, variants, build_units, freeze


def fixture(mode='conditional', alpha=3., width=8, rank=3):
    torch.manual_seed(7)
    module=nn.Linear(width,width,bias=False)
    module.weight.requires_grad_(False)
    basis=torch.linalg.qr(torch.randn(width,rank),mode='reduced').Q
    c=ConditionalGatedController([(1,'K',module)],{(1,'K'):basis},alpha=alpha,gate_mode=mode)
    return module,basis,c


def test_identity_and_no_backbone_registration():
    m,b,c=fixture();x=torch.randn(2,4,8)
    c.set_mask(torch.tensor([[0,1,0,1],[1,0,1,0]],dtype=torch.bool))
    assert torch.equal(m(x),nn.functional.linear(x,m.weight))
    assert c.num_scalars()==9+3+1
    assert all(id(p)!=id(m.weight) for p in c.parameters())
    assert not c.bases[(1,'K')].requires_grad
    c.close();assert not m._forward_hooks


@pytest.mark.parametrize('mode', ['conditional','scalar','half','one'])
def test_mask_and_bound(mode):
    m,b,c=fixture(mode);z=torch.randn(2,5,8);mask=torch.zeros(2,5,dtype=torch.bool);mask[:,1:3]=True
    c.set_mask(mask)
    with torch.no_grad():c.raw['1:K'].normal_()
    y=c._hook(1,'K')(m,None,z)
    assert torch.equal(y[~mask],z[~mask])
    a=c.operators()['1:K']
    assert torch.linalg.vector_norm(a)<3
    assert torch.all(torch.linalg.vector_norm((y-z)[mask],dim=-1)<=3*torch.linalg.vector_norm((z@b)[mask],dim=-1)+1e-5)
    c.set_mask(torch.zeros_like(mask));assert torch.equal(c._hook(1,'K')(m,None,z),z)
    c.clear_mask();assert c._hook(1,'K')(m,None,z) is z
    c.close()


def test_one_is_exact_legacy_full_and_half_is_half():
    m,b,c=fixture('one');z=torch.randn(1,4,8);mask=torch.ones(1,4,dtype=torch.bool);c.set_mask(mask)
    with torch.no_grad():c.raw['1:K'].normal_()
    r=c.raw['1:K'];a=3*r/(1+torch.linalg.vector_norm(r));expected=z+((z@b)@a)@b.T
    assert torch.equal(c._hook(1,'K')(m,None,z),expected)
    c.close()
    m,b,c=fixture('half');c.set_mask(mask)
    with torch.no_grad():c.raw['1:K'].copy_(r)
    assert torch.equal(c._hook(1,'K')(m,None,z),z+.5*(((z@b)@a)@b.T))
    c.close()


def test_gate_is_input_dependent_and_batch_independent():
    m,b,c=fixture();z=torch.randn(2,3,8)
    with torch.no_grad():c.gate_weight['1:K'].copy_(torch.tensor([1.,-1.,.5]))
    gates=c.gates('1:K',z@b)
    assert gates.min()>0 and gates.max()<1 and gates.std()>0
    assert torch.allclose(gates[0],c.gates('1:K',(z[:1]@b))[0])
    c.close()


def test_gradients_and_first_zero_gate_step():
    m,b,c=fixture();z=torch.randn(1,4,8);c.set_mask(torch.ones(1,4,dtype=torch.bool))
    loss=c._hook(1,'K')(m,None,z).square().mean();loss.backward()
    assert c.raw['1:K'].grad.abs().sum()>0
    assert c.gate_weight['1:K'].grad.abs().sum()==0
    c.zero_grad()
    with torch.no_grad():c.raw['1:K'].normal_(0,.1)
    c._hook(1,'K')(m,None,z).square().mean().backward()
    assert c.gate_weight['1:K'].grad.abs().sum()>0
    assert c.gate_bias['1:K'].grad.abs().sum()>0
    assert m.weight.grad is None
    c.close()


def test_payload_exact_and_mismatch_rejected(tmp_path):
    m,b,c=fixture();z=torch.randn(1,4,8);mask=torch.ones(1,4,dtype=torch.bool);c.set_mask(mask)
    with torch.no_grad():
        c.raw['1:K'].normal_();c.gate_weight['1:K'].normal_();c.gate_bias['1:K'].fill_(.3)
    y=c._hook(1,'K')(m,None,z).detach();payload=c.payload();c.close()
    p=tmp_path/'controller.pt';torch.save(payload,p);payload=torch.load(p,weights_only=True)
    m,b,d=fixture();d.restore(payload);d.set_mask(mask)
    assert torch.equal(y,d._hook(1,'K')(m,None,z))
    for key,value in [('alpha',1.),('gate_mode','scalar'),('format','legacy')]:
        bad=deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError):d.restore(bad)
    bad=deepcopy(payload);bad['raw']['1:K'].fill_(float('nan'))
    with pytest.raises(ValueError):d.restore(bad)
    bad=deepcopy(payload);bad['gate_weight']={}
    with pytest.raises(ValueError):d.restore(bad)
    d.close()


def test_backward_reaches_upstream_and_no_answer_context_argument():
    m,b,c=fixture();z=torch.randn(1,4,8,requires_grad=True);c.set_mask(torch.ones(1,4,dtype=torch.bool))
    with torch.no_grad():c.raw['1:K'].normal_(0,.1);c.gate_weight['1:K'].normal_()
    c._hook(1,'K')(m,None,z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert 'label' not in __import__('inspect').signature(c.gates).parameters
    c.close()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_dtype_and_finite_zero_features(dtype):
    m,b,c=fixture();z=torch.zeros(1,2,8,dtype=dtype);c.set_mask(torch.ones(1,2,dtype=torch.bool))
    with torch.no_grad():c.raw['1:K'].normal_()
    y=c._hook(1,'K')(m,None,z)
    assert y.dtype==dtype and torch.isfinite(y).all()
    c.close()


@pytest.mark.parametrize('bad', [-1.,float('nan'),float('inf')])
def test_invalid_alpha(bad):
    with pytest.raises(ValueError):fixture(alpha=bad)


def test_invalid_mask_and_mode():
    m,b,c=fixture()
    with pytest.raises(ValueError):c.set_mask(torch.ones(1,2))
    c.set_mask(torch.ones(1,2,dtype=torch.bool))
    with pytest.raises(ValueError):c._hook(1,'K')(m,None,torch.zeros(1,3,8))
    c.close()
    with pytest.raises(ValueError):fixture('unknown')


def test_plan_registered_and_gate_is_primary():
    assert set(CASES)=={f'CG{i:02}' for i in range(10)}
    vs=variants('CG01','single')
    assert {'conditional-gated','static-full','lora1','lora4','frozen'}<= {v['name'] for v in vs}
    gate=next(v for v in vs if v['name']=='conditional-gated')
    assert gate['engine']=='conditional_gated' and gate['objective']=='token_mean'
    assert gate['rank']==16 and gate['gate_mode']=='conditional'


def test_tuning_includes_alpha_two_and_equal_search_count():
    from collections import Counter
    vs=variants('CG02','single')
    counts=Counter(v['family'] for v in vs)
    assert all(counts[f]==6 for f in ('conditional-gated','static-full','lora-r1','lora-r4','lora-r16','dora-r4'))
    assert {v['alpha'] for v in vs if v['engine']=='conditional_gated'}=={1.,2.,3.}
    assert not any(v['engine']=='loreft' for v in vs) # unvalidated no-op baseline is not silently promoted


def test_ablation_isolates_gate_and_basis():
    vs={v['name']:v for v in variants('CG04','single')}
    assert {'fixed-half','learned-scalar','random-gated','pca-gated','static-full','conditional-gated'}<=vs.keys()
    assert vs['random-gated']['gate_mode']=='conditional' and vs['random-gated']['basis_mode']=='random'
    assert vs['fixed-half']['gate_mode']=='half'


def make_base():
    from aperture_rebuttal_cases.plan import MODEL_INFO,PANEL
    return {'models':[{'key':k,'path':f'models/{k}'} for k in MODEL_INFO],
            'datasets':[{'key':d,'kind':'paired' if d.startswith('bright') else 'single','classes':3 if d.startswith('bright') else 2,'query_count':300 if d.startswith('bright') else 500} for d in PANEL],
            'prepared_root':'data/old_prepared','run_root':'runs/old'}


def test_budget_and_paths_separate(tmp_path):
    units=build_units('CG03',make_base(),tmp_path/'runs/cg',tmp_path,tmp_path/'data/prepared')
    assert len(units)==36
    assert {u['total_labels'] for u in units}=={16,32,64}
    assert len({u['key'] for u in units})==36
    assert all('/CG03/units/' in u['output'] and u['case']=='CG03' for u in units)
    assert all(u['batch_size']==1 for u in units)


def test_budget_diagnostic_not_silent_main_change():
    vs=variants('CG08','single')
    for family in ('conditional-gated','static-full'):
        assert {v['steps'] for v in vs if v['family']==family}=={1,2,4,8,16}
    assert {v['lora_passes'] for v in vs if v['family']=='lora-r16'}=={1,2,4,8,16}
