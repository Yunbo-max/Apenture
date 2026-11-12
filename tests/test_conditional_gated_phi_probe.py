"""Same-shape causality probes: synthetic numerical noise is NOT native evidence."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aperture_conditional_gated import audit, engine
from aperture_conditional_gated.controller import ConditionalGatedController
from aperture_rebuttal_cases import store
from test_conditional_gated_integration import fake_module, ToyModel, toy_roles


class Tokenizer:
    all_special_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    def get_vocab(self):return {f't{i}': i for i in range(64)}


class ShapeBackend:
    """Causal prefix with a deliberately length-dependent rounding surrogate.

    Real Phi uses a negative image placeholder. A label suffix of one vs two
    tokens makes the old audit compare distinct shapes, even with no leakage.
    """
    def __init__(self, model, *, leak=False, shape_noise=True, mutate=False):
        self.model = model
        self.base = model
        self.config = SimpleNamespace(vocab_size=64, image_token_id=63)
        self.processor = SimpleNamespace(tokenizer=Tokenizer())
        self.leak, self.shape_noise, self.mutate = leak, shape_noise, mutate
        self.prepared_calls = 0
        self.forward_inputs = []
        self.prepared_inputs = []
        self.job = {'family': 'phi3_v'}

    def batch(self, row, candidate):
        self.prepared_calls += 1
        answer = [11] if candidate == 'a0' else [12, 13]
        ids = torch.tensor([[9, -1, -2, 10, *answer]])
        mask = torch.tensor([[False, True, True, False]+[False]*len(answer)])
        batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                 'position_ids': torch.arange(ids.shape[1]).reshape(1,-1),
                 'pixel_values': torch.arange(12.).reshape(1,3,2,2),
                 'image_sizes': torch.tensor([[2,2]]),
                 'token_type_ids': mask.long()}
        self.prepared_inputs.append({k:v.clone() for k,v in batch.items()})
        return batch, (4, ids.shape[1]), mask, {'image_tokens':2}

    def forward_batch(self, batch):
        self.forward_inputs.append({k:v.clone() for k,v in batch.items()})
        ids = batch['input_ids']
        length = ids.shape[1]
        x = torch.arange(length*8,dtype=torch.float32).reshape(1,length,8)/20-.7
        direction = torch.tensor([1.,-.5,.3,-.8,.2,1.,-.4,.5])
        if self.shape_noise:x = x + (length % 2)*2*direction
        if self.leak:x = x + (ids[0,4].float().remainder(2)*2-1)*4*direction
        if self.mutate:
            # Native models may normalize a temporary input tensor in place.
            ids.clamp_min_(0)
        z = self.model.proj(x)
        value = (z[:,1:3]*torch.arange(1.,9.)).mean()
        logits = value*torch.linspace(-1.,1.,64).reshape(1,1,64)
        return SimpleNamespace(logits=logits.expand(1,length,64))

    def candidate(self,row,candidate,controller=None):
        batch,span,mask,contract=self.batch(row,candidate)
        if controller is not None:controller.set_mask(mask)
        out=self.forward_batch(batch)
        start,end=span
        lp=out.logits[0,start-1:end-1].log_softmax(-1)
        values=lp.gather(-1,batch['input_ids'][0,start:end,None]).flatten()
        return values,mask,contract


def make_controller(mode='conditional'):
    model=ToyModel().eval()
    c=ConditionalGatedController([(1,'K',model.proj)],{(1,'K'):torch.eye(8)[:,:3]},gate_mode=mode)
    with torch.no_grad():
        c.raw['1:K'].copy_(torch.eye(3)*.03)
        if mode=='conditional':c.gate_weight['1:K'].copy_(torch.tensor([-.2,0.,.2]))
    return model,c


def wire_audit(monkeypatch,tmp_path,backend):
    @contextmanager
    def native(job):yield backend.model,backend,None
    fake_module(monkeypatch,'aperture_comprehensive.backend',loaded=native,
                decoder_sites=lambda *a:[(1,'K',backend.model.proj)])
    monkeypatch.setattr(engine,'setup',lambda:torch)
    monkeypatch.setattr(engine,'binding',lambda *a:{'synthetic':'fixed-shape'})
    monkeypatch.setattr(audit,'load_roles',lambda *a,**kw:toy_roles())
    return {'output':str(tmp_path),'variants':[], 'layers':[1], 'rank':3, 'alpha':3.,
            'task_type':'classification','classes':2}


def test_phi_length_noise_is_not_called_answer_leakage(monkeypatch,tmp_path):
    """Fails on old audit's variable-length string substitution."""
    backend=ShapeBackend(ToyModel().eval())
    unit=wire_audit(monkeypatch,tmp_path,backend)
    audit.audit(unit,{})
    record=store.read(tmp_path/'audit/audit.json')
    assert record['passed'] and record['query_forwards']==0
    assert all(r['gate_candidate_delta']<=.005 for r in record['records'])
    assert all(r['causality_probe']['protocol']=='same-shape-answer-token-v1' for r in record['records'])
    assert not backend.model.proj._forward_hooks


def test_real_content_leakage_still_blocks_native_audit(monkeypatch,tmp_path):
    backend=ShapeBackend(ToyModel().eval(),leak=True)
    unit=wire_audit(monkeypatch,tmp_path,backend)
    with pytest.raises(RuntimeError,match='content|candidate'):
        audit.audit(unit,{})
    assert not (tmp_path/'audit/DONE.json').exists()
    assert not backend.model.proj._forward_hooks


@pytest.mark.parametrize('label',['a0','a1'])
@pytest.mark.parametrize('mode',['conditional','scalar','half','one'])
def test_probe_changes_only_answer_ids_same_shape(label,mode):
    from aperture_conditional_gated.causal_probe import same_shape_gate_probe
    model,c=make_controller(mode)
    backend=ShapeBackend(model)
    row={**toy_roles()['support'][0],'label':label}
    try:
        result=same_shape_gate_probe(backend,row,c)
        assert result['protocol']=='same-shape-answer-token-v1'
        assert result['threshold']==.005 and result['probe_count']==2
        assert result['max_gate_delta']==0.
        assert result['reference_replay_delta']==0.
        assert backend.prepared_calls==1  # no alternate string re-tokenization
        assert len(backend.forward_inputs)==4  # reference twice, two probes
        ref=backend.forward_inputs[0]
        for trial in backend.forward_inputs[1:]:
            for key in ref:
                assert trial[key].shape==ref[key].shape
                assert trial[key].dtype==ref[key].dtype
                if key!='input_ids':assert torch.equal(trial[key],ref[key])
            assert torch.equal(trial['input_ids'][:,:4],ref['input_ids'][:,:4])
        for probe in backend.forward_inputs[2:]:
            assert torch.all(probe['input_ids'][:,4:]!=ref['input_ids'][:,4:])
            assert torch.all(probe['input_ids'][:,4:]>=11)
            assert not torch.any(probe['input_ids'][:,4:]==63)  # visual special
        assert not torch.equal(backend.forward_inputs[2]['input_ids'],backend.forward_inputs[3]['input_ids'])
        assert c._mask is None and not c.trace_enabled and not c.trace
    finally:c.close()


def test_probe_copies_prepared_inputs_before_native_inplace_normalization():
    from aperture_conditional_gated.causal_probe import same_shape_gate_probe
    model,c=make_controller();backend=ShapeBackend(model,mutate=True)
    try:
        result=same_shape_gate_probe(backend,toy_roles()['support'][0],c)
        assert result['max_gate_delta']==0.
        assert all(b['input_ids'][0,1:3].tolist()==[-1,-2] for b in backend.forward_inputs)
        assert backend.prepared_inputs[0]['input_ids'][0,1:3].tolist()==[-1,-2]
    finally:c.close()


def test_probe_keeps_nonanswer_suffix_and_positions_unchanged():
    from aperture_conditional_gated.causal_probe import same_shape_gate_probe
    model,c=make_controller();backend=ShapeBackend(model)
    batch_method=backend.batch
    def batch(row,label):
        b,span,m,contract=batch_method(row,label)
        for k in ('input_ids','attention_mask','token_type_ids','position_ids'):
            b[k]=torch.cat([b[k],torch.full((1,2),7 if k=='input_ids' else 0,dtype=b[k].dtype)],dim=1)
        m=torch.cat([m,torch.zeros(1,2,dtype=torch.bool)],dim=1)
        return b,span,m,contract
    backend.batch=batch
    try:
        same_shape_gate_probe(backend,toy_roles()['support'][0],c)
        assert all(torch.equal(b['input_ids'][:,-2:],torch.tensor([[7,7]])) for b in backend.forward_inputs)
    finally:c.close()


@pytest.mark.parametrize('problem',['no_visual','answer_overlaps_visual','zero_length','past_end','padding_answer','no_safe_tokens','not_integer'])
def test_malformed_probe_inputs_fail_closed(problem):
    from aperture_conditional_gated.causal_probe import same_shape_gate_probe
    model,c=make_controller();backend=ShapeBackend(model)
    original=backend.batch
    def batch(row,candidate):
        b,span,m,contract=original(row,candidate)
        if problem=='no_visual':m.zero_()
        elif problem=='answer_overlaps_visual':m[0,span[0]]=True
        elif problem=='zero_length':span=(4,4)
        elif problem=='past_end':span=(4,99)
        elif problem=='padding_answer':b['attention_mask'][0,4]=0
        elif problem=='not_integer':b['input_ids']=b['input_ids'].float()
        return b,span,m,contract
    backend.batch=batch
    if problem=='no_safe_tokens':backend.processor.tokenizer.all_special_ids=list(range(64))
    try:
        with pytest.raises((ValueError,RuntimeError)):same_shape_gate_probe(backend,toy_roles()['support'][0],c)
        assert not backend.forward_inputs
        assert c._mask is None and not c.trace_enabled
    finally:c.close()


@pytest.mark.parametrize('problem',['missing_trace','nan_trace','inf_trace','constant_wrong_size','native_exception','reference_instability'])
def test_probe_rejects_invalid_traces_and_cleans_up(problem):
    from aperture_conditional_gated.causal_probe import same_shape_gate_probe
    model,c=make_controller();backend=ShapeBackend(model)
    original=backend.forward_batch
    def forward(batch):
        out=original(batch)
        if problem=='missing_trace':c.trace={}
        elif problem=='nan_trace':c.trace['1:K'].fill_(float('nan'))
        elif problem=='inf_trace':c.trace['1:K'].fill_(float('inf'))
        elif problem=='constant_wrong_size':c.trace['1:K']=torch.zeros(1)
        elif problem=='native_exception':raise RuntimeError('native failure')
        elif problem=='reference_instability':c.trace['1:K']+=.02*len(backend.forward_inputs)
        return out
    backend.forward_batch=forward
    try:
        with pytest.raises((RuntimeError,ValueError)):same_shape_gate_probe(backend,toy_roles()['support'][0],c)
        assert c._mask is None and not c.trace_enabled and not c.trace
    finally:c.close()


@pytest.mark.parametrize('family',['phi3_v','qwen2_5_vl','qwen3_vl','gemma3','internvl_chat'])
def test_probe_uses_actual_native_forward_dispatch_without_retokenizing(family):
    """Executes the unchanged repository forward_batch function; model is tiny."""
    import ast
    from types import MethodType
    from aperture_conditional_gated.causal_probe import same_shape_gate_probe
    source=Path(__file__).resolve().parents[1]/'src/aperture_comprehensive/backend.py'
    cls=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='Backend')
    definition=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward_batch')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),namespace)
    model,c=make_controller();backend=ShapeBackend(model);backend.job['family']=family
    synthetic_forward=backend.forward_batch
    forwarded=[]
    if family=='internvl_chat':
        original_batch=backend.batch
        def batch(row,label):
            b,s,m,ctx=original_batch(row,label)
            b['input_ids'][0,1:3]=63
            return b,s,m,ctx
        backend.batch=batch;backend.config.img_context_token_id=63
        embedding=nn.Embedding(64,8)
        embedding.weight.requires_grad_(False)
        class Language(nn.Module):
            def get_input_embeddings(self):return embedding
            def forward(self,inputs_embeds,attention_mask,use_cache,return_dict):
                forwarded.append((use_cache,return_dict,list(inputs_embeds.shape)))
                # Each visual token's prefix calculation is causal in this fixture.
                z=model.proj(inputs_embeds)
                return SimpleNamespace(logits=z)
        language=Language().eval()
        backend.base=SimpleNamespace(language_model=language,
            extract_feature=lambda pixels:torch.arange(16.).reshape(1,2,8))
    else:
        def forward(self,use_cache,return_dict,**batch):
            forwarded.append((use_cache,return_dict,list(batch['input_ids'].shape)))
            return synthetic_forward(batch)
        model.forward=MethodType(forward,model)
    backend.forward_batch=MethodType(namespace['forward_batch'],backend)
    try:
        state={k:p.detach().clone() for k,p in c.named_parameters()}
        result=same_shape_gate_probe(backend,toy_roles()['support'][0],c)
        assert result['max_gate_delta']==0.
        assert len(forwarded)==4 and all(not cache and returns for cache,returns,_ in forwarded)
        assert len({tuple(shape) for _,_,shape in forwarded})==1
        assert all(torch.equal(v,dict(c.named_parameters())[k]) for k,v in state.items())
        assert all(p.grad is None for p in c.parameters())
    finally:c.close()


def test_probe_rejects_training_mode():
    from aperture_conditional_gated.causal_probe import same_shape_gate_probe
    model,c=make_controller();model.train();backend=ShapeBackend(model)
    try:
        with pytest.raises(ValueError,match='eval'):same_shape_gate_probe(backend,toy_roles()['support'][0],c)
        assert not backend.forward_inputs
    finally:c.close()


@pytest.mark.parametrize('field',['pixel_values','image_sizes','attention_mask','position_ids','token_type_ids','input_ids'])
def test_unchanged_native_fields_are_actually_validated(field):
    from aperture_conditional_gated.causal_probe import _check_only_answer_changed
    backend=ShapeBackend(ToyModel().eval())
    b,span,_,_=backend.batch(toy_roles()['support'][0],'a0')
    changed={k:v.clone() for k,v in b.items()}
    changed[field].view(-1)[0]+=1
    with pytest.raises(ValueError):_check_only_answer_changed(b,changed,*span)
