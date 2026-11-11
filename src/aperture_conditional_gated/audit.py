"""Fail-closed native support audit: identity, gradients, masks, gate causality."""
from pathlib import Path
import torch
from aperture_rebuttal_cases import store
from aperture_rebuttal_cases.data import load_roles
from aperture_rebuttal_cases.generation import training_row
from .controller import ConditionalGatedController


def audit(unit,plan):
    from .engine import setup,binding
    from aperture_comprehensive.backend import loaded as native_loaded,decoder_sites
    setup();folder=Path(unit['output'])/'audit';identity=binding(unit,plan)
    if (folder/'DONE.json').exists():store.verify(folder,identity);return
    roles=load_roles(unit,images=True);records=[]
    configs={(v.get('gate_mode','conditional'),tuple(v['sites']),v['rank'],v['alpha'])
             for v in unit['variants'] if v['engine']=='conditional_gated'}
    if not configs:configs={('conditional',('K','V'),unit['rank'],unit['alpha'])}
    with native_loaded({**unit,'arm':'frozen'}) as (model,backend,_):
        for mode,kind,rank,alpha in sorted(configs):
            sites=[s for s in decoder_sites(model,unit['layers']) if s[1] in kind]
            bases={}
            for l,k,m in sites:
                gen=torch.Generator().manual_seed(l+(100 if k=='V' else 0))
                bases[(l,k)]=torch.linalg.qr(torch.randn(m.out_features,rank,generator=gen),mode='reduced').Q
            c=ConditionalGatedController(sites,bases,alpha,mode)
            try:
                examples=roles['support'][:1] if unit['task_type']=='vqa' else [next(r for r in roles['support'] if r['label_id']==i) for i in range(unit['classes'])]
                for original in examples:
                    row=training_row(original) if unit['task_type']=='vqa' else original
                    with torch.no_grad():plain,_,_=backend.candidate(row,row['label'])
                    with torch.no_grad():
                        for p in c.parameters():p.zero_()
                    adapted,mask,contract=backend.candidate(row,row['label'],c)
                    zero=float((plain-adapted).detach().abs().max())
                    if not torch.isfinite(adapted).all() or zero>.005:raise RuntimeError('Native zero-residual identity failed')
                    (-adapted.mean()).backward()
                    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in c.raw.values()):
                        raise RuntimeError('No native residual gradient')
                    c.zero_grad(set_to_none=True);c.clear_mask()
                    with torch.no_grad():
                        for p in c.raw.values():p.copy_(torch.eye(rank,device=p.device)*.03)
                        for p in c.gate_weight.values():p.copy_(torch.linspace(-.2,.2,rank,device=p.device))
                    c.trace_enabled=True;changed,_,_=backend.candidate(row,row['label'],c)
                    (-changed.mean()).backward()
                    grad={k:float(p.grad.detach().norm()) if p.grad is not None else None for k,p in c.named_parameters()}
                    if any(v is None or not __import__('math').isfinite(v) for v in grad.values()):
                        raise RuntimeError('Missing/nonfinite native controller gradient')
                    if mode=='conditional' and not any(v>0 for k,v in grad.items() if k.startswith('gate_weight')):
                        raise RuntimeError('Gate has no native data gradient after residual activation')
                    first={k:v.clone() for k,v in c.trace.items()};c.zero_grad(set_to_none=True);c.clear_mask()
                    alternative=next((x for x in row.get('candidate_labels',[]) if x!=row['label']),'not the answer')
                    probe=dict(row)
                    if alternative not in row.get('candidate_labels',[]):
                        probe['candidate_labels']=[*row.get('candidate_labels',[]),alternative]
                    with torch.no_grad():backend.candidate(probe,alternative,c)
                    if set(first)!=set(c.trace):raise RuntimeError('Gate trace site missing')
                    gate_delta=max(float((first[k]-c.trace[k]).abs().max()) for k in first)
                    if gate_delta>.005:raise RuntimeError('Visual gate depends on future candidate; causal/prompt audit failed')
                    c.trace_enabled=False;c.clear_mask()
                    saved=c.payload()
                    with torch.no_grad():
                        for p in c.parameters():p.zero_()
                    c.restore(saved)
                    with torch.no_grad():restored,_,_=backend.candidate(row,row['label'],c)
                    replay=float((changed.detach()-restored).abs().max())
                    if replay>.005:raise RuntimeError('Native gated payload replay failed')
                    c.clear_mask()
                    records.append({'sample_id':row['sample_id'],'gate_mode':mode,'alpha':alpha,'zero_max_delta':zero,
                                    'gate_candidate_delta':gate_delta,'payload_replay_delta':replay,'grad_norms':grad,
                                    'image_tokens':int(mask.sum()),'prompt_contract':contract})
                # Explicitly check row isolation AND zero-visual identity at every site.
                for l,k,m in sites:
                    z=torch.randn(1,4,m.out_features,device=next(model.parameters()).device,dtype=torch.bfloat16)
                    mask=torch.tensor([[False,True,False,True]],device=z.device);c.set_mask(mask)
                    y=c._hook(l,k)(m,None,z)
                    if not torch.equal(y[~mask],z[~mask]):raise RuntimeError('Nonvisual row changed')
                    c.set_mask(torch.zeros_like(mask))
                    if not torch.equal(c._hook(l,k)(m,None,z),z):raise RuntimeError('Text-only identity failed')
            finally:c.close()
    store.atomic(folder/'audit.json',{'passed':True,'query_forwards':0,'records':records,
                  'scope':'native support: identity, nonzero activated-gate gradients, answer independence, payload, row isolation',
                  'performance_claim':False})
    store.seal(folder,identity)
