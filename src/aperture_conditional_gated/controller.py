"""Input-conditioned, bounded visual-token residuals. No query-label inputs."""
from __future__ import annotations
import math
from typing import Mapping
import torch
from torch import nn

FORMAT = 'aperture-conditional-gated-v1'


class ConditionalGatedController(nn.Module):
    """A scalar gate per visual token and K/V site, conditioned on z @ U.

    The backbone is deliberately held in an ordinary list, not registered as a
    child module. U is frozen. R, w and b are optimized. The gate starts at .5;
    R starts at zero, giving an exactly zero residual. A zero first-step gate
    data gradient is expected, not evidence of an inactive gate.

    'half' and 'scalar' isolate attenuation and input-independent gating;
    'one' is the exact static Full control. They are NOT conditional methods.
    """
    def __init__(self, modules, bases, alpha=3., gate_mode='conditional', eps=1e-6):
        super().__init__()
        if gate_mode not in ('conditional','scalar','half','one'):
            raise ValueError('unknown gate_mode')
        if not math.isfinite(alpha) or alpha<0 or not math.isfinite(eps) or eps<=0:
            raise ValueError('invalid alpha or normalization epsilon')
        targets=list(modules)
        keys=[(l,k) for l,k,_ in targets]
        if not keys or len(set(keys))!=len(keys) or set(keys)!=set(bases):
            raise ValueError('sites and bases must be nonempty, unique and exactly matched')
        checked={}
        for l,k,m in targets:
            b=bases[(l,k)].detach().float().clone()
            if b.ndim!=2 or b.shape[0]!=m.out_features or not 0<b.shape[1]<=b.shape[0]:
                raise ValueError('basis/site dimension mismatch')
            if not torch.isfinite(b).all() or not torch.allclose(b.T@b,torch.eye(b.shape[1],device=b.device),atol=2e-4):
                raise ValueError('basis must be finite and orthonormal')
            checked[(l,k)]=b
        self.targets=targets;self.bases=checked
        self.alpha=float(alpha);self.gate_mode=gate_mode;self.eps=float(eps)
        self.raw=nn.ParameterDict();self.gate_weight=nn.ParameterDict();self.gate_bias=nn.ParameterDict()
        self._mask=None;self._handles=[];self.trace_enabled=False;self.trace={}
        for l,k,m in targets:
            key=f'{l}:{k}';b=self.bases[(l,k)];r=b.shape[1]
            device=next(m.parameters(),b).device
            self.raw[key]=nn.Parameter(torch.zeros(r,r,device=device,dtype=torch.float32))
            if gate_mode=='conditional':
                self.gate_weight[key]=nn.Parameter(torch.zeros(r,device=device,dtype=torch.float32))
            if gate_mode in ('conditional','scalar'):
                self.gate_bias[key]=nn.Parameter(torch.zeros((),device=device,dtype=torch.float32))
        try:
            for l,k,m in targets:self._handles.append(m.register_forward_hook(self._hook(l,k)))
        except Exception:
            self.close();raise

    def operators(self):
        return {k:self.alpha*r/(1+torch.linalg.vector_norm(r)) for k,r in self.raw.items()}

    def gates(self,key,low):
        if self.gate_mode in ('half','one'):
            return torch.full((*low.shape[:-1],1),.5 if self.gate_mode=='half' else 1.,device=low.device,dtype=torch.float32)
        logits=self.gate_bias[key]
        if self.gate_mode=='conditional':
            h=low.float()
            h=h/torch.sqrt(h.square().mean(dim=-1,keepdim=True)+self.eps)
            logits=(h*self.gate_weight[key]).sum(dim=-1,keepdim=True)+logits
        else:logits=logits.expand(*low.shape[:-1],1)
        return torch.sigmoid(logits)

    def _hook(self,layer,kind):
        key=f'{layer}:{kind}'
        def apply(module,inputs,z):
            mask=self._mask
            if mask is None:return z
            if not torch.is_tensor(z) or z.ndim!=3 or mask.shape!=z.shape[:2] or mask.device!=z.device:
                raise ValueError('projection/mask shape or device mismatch; use no-cache generation')
            b=self.bases[(layer,kind)].to(z);low=z@b
            a=self.operators()[key].to(z)
            gate=self.gates(key,low)
            delta=((low@a)@b.T)*gate.to(z)
            if self.trace_enabled:
                self.trace[key]=gate.detach()[mask].float().cpu().clone()
            return torch.where(mask.unsqueeze(-1),z+delta,z)
        return apply

    def set_mask(self,mask):
        if not torch.is_tensor(mask) or mask.ndim!=2 or mask.dtype!=torch.bool:
            raise ValueError('visual mask must be a 2-D bool tensor')
        self._mask=mask

    def clear_mask(self):self._mask=None

    def close(self):
        self.clear_mask()
        for handle in self._handles:handle.remove()
        self._handles=[];self.trace={}

    def num_scalars(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def diagnostics(self):
        return {key:{'raw_frobenius':float(torch.linalg.vector_norm(self.raw[key].detach())),
                     'operator_frobenius':float(torch.linalg.vector_norm(a.detach())),
                     'gate_weight_norm':float(torch.linalg.vector_norm(self.gate_weight[key].detach())) if key in self.gate_weight else 0.,
                     'gate_bias':float(self.gate_bias[key].detach()) if key in self.gate_bias else None,
                     'gate_mode':self.gate_mode} for key,a in self.operators().items()}

    def payload(self):
        cpu=lambda ps:{k:v.detach().cpu().clone() for k,v in ps.items()}
        return {'format':FORMAT,'gate_mode':self.gate_mode,'alpha':self.alpha,'eps':self.eps,
                'conditioning':'per-token RMS-normalized visual z@U; no labels or batch pooling',
                'bases':{f'{l}:{k}':b.detach().cpu().clone() for (l,k),b in self.bases.items()},
                'raw':cpu(self.raw),'gate_weight':cpu(self.gate_weight),'gate_bias':cpu(self.gate_bias)}

    def restore(self,payload:Mapping):
        for k,v in [('format',FORMAT),('gate_mode',self.gate_mode),('alpha',self.alpha),('eps',self.eps)]:
            if payload.get(k)!=v:raise ValueError(f'Incompatible gated payload: {k}')
        expected={f'{l}:{k}':b for (l,k),b in self.bases.items()}
        groups=[('bases',expected),('raw',self.raw),('gate_weight',self.gate_weight),('gate_bias',self.gate_bias)]
        # Validate everything BEFORE mutating any parameter.
        for name,params in groups:
            saved=payload.get(name,{})
            if set(saved)!=set(params):raise ValueError(f'Payload keys mismatch: {name}')
            for k,v in params.items():
                t=saved[k]
                if not torch.is_tensor(t) or t.shape!=v.shape or not torch.isfinite(t).all():
                    raise ValueError(f'Payload shape/nonfinite tensor: {name}/{k}')
                if name=='bases' and not torch.equal(t.detach().cpu().float(),v.detach().cpu().float()):
                    raise ValueError('Payload basis differs from constructed controller')
        with torch.no_grad():
            for name,params in groups[1:]:
                for k,v in params.items():v.copy_(payload[name][k].to(v))
