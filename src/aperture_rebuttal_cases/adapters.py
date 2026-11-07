"""Original APERTURE plus explicitly named baseline/ablation adapters."""
from contextlib import contextmanager
import torch
from torch import nn


def ablation_operator(raw, alpha, mode):
    matrix=torch.diag(raw) if mode=='diagonal_matched' else raw
    if mode=='unbounded':return alpha*matrix
    if mode in ('full','diagonal_matched'):return alpha*matrix/(1+torch.linalg.vector_norm(matrix))
    raise ValueError('Unsupported operator')


class MaskedOperator(nn.Module):
    """Ablations only. Main Full uses the original implementation unchanged."""
    def __init__(self, sites, bases, alpha, mode):
        super().__init__();self.targets=list(sites);self.bases=bases;self.alpha=alpha;self.mode=mode
        self.raw=nn.ParameterDict();self.mask=None;self.handles=[]
        for layer,kind,module in sites:
            key=f'{layer}:{kind}';r=bases[(layer,kind)].shape[1];device=next(module.parameters()).device
            self.raw[key]=nn.Parameter(torch.zeros((r,) if mode=='diagonal_matched' else (r,r),device=device))
            self.handles.append(module.register_forward_hook(self._hook(layer,kind)))
    def _hook(self,l,k):
        def call(module,inputs,z):
            if self.mask is None:return z
            if z.shape[:2]!=self.mask.shape:raise ValueError('Mask/activation mismatch')
            b=self.bases[(l,k)].to(z);a=ablation_operator(self.raw[f'{l}:{k}'],self.alpha,self.mode).to(z)
            return z+self.mask.to(z).unsqueeze(-1)*((z@b)@a@b.T)
        return call
    def set_mask(self,m):self.mask=m
    def clear_mask(self):self.mask=None
    def close(self):
        self.clear_mask()
        for h in self.handles:h.remove()
        self.handles=[]
    def payload(self):
        return {'bases':{f'{l}:{k}':b.detach().cpu().clone() for (l,k),b in self.bases.items()},
                'raw':{k:v.detach().cpu().clone() for k,v in self.raw.items()},'alpha':self.alpha,'mode':self.mode}
    def restore(self,payload):
        with torch.no_grad():
            for k,v in payload['raw'].items():self.raw[k].copy_(v)
    def num_scalars(self):return sum(p.numel() for p in self.parameters())


class LoReFTVisual(nn.Module):
    """Official pyreft operator, applied at declared decoder block INPUT visual rows.

    This VLM placement is our baseline adaptation, not an official VLM benchmark
    reproduction. Saving each official intervention separately respects its custom
    state_dict implementation; do not silently substitute a hidden-state lens.
    """
    def __init__(self, model, layers, rank, payload=None):
        super().__init__()
        from pyreft import LoreftIntervention
        from vigor_handoff.worker import discover_sites
        self.targets=discover_sites(model,layers,['H']);self.modules_=nn.ModuleDict();self.mask=None;self.handles=[]
        for l,k,site in self.targets:
            device=next(site.parameters()).device
            module=LoreftIntervention(embed_dim=site.out_features,low_rank_dimension=rank,dtype=torch.float32,dropout=0.)
            module=module.to(device)
            if payload is not None:module.load_state_dict(payload['interventions'][str(l)])
            self.modules_[str(l)]=module
            self.handles.append(site.register_forward_hook(self._hook(str(l))))
        self.rank=rank
    def _hook(self,key):
        def call(module,inputs,z):
            if self.mask is None:return z
            if z.shape[:2]!=self.mask.shape:raise ValueError('LoReFT mask mismatch')
            changed=self.modules_[key](z.float()).to(z)
            return torch.where(self.mask.unsqueeze(-1),changed,z)
        return call
    def set_mask(self,m):self.mask=m
    def clear_mask(self):self.mask=None
    def close(self):
        self.clear_mask()
        for h in self.handles:h.remove()
        self.handles=[]
    def payload(self):
        return {'interventions':{k:{n:t.detach().cpu().clone() for n,t in v.state_dict().items()} for k,v in self.modules_.items()},
                'rank':self.rank,'placement':'decoder_block_input_visual_rows'}
    def num_scalars(self):return sum(p.numel() for p in self.parameters())


def lora_options(variant, targets):
    return dict(r=variant['lora_rank'], lora_alpha=variant['lora_alpha'],
                lora_dropout=variant['lora_dropout'], target_modules=targets,
                use_dora=variant['engine']=='dora', bias='none', task_type='CAUSAL_LM')


def native_job(unit, variant):
    # The variant's family groups baselines; it is NOT the VLM model_type.
    return {**unit, **variant, 'family': unit['family'], 'arm': 'frozen'}


@contextmanager
def loaded(unit,variant,restore=None):
    from aperture_comprehensive.backend import loaded as native_loaded, Backend, decoder_sites, lora_targets
    from vigor_handoff.core import ResidualController, extract_basis
    from .data import load_roles
    from .generation import training_row
    job=native_job(unit,variant)
    controller=None
    with native_loaded(job) as (model,backend,_native_controller):
        # Native loaded() returns (model, backend, controller). The processor
        # belongs to backend; the third item is None for the frozen arm.
        processor=backend.processor
        try:
            engine=variant['engine']
            if engine in ('lora','dora'):
                from peft import LoraConfig,PeftModel,get_peft_model
                if restore:model=PeftModel.from_pretrained(model,str(restore/'adapter'),is_trainable=False,local_files_only=True)
                else:
                    config=LoraConfig(**lora_options(variant,lora_targets(model)))
                    model=get_peft_model(model,config)
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                    model.enable_input_require_grads()
                if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):raise ValueError('Trainable base weights')
                backend=Backend(job,model,processor)
            elif engine=='loreft':
                payload=torch.load(restore/'controller.pt',weights_only=True,map_location='cpu') if restore else None
                controller=LoReFTVisual(model,job['layers'],variant['lora_rank'],payload)
            elif engine=='aperture':
                sites=[s for s in decoder_sites(model,job['layers']) if s[1] in job['sites']]
                if restore:
                    payload=torch.load(restore/'controller.pt',weights_only=True,map_location='cpu')
                    bases={(int(k.split(':')[0]),k.split(':')[1]):b for k,b in payload['bases'].items()};spectra={};seconds=0.
                else:
                    import time
                    support=load_roles(unit)['support']
                    if unit['task_type']=='vqa':support=[training_row(r) for r in support]
                    torch.cuda.synchronize();start=time.perf_counter()
                    bases,spectra=extract_basis(model,sites,support,backend.loss_and_mask,job['rank'],job['basis_mode'],job['basis_seed'])
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                klass=ResidualController if job['mode']=='full' else MaskedOperator
                controller=klass(sites,bases,job['alpha'],job['mode'])
                if restore:controller.restore(payload)
                controller.extraction_seconds=seconds;controller.basis_spectra=spectra
            model.eval()
            yield model,backend,controller
        finally:
            if controller is not None:controller.close()


def train_control(model,backend,controller,support,job):
    from vigor_handoff.core import fit_coefficients
    if job['engine']=='aperture':
        return fit_coefficients(model,controller,support,lambda r:backend.loss_and_mask(r,controller)[0],
                                steps=job['steps'],lr=job['lr'],l2=job['l2'],reduction=job['loss_reduction'])
    # LoReFT baseline: full-support mean objective; explicit adaptation protocol.
    params=list(controller.parameters());opt=torch.optim.AdamW(params,lr=job['lora_lr']);history=[]
    for step in range(job['lora_passes']):
        opt.zero_grad(set_to_none=True);total=0.
        for r in support:
            try:
                loss,_=backend.loss_and_mask(r,controller)
                (loss/len(support)).backward();total+=float(loss.detach())
            finally:controller.clear_mask()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        history.append({'step':step+1,'data_loss_mean':total/len(support),'gradient_norm':float(norm)})
    return history
