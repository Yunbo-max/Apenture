"""Native six-backbone scoring with explicit visual rows and decoder sites."""
from contextlib import contextmanager
from copy import deepcopy
import gc
import inspect
import re

import numpy as np
from PIL import Image
import torch
from torch import nn

from aperture_table.backend import answer_span, answer_logprobs, _decoder_modules
from aperture_final.numeric import probabilities
from vigor_handoff.protocol import digest
from .protocol import MODELS


class SliceSite(nn.Module):
    """Identity site exposes one fused projection slice to the existing controller."""
    def __init__(self, width, reference):
        super().__init__()
        self.out_features=width
        self.anchor=nn.Parameter(reference.new_empty(0),requires_grad=False)

    def forward(self, x):
        return x


def phi_sites(model, layers):
    found={}
    for name,module in list(model.named_modules()):
        match=re.search(r'(?:^|\.)layers\.(\d+)\.self_attn\.qkv_proj$',name)
        if match and int(match[1]) in layers:
            layer=int(match[1]);width=module.in_features
            if module.out_features!=3*width or layer in found:
                raise ValueError('unsupported or ambiguous Phi fused QKV layout')
            if not hasattr(module,'aperture_k_site'):
                module.add_module('aperture_k_site',SliceSite(width,module.weight))
                module.add_module('aperture_v_site',SliceSite(width,module.weight))
                def split(mod,inputs,out,width=width):
                    return torch.cat((out[...,:width],mod.aperture_k_site(out[...,width:2*width]),
                                      mod.aperture_v_site(out[...,2*width:])),dim=-1)
                module._aperture_slice_handle=module.register_forward_hook(split)
            found[layer]=module
    if set(found)!=set(layers):
        raise ValueError('missing Phi decoder projections')
    return [(layer,kind,getattr(found[layer],f'aperture_{kind.lower()}_site')) for layer in layers for kind in ('K','V')]


def decoder_sites(model, layers):
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    if base.config.model_type=='phi3_v':return phi_sites(base,layers)
    modules=_decoder_modules(base)
    if any((l,k) not in modules for l in layers for k in ('K','V')):
        raise ValueError('missing decoder K/V sites')
    return [(l,k,modules[(l,k)][1]) for l in layers for k in ('K','V')]


def lora_targets(model):
    if model.config.model_type!='phi3_v':
        from aperture_table.backend import lora_targets as native_targets
        return native_targets(model)
    found={}
    for name,module in model.named_modules():
        match=re.search(r'(?:^|\.)layers\.(\d+)\.self_attn\.(qkv_proj|o_proj)$',name)
        if match:
            key=(int(match[1]),match[2])
            if key in found:raise ValueError('ambiguous Phi LoRA targets')
            found[key]=name
    expected={(l,k) for l in range(model.config.num_hidden_layers) for k in ('qkv_proj','o_proj')}
    if set(found)!=expected:raise ValueError('incomplete Phi decoder LoRA targets')
    return [found[k] for k in sorted(found)]


def checked_mask(batch, config, family, answer_start, images, mode):
    ids=batch['input_ids']
    if family=='phi3_v':visual=ids<0
    else:
        token=(getattr(config,'image_token_index',None) if family=='gemma3'
               else getattr(config,'img_context_token_id',None) if family=='internvl_chat'
               else getattr(config,'image_token_id',None))
        if not isinstance(token,int):raise ValueError('missing explicit visual token ID')
        visual=ids==token
    if not visual.any() or visual[:,answer_start:].any():raise ValueError('missing or answer-overlapping visual rows')
    positions=visual[0].nonzero().flatten()
    cuts=(positions[1:]!=positions[:-1]+1).nonzero().flatten()+1
    groups=list(torch.tensor_split(positions,cuts.tolist()))
    if ids.shape[0]!=1 or len(groups)!=images:raise ValueError('wrong image group count')
    if family in ('qwen2_5_vl','qwen3_vl'):
        grid=batch.get('image_grid_thw');merge=getattr(config.vision_config,'spatial_merge_size',None)
        if grid is None or grid.shape!=(images,3) or merge is None:raise ValueError('Qwen image grid count mismatch')
        counts=grid.prod(-1)//(merge**2)
        if counts.tolist()!=[len(g) for g in groups]:raise ValueError('Qwen visual token/grid mismatch')
    elif family=='gemma3':
        types=batch.get('token_type_ids')
        if types is None or not torch.equal(types==1,visual):raise ValueError('Gemma image token types mismatch')
        if any(len(g)!=config.mm_tokens_per_image for g in groups):raise ValueError('Gemma image expansion mismatch')
    elif family=='internvl_chat':
        if any(len(g)!=config.comprehensive_image_tokens for g in groups):raise ValueError('InternVL image expansion mismatch')
        if batch['pixel_values'].shape[0]!=images:raise ValueError('InternVL image tensor count mismatch')
    elif family=='phi3_v':
        if any(not torch.all(ids[0,g]==-(i+1)) for i,g in enumerate(groups)):
            raise ValueError('Phi numbered image placeholder mismatch')
        if batch['image_sizes'].shape[0]!=images:raise ValueError('Phi image size count mismatch')
    mask=torch.zeros_like(visual)
    if mode=='post' and images==2:mask[0,groups[1]]=True
    elif mode=='all_visual':mask=visual
    else:raise ValueError('invalid image mask policy')
    return mask


class Backend:
    def __init__(self,job,model,processor):
        self.job=job;self.model=model;self.processor=processor
        self.base=model.get_base_model() if hasattr(model,'get_base_model') else model
        self.config=self.base.config;self.device=next(model.parameters()).device

    def _images(self,row):
        paths=[row['pre_image'],row['post_image']] if self.job['kind']=='paired' else [row['image']]
        images=[]
        for path in paths:
            with Image.open(path) as im:im=im.convert('RGB')
            if self.job['family']!='gemma3':im=im.resize((self.job['image_size'],)*2,Image.Resampling.BICUBIC)
            images.append(im)
        return images

    def batch(self,row,candidate):
        if candidate not in row['candidate_labels']:raise ValueError('unknown candidate')
        family=self.job['family'];images=self._images(row)
        question=row['question']
        if len(images)==2:question='The first image is before the event; the second is after the event. '+question
        if family=='internvl_chat':
            tokenizer=self.processor
            template=deepcopy(self.base.conv_template)
            template.system_message='Answer the visual question with exactly one candidate answer and no explanation.'
            tokens='<img>'+'<IMG_CONTEXT>'*self.config.comprehensive_image_tokens+'</img>'
            template.append_message(template.roles[0],('\n'.join([tokens]*len(images)))+'\n'+question)
            template.append_message(template.roles[1],None)
            prompt=template.get_prompt()
            prefix=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
            full=tokenizer(prompt+candidate,return_tensors='pt',add_special_tokens=False)
            mean=torch.tensor((.485,.456,.406)).view(3,1,1);std=torch.tensor((.229,.224,.225)).view(3,1,1)
            pixels=torch.stack([(torch.from_numpy(np.array(im,dtype=np.float32)).permute(2,0,1)/255-mean)/std for im in images])
            full['pixel_values']=pixels
        elif family=='phi3_v':
            prompt='<|user|>\n'+''.join(f'<|image_{i+1}|>\n' for i in range(len(images)))+question+'<|end|>\n<|assistant|>\n'
            prefix=self.processor(prompt,images=images,return_tensors='pt')
            full=self.processor(prompt+candidate,images=images,return_tensors='pt')
        else:
            messages=[{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':question}]}]
            prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            kwargs=dict(images=[images] if family=='gemma3' else images,return_tensors='pt',add_special_tokens=False)
            if family=='gemma3':kwargs.update(do_pan_and_scan=False,return_mm_token_type_ids=True)
            prefix=self.processor(text=[prompt],**kwargs);full=self.processor(text=[prompt+candidate],**kwargs)
        span=answer_span(prefix['input_ids'],full['input_ids'])
        for k in ('pixel_values','image_grid_thw','image_sizes'):
            if k in prefix and (k not in full or not torch.equal(prefix[k],full[k])):
                raise ValueError('candidate changed image processing')
        mask=checked_mask(full,self.config,family,span[0],len(images),self.job['mask'])
        contract={'prefix_ids':prefix['input_ids'][0].tolist(),'candidates':row['candidate_labels'],
                  'image_tokens':int(mask.sum()),'family':family,'image_size':self.job['image_size'],
                  'pixel_shape':list(full['pixel_values'].shape),'mask':self.job['mask']}
        full={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in full.items()}
        full['pixel_values']=full['pixel_values'].to(dtype=next(self.model.parameters()).dtype)
        return full,span,mask.to(self.device),contract

    def candidate(self,row,candidate,controller=None):
        batch,span,mask,contract=self.batch(row,candidate)
        if controller is not None:controller.set_mask(mask)
        n=span[1]-span[0]
        if self.job['family']=='internvl_chat':
            ids=batch['input_ids'];lm=self.base.language_model
            embeddings=lm.get_input_embeddings()(ids).clone()
            vision=self.base.extract_feature(batch['pixel_values'])
            slots=ids==self.config.img_context_token_id
            if int(slots.sum())!=vision.shape[0]*vision.shape[1]:raise ValueError('InternVL embedding/token mismatch')
            embeddings[slots]=vision.reshape(-1,vision.shape[-1])
            out=lm(inputs_embeds=embeddings,attention_mask=batch.get('attention_mask'),
                   use_cache=False,logits_to_keep=n+1,return_dict=True)
        else:
            kwargs={'use_cache':False,'return_dict':True}
            if 'logits_to_keep' in inspect.signature(self.base.forward).parameters:kwargs['logits_to_keep']=n+1
            elif self.job['family']!='phi3_v':raise RuntimeError('native selective-logit API unavailable')
            out=self.model(**batch,**kwargs)
        return answer_logprobs(out.logits,batch['input_ids'],span),mask,contract

    def loss_and_mask(self,row,controller=None):
        lp,mask,_=self.candidate(row,row['label'],controller)
        return -lp.mean(),mask

    def score(self,row,controller=None):
        scores=[];contract=None
        try:
            with torch.no_grad():
                for c in row['candidate_labels']:
                    lp,_,ctx=self.candidate(row,c,controller)
                    if contract is not None and ctx!=contract:raise ValueError('candidate-specific prompt context')
                    contract=ctx;scores.append(float(lp.sum()))
            p=probabilities([scores])[0]
            return dict(sample_id=row['sample_id'],label_id=row['label_id'],mean_log_scores=scores,
                        probabilities=p.tolist(),prediction=int(np.argmax(scores)),
                        prompt_contract_hash=digest(contract),image_tokens=contract['image_tokens'])
        finally:
            if controller is not None:controller.clear_mask()

    # --- batched paths (APERTURE_BATCH>1): right-padded, row order preserved ---

    def batch_rows(self, rows, candidates=None):
        """One padded batch for `rows`; candidates[i] is row i's answer suffix
        (default: the row's own label). Returns (batch_dict, meta) where
        meta[i] = {'ids','span','mask','sample_id'}. Rows are RIGHT-padded to
        the longest sequence so every real token keeps the exact position it
        has in a single-row forward (rotary/mrope positions unchanged);
        attention_mask pads with 0, per-row controller masks pad with False.
        Row order is preserved. Spans need no offset because real tokens start
        at position 0 in every row."""
        rows=list(rows)
        if not rows:
            raise ValueError('empty batch')
        cands = list(candidates) if candidates is not None else [r['label'] for r in rows]
        if len(cands)!=len(rows):
            raise ValueError('one candidate suffix per batch row is required')
        per = [self.batch(r, c) for r, c in zip(rows, cands)]
        fulls = [p[0] for p in per]
        keys = set(fulls[0])
        if any(set(f)!=keys for f in fulls):
            raise ValueError('inconsistent native processor fields across batch rows')
        batch = {}
        sequence_keys=('input_ids','attention_mask','token_type_ids','mm_token_type_ids')
        maxlen=max(f['input_ids'].shape[1] for f in fulls)
        for k in sorted(keys):
            tensors=[f[k] for f in fulls]
            if not all(torch.is_tensor(t) for t in tensors):
                raise ValueError(f'unsupported non-tensor native processor field: {k}')
            if k in sequence_keys:
                stacked=[]
                for f,t in zip(fulls,tensors):
                    if t.ndim!=2 or t.shape!=f['input_ids'].shape:
                        raise ValueError(f'native token field shape mismatch: {k}')
                    n=maxlen-t.shape[1]
                    if n:
                        t=torch.cat([t,torch.zeros(t.shape[0],n,dtype=t.dtype,device=t.device)],dim=1)
                    stacked.append(t)
                batch[k]=torch.cat(stacked,dim=0)
            else:
                if k=='pixel_values' and self.job['family']=='gemma3':
                    # Only a CHW image needs a leading image axis. Qwen's
                    # (total_patches, patch_dim) is already flattened and must
                    # concatenate along patches, including unequal grids.
                    tensors=[t.unsqueeze(0) if t.ndim==3 else t for t in tensors]
                if any(t.ndim!=tensors[0].ndim or t.shape[1:]!=tensors[0].shape[1:] for t in tensors):
                    raise ValueError(f'incompatible native batch field shapes: {k}')
                batch[k]=torch.cat(tensors,dim=0)
        meta = [{'ids': f['input_ids'][0], 'span': p[1], 'mask': p[2],
                 'sample_id': r['sample_id'],
                 'image_tokens': int(p[2].sum()), 'contract': p[3]}
                for f, p, r in zip(fulls, per, rows)]
        return batch, meta

    def forward_batch(self, batch):
        family = self.job['family']
        if family == 'internvl_chat':
            ids = batch['input_ids']
            lm = self.base.language_model
            embeddings = lm.get_input_embeddings()(ids).clone()
            vision = self.base.extract_feature(batch['pixel_values'])
            slots = ids == self.config.img_context_token_id
            if int(slots.sum()) != vision.shape[0] * vision.shape[1]:
                raise ValueError('InternVL embedding/token mismatch')
            embeddings[slots] = vision.reshape(-1, vision.shape[-1])
            return lm(inputs_embeds=embeddings, attention_mask=batch.get('attention_mask'),
                      use_cache=False, return_dict=True)
        kwargs = {'use_cache': False, 'return_dict': True}
        return self.model(**batch, **kwargs)

    def candidate_batch(self, rows, candidates=None, controller=None):
        """Batched analogue of candidate(): returns (per_row_lps (B,), mask (B,seq), contract, meta).
        Right-padded rows keep token positions identical to single-row forwards;
        logits are full-sequence so per-row span slicing needs no offset."""
        batch, meta = self.batch_rows(rows, candidates)
        maxlen = batch['input_ids'].shape[1]
        masks = []
        for m in meta:
            n = maxlen - m['mask'].shape[1]
            if n:
                m['mask'] = torch.cat([m['mask'], torch.zeros(m['mask'].shape[0], n, dtype=torch.bool,
                                                              device=m['mask'].device)], dim=1)
            masks.append(m['mask'])
        mask = torch.cat(masks, dim=0)
        if controller is not None:
            controller.set_mask(mask)
        out = self.forward_batch(batch)
        lps = []
        for i, m in enumerate(meta):
            start = m['span'][0]; end = m['span'][1]
            selected = out.logits[i, start - 1:end - 1].float()
            logp = torch.log_softmax(selected, -1)
            lp = logp.gather(1, m['ids'][start:end].view(-1, 1)).flatten()
            lps.append(lp.sum())
        contract = {'batch': len(rows), 'family': self.job['family'],
                    'image_tokens': [m['image_tokens'] for m in meta],
                    'rows': [m['contract'] for m in meta]}
        return torch.stack(lps), mask, contract, meta


@contextmanager
def loaded(job,restore=None):
    import transformers
    from pathlib import Path
    from packaging.version import Version
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from vigor_handoff.core import ResidualController
    if not Version('4.57.1')<=Version(transformers.__version__)<Version('5'):
        raise RuntimeError('requires Transformers >=4.57.1,<5')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    family=job['family'];path=job['model_path'];model=controller=None
    with sdpa_kernel(SDPBackend.MATH):
        try:
            kwargs=dict(local_files_only=True,device_map={'':0},torch_dtype=torch.bfloat16)
            if family=='internvl_chat':
                model=transformers.AutoModel.from_pretrained(path,trust_remote_code=True,use_flash_attn=False,**kwargs)
                processor=transformers.AutoTokenizer.from_pretrained(path,trust_remote_code=True,use_fast=False,local_files_only=True)
                token=int(processor.convert_tokens_to_ids('<IMG_CONTEXT>'))
                model.img_context_token_id=token;model.config.img_context_token_id=token
                size=job['image_size'];patch=model.config.vision_config.patch_size
                model.config.comprehensive_image_tokens=int((size//patch)**2*model.downsample_ratio**2)
                model.language_model.config.use_cache=False
                # The wrapper has no checkpoint API; expose its native decoder's API.
                for method in ('gradient_checkpointing_enable','gradient_checkpointing_disable','enable_input_require_grads'):
                    setattr(model,method,getattr(model.language_model,method))
            elif family=='phi3_v':
                processor=transformers.AutoProcessor.from_pretrained(path,trust_remote_code=True,local_files_only=True,num_crops=4)
                config=transformers.AutoConfig.from_pretrained(path,trust_remote_code=True,local_files_only=True)
                config._attn_implementation='eager'
                config._attn_implementation_internal='eager'
                model=transformers.AutoModelForCausalLM.from_pretrained(path,config=config,trust_remote_code=True,attn_implementation='eager',**kwargs)
            else:
                processor=transformers.AutoProcessor.from_pretrained(path,local_files_only=True)
                model=getattr(transformers,MODELS[job['model_key']][2]).from_pretrained(path,attn_implementation='sdpa',**kwargs)
            if model.config.model_type!=family or getattr(model,'is_quantized',False):raise ValueError('wrong or quantized model')
            if {p.device.type for p in model.parameters()}!={'cuda'}:raise ValueError('CPU/meta offload is forbidden')
            for p in model.parameters():p.requires_grad_(False)
            if job['arm'].startswith('lora'):
                from peft import LoraConfig,get_peft_model,PeftModel
                if restore:model=PeftModel.from_pretrained(model,str(Path(restore)/'adapter'),is_trainable=False,local_files_only=True)
                else:
                    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=lora_targets(model),bias='none',task_type='CAUSAL_LM'))
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
                    model.enable_input_require_grads()
                if any('lora_' not in n or any(x in n.lower() for x in ('vision','visual')) for n,p in model.named_parameters() if p.requires_grad):
                    raise ValueError('trainable weights escaped decoder LoRA')
            elif restore and job['arm'] in ('aperture','random'):
                payload=torch.load(Path(restore)/'controller.pt',weights_only=True,map_location='cpu')
                bases={(int(k.split(':')[0]),k.split(':')[1]):v for k,v in payload['bases'].items()}
                controller=ResidualController(decoder_sites(model,job['layers']),bases,payload['alpha'],'full')
                controller.restore(payload)
            if restore:
                for p in model.parameters():p.requires_grad_(False)
            model.eval();model.config.use_cache=False
            yield model,Backend(job,model,processor),controller
        finally:
            if controller is not None:controller.close()
            model=None;gc.collect();torch.cuda.empty_cache()
