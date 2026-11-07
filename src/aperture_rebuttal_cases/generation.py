"""Label-free, no-cache greedy VQA generation, identical policy for every arm."""
import inspect


def question(row):
    return row['question'].strip()+'\nAnswer with a short answer only.'


def training_row(row):
    return {**row,'question':question(row)}


def generate(backend,row,controller=None,max_new_tokens=32):
    import torch
    # Build a prefix using a CONSTANT disposable suffix, never an answer or option.
    public={k:row[k] for k in ('sample_id','group_id','image','question')}
    public.update(question=question(row),candidate_labels=['*'],label='*',label_id=0)
    batch,span,mask,_=backend.batch(public,'*');n=span[0]
    for key in ('input_ids','attention_mask','token_type_ids'):
        if key in batch:batch[key]=batch[key][:,:n]
    mask=mask[:,:n];generated=[]
    model=backend.model;cfg=getattr(backend.base,'generation_config',None)
    eos=getattr(cfg,'eos_token_id',None)
    tokenizer=getattr(backend.processor,'tokenizer',backend.processor)
    if eos is None:eos=tokenizer.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos]);stopped=False
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if controller is not None:controller.set_mask(mask)
                kwargs={'use_cache':False,'return_dict':True}
                if 'logits_to_keep' in inspect.signature(backend.base.forward).parameters:kwargs['logits_to_keep']=1
                logits=model(**batch,**kwargs).logits[:,-1,:].float()
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite generation logits')
                token=int(logits.argmax(-1).item())
                if token in eos:stopped=True;break
                generated.append(token)
                batch['input_ids']=torch.cat((batch['input_ids'],batch['input_ids'].new_tensor([[token]])),dim=1)
                if 'attention_mask' in batch:batch['attention_mask']=torch.cat((batch['attention_mask'],batch['attention_mask'].new_ones((1,1))),dim=1)
                if 'token_type_ids' in batch:batch['token_type_ids']=torch.cat((batch['token_type_ids'],batch['token_type_ids'].new_zeros((1,1))),dim=1)
                mask=torch.cat((mask,mask.new_zeros((1,1))),dim=1)
    finally:
        if controller is not None:controller.clear_mask()
    return {'sample_id':row['sample_id'],'group_id':row['group_id'],
            'pred_answer':tokenizer.decode(generated,skip_special_tokens=True),
            'token_ids':generated,'stopped_eos':stopped,'hit_length_cap':not stopped,
            'policy':'greedy_no_cache_32_tokens','candidate_answers_provided':False}
