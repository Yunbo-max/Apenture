"""Pinned full text datasets and label-free sum-log-likelihood scoring."""
from pathlib import Path
from aperture_rebuttal_cases import store

DATASETS={
 'arc_challenge':{'id':'allenai/ai2_arc','config':'ARC-Challenge','split':'test','count':1172,
                   'revision':'d67d66ae44ae69ff0c82ba64779a3736b2310ce4'},
 'boolq':{'id':'google/boolq','config':'default','split':'validation','count':3270,
          'revision':'9aa7c5589ad6dbb19e78b619ed88a79f06e99aad'},
}


def normalize(name,data):
    output=[]
    for i,r in enumerate(data):
        if name=='arc_challenge':
            choices=r['choices'];labels=choices['label'];candidates=choices['text']
            if r['answerKey'] not in labels or len(set(labels))!=len(labels):raise ValueError('Bad ARC answer key')
            row={'sample_id':str(r['id']),'group_id':str(r['id']),'question':r['question']+'\nAnswer the science question briefly.',
                 'candidates':candidates,'label_id':labels.index(r['answerKey'])}
        elif name=='boolq':
            if type(r['answer']) is not bool:raise ValueError('BoolQ answer must be bool')
            row={'sample_id':f'boolq-{i:05d}','group_id':store.digest(r['passage']),
                 'question':'Passage: '+r['passage']+'\nQuestion: '+r['question']+'\nAnswer yes or no.',
                 'candidates':['no','yes'],'label_id':int(r['answer'])}
        else:raise ValueError('Unknown text dataset')
        if len(row['candidates'])<2 or any(not isinstance(c,str) or not c for c in row['candidates']):raise ValueError('Bad text candidates')
        output.append(row)
    if len({r['sample_id'] for r in output})!=len(output):raise ValueError('Duplicate text IDs')
    return output


def check_count(name,rows):
    if len(rows)!=DATASETS[name]['count']:raise ValueError('Incomplete official text population; no silent subset')


def prepare(root):
    from datasets import load_dataset
    root=Path(root);provenance={'protocol':'aperture-retention-text-v1','datasets':DATASETS,'files':{},
                             'prompt_policy':'native chat template, no examples, no OCR, no image, sum answer-token log probabilities'}
    for name,spec in DATASETS.items():
        raw=load_dataset(spec['id'],spec['config'],split=spec['split'],revision=spec['revision'])
        rows=normalize(name,raw);check_count(name,rows)
        p=root/(name+'.jsonl');store.write_rows(p,rows);provenance['files'][name]=store.sha(p)
    store.atomic(root/'provenance.json',provenance,immutable=True)
    return {'prepared':{k:v['count'] for k,v in DATASETS.items()},'output':str(root),'training':False}


def public_row(row):
    return {k:row[k] for k in ('question','candidates')}


def score_text(model,processor,row,controller=None):
    """Text-only: retain installed controller, give it an all-false visual mask."""
    import inspect
    import torch
    row=public_row(row);tokenizer=getattr(processor,'tokenizer',processor)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':row['question']}],tokenize=False,add_generation_prompt=True)
    prefix=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids']
    device=next(model.parameters()).device;out=[];contracts=[]
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    cfg=base.config;text_cfg=getattr(cfg,'text_config',cfg)
    for candidate in row['candidates']:
        batch=tokenizer(prompt+candidate,return_tensors='pt',add_special_tokens=False)
        ids=batch['input_ids'];start=prefix.shape[1];end=ids.shape[1]
        if end<=start or not torch.equal(ids[:,:start],prefix):raise ValueError('Text answer does not preserve exact prompt prefix')
        limit=getattr(text_cfg,'max_position_embeddings',None)
        if limit and end>limit:raise ValueError('Text context exceeds native maximum; no truncation')
        for token in (getattr(cfg,'image_token_index',None),getattr(cfg,'image_token_id',None)):
            if isinstance(token,int) and (ids==token).any():raise ValueError('Image placeholder in text-only prompt')
        batch={k:v.to(device) for k,v in batch.items() if k in ('input_ids','attention_mask')}
        mask=torch.zeros_like(batch['input_ids'],dtype=torch.bool)
        if controller is not None:controller.set_mask(mask)
        kwargs={'use_cache':False,'return_dict':True}
        sig=inspect.signature(base.forward).parameters
        if 'logits_to_keep' in sig:kwargs['logits_to_keep']=end-start+1
        try:
            with torch.no_grad():
                logits=model(**batch,**kwargs).logits
                # Native selective-logit path retains the final answer span + predictor.
                if logits.shape[1]==end:part=logits[:,start-1:end-1]
                elif logits.shape[1]==end-start+1:part=logits[:,:-1]
                else:raise ValueError('Unexpected native text logits shape')
                targets=batch['input_ids'][:,start:end]
                lp=part.float().log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
                if not torch.isfinite(lp).all():raise ValueError('Nonfinite text likelihood')
                out.append(float(lp.sum()))
        finally:
            if controller is not None:controller.clear_mask()
        contracts.append({'prefix':prefix[0].tolist(),'candidate':candidate,'ids':ids[0].tolist()})
    return {'scores':out,'prompt_contract_hash':store.digest(contracts),'image_tokens':0}
