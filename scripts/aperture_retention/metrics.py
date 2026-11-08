"""Raw-score retention metrics and paired query-group uncertainty."""
from collections import defaultdict
import numpy as np


def arrays(rows):
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate prediction population')
    predictions=[];losses=[];briers=[];labels=[]
    for r in rows:
        s=np.asarray(r['scores'],dtype=float);y=r['label_id']
        if s.ndim!=1 or len(s)<2 or not np.isfinite(s).all() or type(y) is not int or not 0<=y<len(s):
            raise ValueError('Invalid finite scores/label')
        if not r.get('group_id'):raise ValueError('Missing query group')
        s=s-s.max();lp=s-np.log(np.exp(s).sum());p=np.exp(lp)
        predictions.append(int(s.argmax()));labels.append(y);losses.append(-lp[y])
        briers.append(float(np.square(p-np.eye(len(p))[y]).sum()))
    return np.array(labels),np.array(predictions),np.array(losses),np.array(briers)


def summarize(rows):
    y,p,nll,brier=arrays(rows);classes=range(max(len(r['scores']) for r in rows));f1=[]
    for c in classes:
        tp=int(((y==c)&(p==c)).sum());fp=int(((y!=c)&(p==c)).sum());fn=int(((y==c)&(p!=c)).sum())
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(n=len(rows),accuracy=float((y==p).mean()),macro_f1=float(np.mean(f1)),
                nll=float(nll.mean()),brier=float(brier.mean()),temperature=1.,
                scoring='sum_answer_token_log_probability; raw candidate softmax')


def compare(base,adapted,bootstrap=2000):
    if [(r['sample_id'],r['group_id'],r['label_id']) for r in base]!=[(r['sample_id'],r['group_id'],r['label_id']) for r in adapted]:
        raise ValueError('Paired population/labels/order differ')
    y,p,_,_=arrays(base);yy,q,_,_=arrays(adapted)
    if any(len(a['scores'])!=len(b['scores']) for a,b in zip(base,adapted)):
        raise ValueError('Paired candidate counts differ')
    old=p==y;new=q==yy;delta=new.astype(float)-old.astype(float)
    lost=int((old&~new).sum());fixed=int((~old&new).sum());nold=int(old.sum())
    groups=defaultdict(list)
    for r,d in zip(base,delta):groups[r['group_id']].append(float(d))
    values=list(groups.values());s=np.array([sum(v) for v in values]);n=np.array([len(v) for v in values])
    ci=None
    if bootstrap:
        rng=np.random.default_rng(0);boot=[]
        for _ in range(bootstrap):
            idx=rng.integers(0,len(values),len(values));boot.append(float(s[idx].sum()/n[idx].sum()))
        ci=np.quantile(boot,[.025,.975]).tolist()
    left=summarize(base);right=summarize(adapted)
    return {'baseline':left,'adapted':right,'accuracy_delta':float(delta.mean()),
            'retention_drop':float(-delta.mean()),'macro_f1_delta':right['macro_f1']-left['macro_f1'],
            'nll_change':right['nll']-left['nll'],'original_correct':nold,'correct_to_wrong':lost,
            'wrong_to_correct':fixed,'lost_original_correct_fraction':lost/nold if nold else None,
            'prediction_disagreement':float((p!=q).mean()),
            'max_abs_score_difference':max(float(np.max(np.abs(np.array(a['scores'])-np.array(b['scores'])))) for a,b in zip(base,adapted)),
            'accuracy_delta_ci95':ci,'ci_unit':'query_group_conditional_on_support_seed',
            'groups':len(groups),'few_groups_warning':len(groups)<20}
