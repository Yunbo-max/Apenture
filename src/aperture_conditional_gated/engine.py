"""CG-v1 GPU stages derived from the existing reviewer-case execution contract.
Legacy package remains unchanged. This module adds the conditional loader,
behavioral gate audit and final fit/calibration diagnostics. A CPU test run does
not validate native VLM/CUDA performance. Query requires a sealed selection.
"""
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
import time
import numpy as np
from aperture_rebuttal_cases import store
from aperture_rebuttal_cases.data import load_roles,evaluator
from .plan import variants
from .report import metrics,select


def environment():
    out={}
    for name in ('torch','transformers','peft','pyreft','numpy','Pillow','safetensors'):
        try:out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:out[name]=None
    return out


def binding(unit,plan,variant=None):
    from aperture_comprehensive.worker import check_assets
    return {'unit':unit,'variant':variant,'plan':store.digest(plan),'environment':environment(),
            'runtime':{'APERTURE_BATCH':batch_size(),'batch_contract':'row-preserving-v2'},
            'model_assets':check_assets(unit,Path(unit['output']).parents[2]/'assets',refresh=False),
            'manifests':{k:store.sha(unit[k]) for k in ('support','calibration','query')}}


def setup():
    from aperture_comprehensive.worker import deterministic
    deterministic(0)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Native BF16 CUDA is required; no CPU/offload fallback')
    return torch


def score(backend,rs,controller,vqa=False):
    from aperture_rebuttal_cases.generation import training_row
    result=[]
    B=batch_size()
    if vqa or B<=1:
        for row in rs:
            if vqa:
                with __import__('torch').no_grad():
                    lp,_,_=backend.candidate(training_row(row),row['label'],controller)
                    out={'sample_id':row['sample_id'],'group_id':row['group_id'],
                         'target_nll':-float(lp.mean()),'absolute_sequence_nll':-float(lp.sum())}
                if controller is not None:controller.clear_mask()
            else:
                out={**backend.score(row,controller),'group_id':row['group_id']}
                if not np.isfinite(out['mean_log_scores']).all():raise RuntimeError('Nonfinite candidate scores')
            result.append(out)
        return result
    # Batched non-vqa scoring: one forward per candidate per chunk of rows.
    # Row order, candidate order and per-row probabilities are preserved.
    import torch as _torch
    from aperture_final.numeric import probabilities
    for start in range(0,len(rs),B):
        chunk=rs[start:start+B]
        # Candidate sets/order can vary by question (e.g. ManipBench Q2).
        # Keep the audited single-row policy when a chunk is heterogeneous.
        if any(r['candidate_labels'] != chunk[0]['candidate_labels'] for r in chunk[1:]):
            for row in chunk:
                out={**backend.score(row,controller),'group_id':row['group_id']}
                if not np.isfinite(out['mean_log_scores']).all():raise RuntimeError('Nonfinite candidate scores')
                result.append(out)
            continue
        per_candidate=[];row_contracts=None
        try:
            with _torch.no_grad():
                for c in chunk[0]['candidate_labels']:
                    lps,_,contract,_=backend.candidate_batch(chunk,[c]*len(chunk),controller)
                    if len(lps)!=len(chunk) or len(contract['rows'])!=len(chunk):
                        raise ValueError('Batched score population mismatch')
                    if row_contracts is not None and contract['rows']!=row_contracts:
                        raise ValueError('candidate-specific prompt context')
                    row_contracts=contract['rows']
                    per_candidate.append([float(x) for x in lps])
        finally:
            if controller is not None:controller.clear_mask()
        for i,row in enumerate(chunk):
            scores=[o[i] for o in per_candidate]
            p=probabilities([scores])[0]
            out=dict(sample_id=row['sample_id'],label_id=row['label_id'],mean_log_scores=scores,
                     probabilities=p.tolist(),prediction=int(np.argmax(scores)),
                     prompt_contract_hash=store.digest(row_contracts[i]),image_tokens=row_contracts[i]['image_tokens'],
                     group_id=row['group_id'])
            if not np.isfinite(out['mean_log_scores']).all():raise RuntimeError('Nonfinite candidate scores')
            result.append(out)
    return result


def batch_size():
    import os
    raw=os.environ.get('APERTURE_BATCH','1')
    try:
        value=int(raw)
    except ValueError as exc:
        raise ValueError('APERTURE_BATCH must be a positive integer') from exc
    if value<1:raise ValueError('APERTURE_BATCH must be a positive integer')
    return value


def fit_controller_batched(model,backend,controller,support,job):
    """Same objective and history schema as vigor_handoff.fit_coefficients, with
    chunked batched forwards: one optimizer step per complete support pass.
    loss_reduction 'mean' scales each row loss by 1/N, 'sum' by 1."""
    import torch
    params=[p for p in controller.parameters() if p.requires_grad]
    if not params or job['steps']==0:return []
    optimizer=torch.optim.Adam(params,lr=job['lr'])
    B=max(1,batch_size());N=len(support);history=[]
    for step in range(job['steps']):
        optimizer.zero_grad(set_to_none=True);total=0.
        for start in range(0,N,B):
            chunk=support[start:start+B]
            try:
                lps,_,_,meta=backend.candidate_batch(chunk,None,controller)
                row_losses=torch.stack([-lp/(meta[i]['span'][1]-meta[i]['span'][0])
                                        for i,lp in enumerate(lps)])
                objective=(row_losses.sum()/(N if job['loss_reduction']=='mean' else 1))
                if not torch.isfinite(objective):raise ValueError('non-finite adaptation loss')
                objective.backward();total+=float(row_losses.sum().detach())
            finally:
                controller.clear_mask()
        penalty=job['l2']*sum(p.square().sum() for p in params)
        penalty.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,job.get('clip',1.0),error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):raise ValueError('non-finite controller')
        history.append({'step':step+1,'data_loss_mean':total/N,
                        'penalty':float(penalty.detach()),'reduction':job['loss_reduction'],
                        'objective':total/(N if job['loss_reduction']=='mean' else 1)+float(penalty.detach()),
                        'gradient_norm_before_clip':float(norm)})
    return history


def train_loreft_batched(model,backend,controller,support,job):
    """Batched analogue of adapters.train_control's LoReFT branch: full-support
    mean objective, one optimizer step per pass, row losses scaled by 1/N."""
    import torch
    params=list(controller.parameters());opt=torch.optim.AdamW(params,lr=job['lora_lr']);history=[]
    B=max(1,batch_size());N=len(support)
    for step in range(job['lora_passes']):
        opt.zero_grad(set_to_none=True);total=0.
        for start in range(0,N,B):
            chunk=support[start:start+B]
            try:
                lps,_,_,meta=backend.candidate_batch(chunk,None,controller)
                row_losses=torch.stack([-lp/(meta[i]['span'][1]-meta[i]['span'][0])
                                        for i,lp in enumerate(lps)])
                (row_losses.sum()/N).backward();total+=float(row_losses.sum().detach())
            finally:
                controller.clear_mask()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        history.append({'step':step+1,'data_loss_mean':total/N,'gradient_norm':float(norm)})
    return history


def compare_replay(expected,current,vqa=False):
    if [r['sample_id'] for r in expected]!=[r['sample_id'] for r in current]:raise ValueError('Replay population mismatch')
    field='target_nll' if vqa else 'mean_log_scores'
    diff=max(float(np.max(np.abs(np.asarray(a[field])-np.asarray(b[field])))) for a,b in zip(expected,current))
    if not math.isfinite(diff) or diff>.005:raise RuntimeError(f'Fresh-process replay failed: {diff}; do not widen tolerance')
    return diff


from .audit import audit


def fit(unit,variant,plan):
    torch=setup();folder=Path(unit['output'])/'fits'/variant['name'];b=binding(unit,plan,variant)
    store.verify(Path(unit['output'])/'audit',binding(unit,plan));roles=load_roles(unit,images=True)
    if (folder/'DONE.json').exists():store.verify(folder,b);return
    if folder.exists() and any(folder.iterdir()):
        archive=folder.with_name(folder.name+'-interrupted-'+str(time.time_ns()));folder.rename(archive)
    folder.mkdir(parents=True,exist_ok=True)
    from .adapters import loaded,train_control
    from aperture_rebuttal_cases.generation import training_row
    from vigor_handoff.worker import train_lora
    from aperture_comprehensive.numeric import score_arrays
    from aperture_final.numeric import fit_temperature
    vqa=unit['task_type']=='vqa';support=[training_row(r) for r in roles['support']] if vqa else roles['support']
    job={**unit,**variant};start=time.perf_counter();history=[];extraction=0.;updates=visits=0
    torch.cuda.reset_peak_memory_stats()
    with loaded(unit,variant) as (model,backend,controller):
        extraction_peak_allocated=torch.cuda.max_memory_allocated()
        extraction_peak_reserved=torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();fit_start=time.perf_counter()
        # Extraction occurs inside loaded(); report it explicitly and include it.
        extraction=getattr(controller,'extraction_seconds',0.) if controller else 0.
        trained=controller if controller is not None else model
        before={n:p.detach().float().cpu().clone() for n,p in trained.named_parameters() if p.requires_grad}
        if variant['engine'] in ('lora','dora'):
            history,updates,count=train_lora(model,backend,support,job)
            visits=len(support)*variant['lora_passes'];model.save_pretrained(folder/'adapter',safe_serialization=True)
        elif controller is not None:
            if variant['engine'] in ('aperture','conditional_gated') and batch_size()>1:
                history=fit_controller_batched(model,backend,controller,support,job)
            elif variant['engine']=='loreft' and batch_size()>1:
                history=train_loreft_batched(model,backend,controller,support,job)
            else:
                history=train_control(model,backend,controller,support,job)
            count=controller.num_scalars();updates=len(history);visits=updates*len(support)
            torch.save(controller.payload(),folder/'controller.pt')
            store.atomic(folder/'basis.json',getattr(controller,'basis_spectra',{}))
        else:count=0
        if variant['engine']!='frozen':
            changed=sum(float((p.detach().float().cpu()-before[n]).square().sum()) for n,p in trained.named_parameters() if n in before)
            if not math.isfinite(changed) or changed<=0:raise RuntimeError('No finite nonzero trained-state update')
        torch.cuda.synchronize();fit_seconds=time.perf_counter()-fit_start+extraction
        model.eval();model.gradient_checkpointing_disable()
        for p in model.parameters():p.requires_grad_(False)
        if controller is not None:
            for p in controller.parameters():p.requires_grad_(False)
        cal=score(backend,roles['calibration'],controller,vqa)
        if vqa:
            cal_nll=float(np.mean([r['target_nll'] for r in cal]));temp=1.
        else:
            ss,ys=score_arrays(cal,unit['classes']);cal_nll=metrics(cal)['nll'];temp=fit_temperature(ss,ys,penalty=.01)['temperature']
        # End-state support scores, not the last pre-update training-history loss.
        final_support=score(backend,roles['support'],controller,vqa)
        store.write_rows(folder/'calibration.jsonl',cal);store.write_rows(folder/'support_final.jsonl',final_support)
        store.atomic(folder/'deployment.json',{'variant':variant,'model_id':unit['model_id'],
            'model_assets':b['model_assets'],'layers':unit['layers'],'mask':unit['mask'],
            'image_size':unit['image_size'],'temperature':temp})
        state_files=[folder/'deployment.json']+[p for p in folder.rglob('*') if p.is_file() and (p.suffix in ('.pt','.safetensors') or p.parent.name=='adapter')]
        basis_scalars=sum(v.numel() for v in getattr(controller,'bases',{}).values()) if controller else 0
        from .adapters import gate_evidence
        gate_records={'support':gate_evidence(backend,support,controller),
                      'calibration':gate_evidence(backend,[training_row(r) for r in roles['calibration']] if vqa else roles['calibration'],controller)}
        if controller is not None and hasattr(controller,'diagnostics'):
            gate_records['operator_and_gate_parameters']=controller.diagnostics()
        store.atomic(folder/'gate_diagnostics.json',gate_records)
        stats={'name':variant['name'],'family':variant['family'],'engine':variant['engine'],
            'batch_size':batch_size(),'calibration_nll':cal_nll,'temperature':temp,'selection_criterion':'raw_calibration_nll',
            'optimized_scalars':count,'basis_scalars':basis_scalars,'serialized_state_bytes':sum(p.stat().st_size for p in state_files),
            'fit_seconds_including_extraction':fit_seconds,'basis_extraction_seconds':extraction,
            'process_load_and_fit_seconds':time.perf_counter()-start,'optimizer_updates':updates,'support_visits':visits,
            'basis_visits':len(support) if variant['engine'] in ('aperture','conditional_gated') and variant['basis_mode']!='random' else 0,
            'peak_allocated_bytes':max(extraction_peak_allocated,torch.cuda.max_memory_allocated()),
            'peak_reserved_bytes':max(extraction_peak_reserved,torch.cuda.max_memory_reserved()),
            'device':torch.cuda.get_device_name(0),'history':history,'query_forwards':0,'total_labels':unit['total_labels'],
            'final_support_metrics':None if vqa else metrics(final_support),
            'final_calibration_metrics':None if vqa else metrics(cal),
            'method_version':unit['method_version'],'historical_best_reproduced':False,
            'timing_scope':'basis extraction + optimization + update check/state saving; excludes load/audit/calibration',
            'compute_matched':False,'replay_status':'required_in_fresh_evaluation_process'}
        if not math.isfinite(cal_nll):raise RuntimeError('Nonfinite calibration objective')
        store.atomic(folder/'fit.json',stats);store.seal(folder,b)


def seal_selection(unit,plan):
    folder=Path(unit['output']);records=[];seals={}
    for v in unit['variants']:
        path=folder/'fits'/v['name'];store.verify(path,binding(unit,plan,v))
        records.append(store.read(path/'fit.json'));seals[v['name']]=store.sha(path/'DONE.json')
    chosen=select(records)
    evaluate=[v['name'] for v in unit['variants']] if unit['evaluate_policy']=='all' else list(chosen.values())
    receipt={'selected':chosen,'evaluate':evaluate,'fit_seals':seals,'query_metrics_used':False,'criterion':'raw_calibration_nll',
             'selection_budget':len(records),'tie_break':'lexicographic_predeclared_variant_name'}
    if unit.get('measure_latency',False):
        budget=next(r['fit_seconds_including_extraction'] for r in records if r['name']=='conditional-gated')
        feasible=[r for r in records if r['engine']=='lora' and r['fit_seconds_including_extraction']<=budget]
        receipt['compute_budget_comparator']={'budget_seconds':budget,'feasible':[r['name'] for r in feasible],
            'selected':min(feasible,key=lambda r:(r['calibration_nll'],r['name']))['name'] if feasible else None,
            'note':'calibration selection among measured feasible schedules; not exact compute equality'}
    store.atomic(folder/'selection.json',receipt,immutable=True)
    store.atomic(folder/'selection_receipt'/'selection.json',receipt,immutable=True)
    store.seal(folder/'selection_receipt',{'unit':unit['key'],'plan':store.digest(plan)})


def evaluate(unit,variant,plan,repo):
    torch=setup();root=Path(unit['output']);fitdir=root/'fits'/variant['name'];folder=root/'evaluation'/variant['name']
    selection=store.read(root/'selection.json');store.verify(root/'selection_receipt')
    if selection!=store.read(root/'selection_receipt'/'selection.json'):raise ValueError('Selection receipt changed')
    if variant['name'] not in selection['evaluate']:raise ValueError('Variant not preselected for query scoring')
    store.verify(fitdir,binding(unit,plan,variant))
    if selection['fit_seals'][variant['name']]!=store.sha(fitdir/'DONE.json'):raise ValueError('Selection fit changed')
    b={'fit_seal':store.sha(fitdir/'DONE.json'),'selection':store.sha(root/'selection.json'),'identity':binding(unit,plan,variant)}
    if (folder/'DONE.json').exists():store.verify(folder,b);return
    roles=load_roles(unit,images=True);vqa=unit['task_type']=='vqa'
    from .adapters import loaded
    from aperture_rebuttal_cases.generation import generate
    with loaded(unit,variant,fitdir) as (model,backend,controller):
        cal=score(backend,roles['calibration'],controller,vqa)
        replay=compare_replay(store.rows(fitdir/'calibration.jsonl'),cal,vqa)
        folder.mkdir(parents=True,exist_ok=True);store.atomic(folder/'binding.json',b,immutable=True)
        destination=folder/'predictions.jsonl';existing=store.rows(destination) if destination.exists() else []
        if [r['sample_id'] for r in existing]!=[r['sample_id'] for r in roles['query'][:len(existing)]]:raise ValueError('Interrupted query prefix mismatch')
        # Append only complete flushed records. A malformed tail is an explicit error.
        with destination.open('a',encoding='utf-8') as stream:
            for row in roles['query'][len(existing):]:
                out=generate(backend,row,controller) if vqa else score(backend,[row],controller)[0]
                stream.write(json.dumps(out,allow_nan=False)+'\n');stream.flush();existing.append(out)
        if len(existing)!=len(roles['query']):raise ValueError('Incomplete query population')
        if vqa:
            official=evaluator(plan['settings'],repo)
            data=[{'pred_answer':r['pred_answer'],'gt_answers':q['answers']} for r,q in zip(existing,roles['query'])]
            result={'vqa_accuracy':float(official.TextVQAAccuracyEvaluator().eval_pred_list(data)),
                'n':len(data),'length_cap_rate':sum(r['hit_length_cap'] for r in existing)/len(data),
                'task_scope':'free short-answer VQA only; no reasoning/dialogue claim'}
        else:
            result={'raw':metrics(existing)};temperature=store.read(fitdir/'fit.json')['temperature'];scaled=[]
            for row in existing:
                z=np.asarray(row['mean_log_scores'],float);z=(z-z.max())/temperature;a=np.exp(z);a/=a.sum()
                scaled.append({**row,'mean_log_scores':z.tolist(),'probabilities':a.tolist()})
            result['temperature_scaled']=metrics(scaled);result['temperature']=temperature
        result['calibration_replay_max_delta']=replay;result['batch_size']=batch_size()
        if unit.get('measure_latency',False):
            samples=roles['query'][:12]
            for r in samples[:2]:backend.score(r,controller)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(3):
                torch.cuda.synchronize();start=time.perf_counter()
                for r in samples:backend.score(r,controller)
                torch.cuda.synchronize();times.append((time.perf_counter()-start)/len(samples))
            result['latency']={'seconds_per_query':times,'warmups':2,'queries_per_repeat':len(samples),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'policy':'full candidate scoring including preprocessing; LoRA unmerged',
                'no_merged_lora_speed_claim':True}
            if variant['engine']=='lora':
                from aperture_comprehensive.backend import Backend
                from .adapters import native_job
                merged=model.merge_and_unload(safe_merge=True)
                merged_backend=Backend(native_job(unit,variant),merged,backend.processor)
                try:
                    delta=compare_replay(cal,score(merged_backend,roles['calibration'],None))
                except RuntimeError as exc:
                    result['merged_lora_latency']={'status':'not_equivalent_at_declared_tolerance','reason':str(exc)}
                else:
                    for r in samples[:2]:merged_backend.score(r)
                    merged_times=[]
                    for _ in range(3):
                        torch.cuda.synchronize();tick=time.perf_counter()
                        for r in samples:merged_backend.score(r)
                        torch.cuda.synchronize();merged_times.append((time.perf_counter()-tick)/len(samples))
                    result['merged_lora_latency']={'status':'measured','seconds_per_query':merged_times,'calibration_score_delta':delta}
        store.atomic(folder/'metrics.json',result);store.seal(folder,b)
