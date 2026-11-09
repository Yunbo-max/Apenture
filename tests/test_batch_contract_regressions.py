"""CPU reproducers against production function bodies, without downloading a VLM.

Tokenizer/model outputs are small deterministic boundary fixtures. The real
score(), batch_rows() and plan-freezing logic are executed; no GPU qualification
or benchmark conclusion follows from these tests.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aperture_rebuttal_cases import engine, plan, store


def probabilities(scores):
    values = np.asarray(scores, dtype=float)
    values = np.exp(values-values.max(axis=1, keepdims=True))
    return values/values.sum(axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def cpu_probability_boundary(monkeypatch):
    # Unrelated numeric code is not under test here; avoid native dependencies.
    numeric = ModuleType('aperture_final.numeric')
    numeric.probabilities = probabilities
    monkeypatch.setitem(sys.modules, 'aperture_final.numeric', numeric)
    monkeypatch.setenv('APERTURE_BATCH', '1')


class ScoreBackend:
    """Exact symbolic answer scores reveal wrong candidate-index assignments."""
    def __init__(self):
        self.batch_calls = 0
        self.row_calls = 0

    @staticmethod
    def contract(row):
        return {'sample':row['sample_id'], 'candidates':row['candidate_labels'], 'image_tokens':2}

    def score(self, row, controller=None):
        self.row_calls += 1
        scores = [row['lookup'][c] for c in row['candidate_labels']]
        p = probabilities([scores])[0]
        return dict(sample_id=row['sample_id'], label_id=row['label_id'],
                    mean_log_scores=scores, probabilities=p.tolist(),
                    prediction=int(p.argmax()), prompt_contract_hash=store.digest(self.contract(row)),
                    image_tokens=2)

    def candidate_batch(self, rows, candidates, controller=None):
        self.batch_calls += 1
        for r,c in zip(rows,candidates):
            if c not in r['candidate_labels']:
                raise ValueError('unknown candidate')
        scores = torch.tensor([r['lookup'][c] for r,c in zip(rows,candidates)], dtype=torch.float64)
        contracts = [self.contract(r) for r in rows]
        combined = {'batch':len(rows), 'family':'test', 'image_tokens':[2]*len(rows), 'rows':contracts}
        return scores, None, combined, [{'contract':c} for c in contracts]


def row(i, candidates, correct):
    return dict(sample_id=str(i), group_id=str(i), candidate_labels=candidates,
                label_id=candidates.index(correct), label=correct,
                lookup={c:-.1 if c==correct else -4. for c in candidates})


def compare_score_modes(monkeypatch, rows):
    backend = ScoreBackend()
    monkeypatch.setenv('APERTURE_BATCH','1')
    single = engine.score(backend,rows,None)
    monkeypatch.setenv('APERTURE_BATCH','16')
    batch = engine.score(backend,rows,None)
    return single,batch,backend


def test_reordered_candidates_preserve_per_row_label_semantics(monkeypatch):
    rs=[row(0,['left','right'],'left'),row(1,['right','left'],'left')]
    single,batch,_=compare_score_modes(monkeypatch,rs)
    assert [r['mean_log_scores'] for r in batch] == [r['mean_log_scores'] for r in single]
    assert [r['prediction'] for r in batch] == [r['label_id'] for r in rs]


def test_variable_candidate_sets_and_lengths_are_not_borrowed_from_first_row(monkeypatch):
    rs=[row(0,['left','right'],'left'),row(1,['up','down','middle'],'middle')]
    single,batch,_=compare_score_modes(monkeypatch,rs)
    assert [r['mean_log_scores'] for r in batch] == [r['mean_log_scores'] for r in single]


def test_fixed_candidates_keep_scores_and_per_row_contract_hash(monkeypatch):
    rs=[row(i,['normal','tumor'],'tumor') for i in range(3)]
    single,batch,backend=compare_score_modes(monkeypatch,rs)
    assert [r['mean_log_scores'] for r in batch] == [r['mean_log_scores'] for r in single]
    assert [r['prompt_contract_hash'] for r in batch] == [r['prompt_contract_hash'] for r in single]
    assert backend.batch_calls > 0


def batch_rows_method():
    """Execute the actual method, isolated only from unused model imports."""
    path=ROOT/'src/aperture_comprehensive/backend.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Backend')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='batch_rows')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['batch_rows']


def batch_fixture(lengths=(4,6), family='gemma3', patches=(2,2)):
    entries=[]
    for i,(length,count) in enumerate(zip(lengths,patches)):
        ids=torch.arange(length).unsqueeze(0)
        full={'input_ids':ids, 'attention_mask':torch.ones_like(ids)}
        if family=='gemma3':
            full['token_type_ids']=torch.tensor([[0,1]+[0]*(length-2)])
            full['pixel_values']=torch.ones(1,3,2,2)*(i+1)
        else:
            full['image_grid_thw']=torch.tensor([[1,1,count]])
            full['pixel_values']=torch.arange(count*3).reshape(count,3).float()+100*i
        mask=torch.zeros_like(ids,dtype=torch.bool);mask[:,1]=True
        entries.append((full,(length-1,length),mask,{'row':i}))
    backend=SimpleNamespace(job={'family':family},
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    backend.batch=lambda r,c:deepcopy(entries[int(r['sample_id'])])
    rs=[{'sample_id':str(i),'label':'yes'} for i in range(len(entries))]
    return backend,rs,entries


def test_gemma_token_types_pad_together_with_ids_and_attention():
    backend,rs,entries=batch_fixture()
    batch,meta=batch_rows_method()(backend,rs)
    assert batch['input_ids'].shape == batch['attention_mask'].shape == batch['token_type_ids'].shape == (2,6)
    assert batch['token_type_ids'][0].tolist()==[0,1,0,0,0,0]
    assert batch['attention_mask'][0].tolist()==[1,1,1,1,0,0]
    assert [m['span'] for m in meta] == [(3,4),(5,6)]
    assert torch.equal(batch['input_ids'][0,:4],entries[0][0]['input_ids'][0])


def test_qwen_patches_remain_flattened_and_allow_variable_image_token_counts():
    backend,rs,entries=batch_fixture(family='qwen2_5_vl',patches=(2,3))
    batch,_=batch_rows_method()(backend,rs)
    assert batch['pixel_values'].shape == (5,3)
    assert torch.equal(batch['pixel_values'],torch.cat([e[0]['pixel_values'] for e in entries],0))
    assert batch['image_grid_thw'].shape == (2,3)


def test_single_row_batch_keeps_native_qwen_patch_layout():
    backend,rs,entries=batch_fixture(lengths=(4,),family='qwen2_5_vl',patches=(2,))
    batch,_=batch_rows_method()(backend,rs)
    assert torch.equal(batch['pixel_values'],entries[0][0]['pixel_values'])


def test_candidate_count_mismatch_is_rejected_before_processing():
    backend,rs,_=batch_fixture()
    calls=[]
    original=backend.batch
    backend.batch=lambda *args:(calls.append(args),original(*args))[1]
    with pytest.raises(ValueError,match='candidate'):
        batch_rows_method()(backend,rs,['yes'])
    assert calls==[]


def test_empty_batch_is_rejected():
    backend,_,_=batch_fixture()
    with pytest.raises(ValueError,match='empty'):
        batch_rows_method()(backend,[])


@pytest.mark.parametrize('value',['0','-1','garbage','1.5'])
def test_invalid_batch_setting_does_not_silently_become_one(monkeypatch,value):
    monkeypatch.setenv('APERTURE_BATCH',value)
    with pytest.raises(ValueError,match='APERTURE_BATCH'):
        engine.batch_size()


def test_batch_setting_is_part_of_frozen_plan(tmp_path,monkeypatch):
    repo=tmp_path/'repo';(repo/'configs').mkdir(parents=True)
    base={'prepared_root':'prepared','run_root':'runs/original',
          'models':[{'key':m,'path':f'models/{m}'} for m in plan.MODELS],
          'datasets':[{'key':d,'kind':'paired' if d.startswith('bright') else 'single',
                       'classes':3 if d.startswith('bright') else 2,'query_count':300}
                      for d in plan.PANEL]}
    store.atomic(repo/'configs/base.json',base)
    cfg={'base_config':'configs/base.json','run_root':'runs/debug','prepared_root':'prepared'}
    monkeypatch.setenv('APERTURE_BATCH','1')
    first=plan.freeze('E05',cfg,repo)
    monkeypatch.setenv('APERTURE_BATCH','16')
    with pytest.raises(ValueError,match='Frozen'):
        plan.freeze('E05',cfg,repo)
    assert first['units']


@pytest.mark.parametrize('action',['fit','evaluate'])
def test_fit_and_evaluate_reject_runtime_batch_drift_before_gpu(monkeypatch,action):
    monkeypatch.setenv('APERTURE_BATCH','1')
    monkeypatch.setattr(engine,'setup',lambda:pytest.fail('Mismatch reached GPU setup'))
    args=({'execution_batch_size':16},{},{})
    if action=='evaluate':args=(*args,ROOT)
    with pytest.raises(ValueError,match='APERTURE_BATCH'):
        getattr(engine,action)(*args)


def test_old_unbound_plan_is_not_silently_migrated():
    with pytest.raises(ValueError,match='NEW plan'):
        engine.require_execution_batch({})


def test_inconsistent_processor_fields_are_not_silently_dropped():
    backend,rs,_=batch_fixture()
    original=backend.batch
    def changed(row,candidate):
        full,span,mask,contract=original(row,candidate)
        if row['sample_id']=='1':full.pop('token_type_ids')
        return full,span,mask,contract
    backend.batch=changed
    with pytest.raises(ValueError,match='processor fields'):
        batch_rows_method()(backend,rs)


def test_candidate_specific_prompt_contract_is_rejected(monkeypatch):
    monkeypatch.setenv('APERTURE_BATCH','2')
    backend=ScoreBackend();original=backend.candidate_batch
    def changed(rs,cs,controller=None):
        score,mask,contract,meta=original(rs,cs,controller)
        contract['rows'][0]['leaked_answer']=cs[0]
        return score,mask,contract,meta
    backend.candidate_batch=changed
    with pytest.raises(ValueError,match='candidate-specific'):
        engine.score(backend,[row(i,['normal','tumor'],'normal') for i in range(2)],None)
