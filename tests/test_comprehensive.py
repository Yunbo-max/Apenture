import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def test_registry_builds_full_cartesian_product(tmp_path):
    from aperture_comprehensive.protocol import MODELS, ARMS, make_jobs, PROTOCOL
    config = {'protocol': PROTOCOL, 'run_root': str(tmp_path/'run'),
              'prepared_root': str(tmp_path/'prepared'), 'support_seeds': [0, 1, 2],
              'models': [{'key': k, 'path': str(tmp_path/k)} for k in MODELS],
              'datasets': [{'key': k, 'kind': kind, 'classes': n, 'query_count': 12,
                            'source_support': 'support-{seed}.jsonl', 'source_query': 'query.jsonl'}
                           for k, kind, n in [('bright', 'paired', 3), ('path', 'single', 9), ('q2', 'single', 4)]]}
    jobs = make_jobs(config)
    assert len(jobs) == 6*3*3*5
    assert len({j['job_id'] for j in jobs}) == len(jobs)
    assert {j['arm'] for j in jobs} == {'frozen','lora1','lora4','random','aperture'}
    assert all(j['lora_rank']==16 for j in jobs)
    assert {j['lora_passes'] for j in jobs if j['arm']=='lora4'} == {4}
    assert all(j['mask']=='post' for j in jobs if j['kind']=='paired')


def test_phi_fused_sites_preserve_query_and_unmasked_rows():
    from aperture_comprehensive.backend import phi_sites
    from vigor_handoff.core import ResidualController
    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.qkv_proj=nn.Linear(4,12,bias=False)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.layers=nn.ModuleList([nn.Module()]); self.layers[0].self_attn=Attention()
    model=Model(); sites=phi_sites(model,[0])
    x=torch.randn(1,3,4); base=model.layers[0].self_attn.qkv_proj(x)
    bases={(l,k):torch.eye(4)[:,:2] for l,k,_ in sites}
    controller=ResidualController(sites,bases)
    try:
        controller.set_mask(torch.tensor([[False,True,False]]))
        with torch.no_grad():
            for p in controller.parameters(): p.fill_(.2)
        changed=model.layers[0].self_attn.qkv_proj(x)
        assert torch.equal(changed[...,:4],base[...,:4])
        assert torch.equal(changed[:,[0,2]],base[:,[0,2]])
        assert not torch.equal(changed[:,1,4:],base[:,1,4:])
        changed.sum().backward()
        assert all(p.grad is not None and p.grad.norm()>0 for p in controller.parameters())
        assert [id(m) for _,_,m in phi_sites(model,[0])] == [id(m) for _,_,m in sites]
    finally: controller.close()


def test_paired_masks_validate_both_images_and_select_post():
    from aperture_comprehensive.backend import checked_mask
    ids=torch.tensor([[1,9,9,2,9,9,3,4]])
    batch={'input_ids':ids, 'image_grid_thw':torch.tensor([[1,2,4],[1,2,4]])}
    config=SimpleNamespace(image_token_id=9,vision_config=SimpleNamespace(spatial_merge_size=2))
    mask=checked_mask(batch,config,'qwen3_vl',6,2,'post')
    assert mask.tolist()==[[False,False,False,False,True,True,False,False]]
    batch['image_grid_thw']=torch.tensor([[1,2,4]])
    with pytest.raises(ValueError):checked_mask(batch,config,'qwen3_vl',6,2,'post')


def test_q2_parser_uses_question_options_and_explicit_answer(tmp_path):
    from aperture_comprehensive.data import q2_row
    p=tmp_path/'question_1';p.mkdir()
    (p/'pick_point_question.txt').write_text('Choose (pick one from p_17,p_4,p_9,p_1).')
    (p/'pick_point_answer.txt').write_text('Picking Point: p_9')
    (p/'image.png').write_bytes(b'fixture')
    row=q2_row(p,'bridge','pick_point')
    assert row['candidate_labels']==['Picking Point: p_17','Picking Point: p_4','Picking Point: p_9','Picking Point: p_1']
    assert row['label_id']==2 and row['label']==row['candidate_labels'][2]
    (p/'pick_point_answer.txt').write_text('Picking Point: p_100')
    with pytest.raises(ValueError):q2_row(p,'bridge','pick_point')


def test_preflight_rejects_incomplete_phi_shards(tmp_path):
    from aperture_comprehensive.worker import model_files
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'phi3_v','num_hidden_layers':32}))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'part1.safetensors','b':'part2.safetensors'}}))
    (tmp_path/'part1.safetensors').write_bytes(b'one')
    with pytest.raises(FileNotFoundError,match='shards'):
        model_files({'model_key':'phi','model_path':str(tmp_path),'layers':[14,27]})


def test_plan_rejects_hidden_family_exclusions(tmp_path):
    from aperture_comprehensive.protocol import MODELS, make_jobs, PROTOCOL
    config={'protocol':PROTOCOL,'run_root':str(tmp_path/'r'),'prepared_root':str(tmp_path/'p'),
            'support_seeds':[0],'models':[{'key':k,'path':k} for k in MODELS],
            'datasets':[{'key':'x','kind':'single','classes':2,'families':['qwen3_vl']}]}
    with pytest.raises(ValueError,match='exclusions'):make_jobs(config)


def test_role_validation_rejects_cross_role_pixels_and_groups():
    from aperture_comprehensive.protocol import validate_roles
    def row(i):
        return {'sample_id':str(i),'group_id':str(i),'content_hash':str(i),
                'candidate_labels':['a','b'],'label':'a','label_id':0}
    roles={'support':[row(1)],'calibration':[row(2)],'query':[row(3)]}
    validate_roles(roles,2)
    roles['query'][0]['content_hash']='1'
    with pytest.raises(ValueError,match='content leakage'):validate_roles(roles,2)
    roles['query'][0]['content_hash']='3';roles['query'][0]['group_id']='1'
    with pytest.raises(ValueError,match='group leakage'):validate_roles(roles,2)


def test_summary_never_exports_partial_matrix(tmp_path):
    from aperture_comprehensive.reporting import export
    from aperture_comprehensive.protocol import MODELS, PROTOCOL
    config={'protocol':PROTOCOL,'run_root':str(tmp_path/'r'),'prepared_root':str(tmp_path/'p'),
            'support_seeds':[0],'models':[{'key':k,'path':k} for k in MODELS],
            'datasets':[{'key':'x','kind':'single','classes':2}]}
    with pytest.raises(ValueError,match='incomplete'):export(config)
    assert not (tmp_path/'r'/'exports'/'results.csv').exists()


def test_selection_refuses_comparison_without_random(tmp_path):
    from aperture_comprehensive.worker import seal_unit
    jobs=[{'arm':a,'unit_id':'u','output':str(tmp_path/a)} for a in ['frozen','lora1','lora4','aperture']]
    with pytest.raises(ValueError,match='five-arm'):seal_unit(jobs,tmp_path/'selection.json')


def test_pool_split_allocates_query_quota_and_reserves_support():
    from aperture_comprehensive.data import pool_split
    rows=[{'sample_id':f'{c}-{i}','group_id':f'{c}-{i}','label_id':c} for c in range(4) for i in range(120)]
    support,query=pool_split(rows,4,400,0,'q2')
    other_support,other_query=pool_split(rows,4,400,1,'q2')
    assert len(support)==32 and len(query)==400
    assert {r['group_id'] for r in support}.isdisjoint(r['group_id'] for r in query)
    assert query==other_query


def test_q2_duplicate_options_are_one_semantic_answer(tmp_path):
    from aperture_comprehensive.data import q2_row
    p=tmp_path/'question_2';p.mkdir()
    (p/'ending_tile_question.txt').write_text('Choose (pick one from a_0,b_1,a_0,c_2).')
    (p/'ending_tile_answer.txt').write_text('Ending tile: c_2')
    row=q2_row(p,'bridge','ending_tile')
    assert row['candidate_labels']==['Ending tile: a_0','Ending tile: b_1','Ending tile: c_2']
    assert row['label_id']==2 and row['metadata']['original_options']==['a_0','b_1','a_0','c_2']


def test_variable_candidates_do_not_receive_padding_probability():
    from aperture_comprehensive.numeric import score_arrays
    from aperture_final.numeric import probabilities, metrics
    scores,labels=score_arrays([{'mean_log_scores':[-2.,-1.],'label_id':1},
                              {'mean_log_scores':[-4.,-3.,-2.,-1.],'label_id':3}],4)
    assert scores.shape==(2,4)
    assert probabilities(scores,20)[0,2:].tolist()==[0.,0.]
    assert metrics(scores,labels)['accuracy']==1.


def test_audits_use_canonical_seed_even_for_filtered_execution():
    from aperture_comprehensive.cli import canonical_audits
    jobs=[dict(model_key='phi',domain_key='h2',support_seed=s,arm='aperture') for s in [0,1,2]]
    assert canonical_audits(jobs,[jobs[2]])==[jobs[0]]


def test_robot_frames_can_have_distinct_questions_within_support():
    from aperture_comprehensive.protocol import validate_roles
    def row(i,pixels,question):
        return dict(sample_id=str(i),group_id='episode' if i<3 else 'query-episode',content_hash=pixels,
                    question=question,candidate_labels=['success','failure'],label='success',label_id=0)
    roles={'support':[row(0,'frame','Did it pick up?'),row(1,'frame','Did it place?')],
           'calibration':[row(2,'frame','Did it return?')],
           'query':[row(3,'another-frame','Did it pick up?')]}
    validate_roles(roles,2,fit_cal_groups=False)
    roles['query'][0]['content_hash']='frame'
    with pytest.raises(ValueError,match='content leakage'):validate_roles(roles,2)


def test_guardian_support_seeds_share_query_and_exclude_query_groups():
    from aperture_comprehensive.data import shared_query_support
    pool=[dict(sample_id=f'{c}-{i}',group_id=f'episode-{i}',label_id=c) for c in range(2) for i in range(20)]
    query=[r for r in pool if int(r['group_id'].split('-')[-1])>=15]
    a=shared_query_support(pool,query,2,0,'guardian')
    b=shared_query_support(pool,query,2,1,'guardian')
    assert len(a)==len(b)==16 and a!=b
    assert {r['group_id'] for r in a+b}.isdisjoint(r['group_id'] for r in query)
