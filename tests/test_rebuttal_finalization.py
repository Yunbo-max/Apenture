"""Source-independent CPU tests for read-only rebuttal finalization."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'aperture_rebuttal_finish.py'
spec=importlib.util.spec_from_file_location('aperture_rebuttal_finish',SCRIPT)
if SCRIPT.exists():
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    mod=None


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.support=self.root/'fit.jsonl';self.support.write_text('{"sample_id":"s0"}\n')
        self.cal=self.root/'cal.jsonl';self.cal.write_text('{"sample_id":"c0"}\n')
        self.query=self.root/'query.jsonl';self.query.write_text('{"sample_id":"q0"}\n')
        self.unit='q25--camelyon17-hospital_2--n16--s0'

    def plan(self,case,names):
        u=dict(key=self.unit,model_key='q25',domain='camelyon17-hospital_2',support_seed=0,total_labels=16,
               support=str(self.support),calibration=str(self.cal),query=str(self.query),
               model_id='Qwen/Qwen2.5-VL-7B-Instruct',image_size=448,layers=[14,27],mask='all_visual',
               variants=[dict(name=n,engine='conditional_gated' if n.startswith('conditional') else 'aperture',
                              alpha=3.,lr=.05,steps=4) for n in names], output=str(self.root/case/'units'/self.unit))
        write(self.root/case/'plan.json',{'case':case,'units':[u],'settings':{'run_root':str(self.root)}})
        return u

    def fit(self,u,name,nll):
        p=Path(u['output'])/'fits'/name
        write(p/'fit.json',dict(name=name,calibration_nll=nll,final_support_metrics={'nll':0.2},
              final_calibration_metrics={'nll':nll},optimizer_updates=4))
        # seal's content and digest are checked by store.verify when source exists
        import hashlib
        h=hashlib.sha256((p/'fit.json').read_bytes()).hexdigest()
        write(p/'DONE.json',{'binding':{'fixture':'yes'},'files':{'fit.json':h}})

    def test_selects_lowest_calibration_nll_not_query(self):
        u=self.plan('CG05',['conditional-gated-a1-lr0.01','conditional-gated-a3-lr0.05'])
        self.fit(u,'conditional-gated-a1-lr0.01',0.7)
        self.fit(u,'conditional-gated-a3-lr0.05',0.4)
        # Query result deliberately favors losing candidate; must not matter.
        write(Path(u['output'])/'evaluation'/'conditional-gated-a1-lr0.01'/'result.json',{'accuracy':1.0})
        r=mod.select_candidates(self.root,['CG05'])
        self.assertEqual(r['units'][0]['families']['conditional-gated']['selected']['variant'],
                         'conditional-gated-a3-lr0.05')
        self.assertNotIn('accuracy',str(r))

    def test_incomplete_grid_fails_closed(self):
        u=self.plan('CG05',['conditional-gated-a1-lr0.01','conditional-gated-a3-lr0.05'])
        self.fit(u,'conditional-gated-a3-lr0.05',0.1)
        r=mod.select_candidates(self.root,['CG05'])
        f=r['units'][0]['families']['conditional-gated']
        self.assertEqual(f['status'],'incomplete')
        self.assertIsNone(f['selected'])

    def test_rejects_cross_case_manifest_mismatch(self):
        a=self.plan('CG05',['conditional-gated-a1-lr0.01'])
        b=self.plan('CG08',['conditional-gated-p4'])
        bad=self.root/'bad_cal.jsonl';bad.write_text('changed\n')
        data=json.loads((self.root/'CG08'/'plan.json').read_text())
        data['units'][0]['calibration']=str(bad)
        write(self.root/'CG08'/'plan.json',data)
        self.fit(a,'conditional-gated-a1-lr0.01',0.6)
        self.fit(b,'conditional-gated-p4',0.5)
        with self.assertRaisesRegex(ValueError,'manifests|signature|mismatch'):
            mod.select_candidates(self.root,['CG05','CG08'])

    def test_refuses_unsupported_other_family(self):
        self.plan('CG05',['conditional-gated-a1-lr0.01'])
        with self.assertRaisesRegex(ValueError,'allowed'):
            mod.select_candidates(self.root,['CG01'])

    def test_check_e10_never_claims_unverified_done(self):
        path=self.root/'E10';write(path/'visual'/'plan.json',{
            'settings':{'run_root':str(path)},'part':'visual','jobs':[{'key':'u0'},{'key':'u1'}]})
        write(path/'visual'/'units'/'u0'/'DONE.json',{'files':{'result.json':'not-real'}})
        r=mod.e10_census(path)
        self.assertEqual(r['visual']['valid'],0)
        self.assertEqual(r['visual']['invalid'],['u0'])
        self.assertEqual(r['visual']['missing'],['u1'])

    def test_q34_pred_diag_requires_sealed_pairs(self):
        u=self.plan('CG01',['frozen','conditional-gated'])
        self.assertEqual(mod.q34_diagnose(self.root)['status'],'no_verified_pairs')


if __name__=='__main__':unittest.main()

class SelectedAggregationTests(FinalizationTests):
    def test_selected_report_requires_all_seeds_and_uses_selected_variant(self):
        u=self.plan('CG02',['conditional-gated-a3-lr0.05','lora-r4-lr0.0001-p4'])
        # The family is explicit in actual frozen plans.
        plan_path=self.root/'CG02'/'plan.json'
        plan=read_json(plan_path)
        for v in plan['units'][0]['variants']:
            v['family']='conditional-gated' if v['name'].startswith('conditional') else 'lora-r4'
        # A second planned seed is intentionally missing: never aggregate seed 0 alone.
        from copy import deepcopy
        other=deepcopy(plan['units'][0]);other['key']=u['key'][:-2]+'s1';other['support_seed']=1
        other['output']=str(self.root/'CG02'/'units'/other['key']);plan['units'].append(other)
        write(plan_path,plan)
        self.fit(u,'conditional-gated-a3-lr0.05',0.3)
        self.fit(u,'lora-r4-lr0.0001-p4',0.2)
        unit_path=Path(u['output'])
        write(unit_path/'selection.json',{'selected':{'conditional-gated':'conditional-gated-a3-lr0.05',
                                                     'lora-r4':'lora-r4-lr0.0001-p4'},
                                         'evaluate':['conditional-gated-a3-lr0.05','lora-r4-lr0.0001-p4']})
        seal_file(unit_path/'selection_receipt', 'selection.json', read_json(unit_path/'selection.json'))
        for name,acc in [('conditional-gated-a3-lr0.05',.61),('lora-r4-lr0.0001-p4',.70)]:
            seal_file(unit_path/'evaluation'/name,'result.json',{'raw':{'accuracy':acc,'macro_f1':acc-.1,'nll':1.0,'n':500}})
        r=mod.audit_selected(self.root,['CG02'])
        self.assertEqual(r['status'],'partial')
        self.assertEqual(len(r['unit_rows']),2)
        self.assertEqual(r['summary'],[])
        self.assertEqual({z['family']:z['accuracy'] for z in r['unit_rows']},
                         {'conditional-gated':.61,'lora-r4':.70})

    def test_selected_report_rejects_wrong_calibration_choice(self):
        u=self.plan('CG02',['conditional-gated-a1-lr0.01','conditional-gated-a3-lr0.05'])
        pp=self.root/'CG02'/'plan.json';plan=read_json(pp)
        for v in plan['units'][0]['variants']:v['family']='conditional-gated'
        write(pp,plan)
        self.fit(u,'conditional-gated-a1-lr0.01',0.3)
        self.fit(u,'conditional-gated-a3-lr0.05',0.8)
        folder=Path(u['output'])
        choice={'selected':{'conditional-gated':'conditional-gated-a3-lr0.05'},
                'evaluate':['conditional-gated-a3-lr0.05']}
        write(folder/'selection.json',choice)
        seal_file(folder/'selection_receipt','selection.json',choice)
        seal_file(folder/'evaluation'/'conditional-gated-a3-lr0.05','result.json',
                  {'raw':{'accuracy':.9,'macro_f1':.9,'nll':.1,'n':500}})
        report=mod.audit_selected(self.root,['CG02'])
        self.assertEqual(report['status'],'partial')
        self.assertTrue(any('calibration' in issue['reason'] for issue in report['issues']))
        self.assertEqual(report['unit_rows'],[])


def read_json(path):return json.loads(Path(path).read_text())

def seal_file(folder,name,obj):
    import hashlib
    path=folder/name
    write(path,obj)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    write(folder/'DONE.json',{'binding':{'fixture':'yes'},'files':{name:digest}})

class E10ProvenanceTests(FinalizationTests):
    def test_e10_mismatched_plan_binding_fails_closed(self):
        root=self.root/'retention'
        plan={'part':'visual','settings':{'run_root':str(root)},'jobs':[{'key':'u0'}]}
        write(root/'visual'/'plan.json',plan)
        folder=root/'visual'/'units'/'u0'
        write(folder/'result.json',{'all_identity_checks_passed':True})
        import hashlib
        h=hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()
        write(folder/'DONE.json',{'binding':{'plan':'WRONG_PLAN','job':plan['jobs'][0]},
                                   'files':{'result.json':h}})
        r=mod.e10_census(root)
        self.assertEqual(r['visual']['valid'],0)
        self.assertEqual(r['visual']['invalid'],['u0'])