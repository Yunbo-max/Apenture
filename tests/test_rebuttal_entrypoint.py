"""Offline orchestration tests; no GPU or benchmark result is simulated."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_aperture_rebuttal.py'
spec = importlib.util.spec_from_file_location('aperture_rebuttal_entry', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    (root / 'configs').mkdir(parents=True)
    (root / 'src/aperture').mkdir(parents=True)
    (root / 'scripts').mkdir()
    (root / 'src/aperture/cli.py').write_text('# test source\n')
    (root / 'scripts/run_aperture_rebuttal.py').write_text(SCRIPT.read_text())
    medical = {'protocol': 'aperture_medical_multimodel_v1',
               'run_root': 'runs/old-medical', 'prepared_root': 'data/prepared/old-medical',
               'data_root': 'data/raw/camelyon17', 'hospitals': [0,1,2,3,4],
               'support_seeds': [10,11,12], 'query_count': 500,
               'primary_support': 32, 'models': [{'path': '/models/qwen'}]}
    efficiency = {'protocol': 'aperture_final_v1', 'run_root': 'runs/old-final',
                  'prepared_root': 'data/prepared/old-final', 'support_seeds': [0,1,2],
                  'model_path': '/models/qwen', 'timing_samples': 12,
                  'domains': [{'name': n, 'support': f'data/{n}/seed_{{seed}}/support.jsonl',
                               'query_count': 300} for n in
                              ['hawaii-wildfire','libya-flood','hospital_2']]}
    for name, data in [('aperture_medical_multimodel.json', medical),
                       ('aperture_final_round.json', efficiency)]:
        (root / 'configs' / name).write_text(json.dumps(data))
    return root


def test_plan_preserves_medical_science_and_isolates_outputs(repo):
    before = json.loads((repo / 'configs/aperture_medical_multimodel.json').read_text())
    plan = runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], '0')
    cfg = json.loads(Path(plan['suites'][0]['config']).read_text())
    assert {k:v for k,v in cfg.items() if k not in ('run_root','prepared_root')} == {
        k:v for k,v in before.items() if k not in ('run_root','prepared_root')}
    assert cfg['run_root'] == str(repo / 'runs/new/medical-multihospital/run')
    assert cfg['prepared_root'] == str(repo / 'runs/new/medical-multihospital/prepared')
    assert not (repo / 'runs/old-medical').exists()


def test_efficiency_is_exact_existing_two_domain_seed_zero_subset(repo):
    plan = runner.freeze_plan(repo, repo / 'runs/new', ['efficiency-rank16'], '0')
    cfg = json.loads(Path(plan['suites'][0]['config']).read_text())
    assert [d['name'] for d in cfg['domains']] == ['hawaii-wildfire','hospital_2']
    assert cfg['support_seeds'] == [0]
    assert cfg['timing_samples'] == 12
    assert all(d['query_count'] == 300 for d in cfg['domains'])


def test_same_plan_is_idempotent_but_source_changes_are_rejected(repo):
    path = repo / 'runs/new'
    a = runner.freeze_plan(repo, path, list(runner.SUITES), '0')
    assert runner.freeze_plan(repo, path, list(runner.SUITES), '0') == a
    (repo / 'src/aperture/cli.py').write_text('# changed source\n')
    with pytest.raises(ValueError, match='changed'):
        runner.freeze_plan(repo, path, list(runner.SUITES), '0')


def test_config_tampering_is_rejected(repo):
    plan = runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], '0')
    Path(plan['suites'][0]['config']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], '0')


def test_existing_unbound_root_is_not_overwritten(repo):
    path = repo / 'runs/new'
    path.mkdir(parents=True)
    (path / 'old-results.json').write_text('keep')
    with pytest.raises(ValueError, match='nonempty'):
        runner.freeze_plan(repo, path, ['medical-multihospital'], '0')
    assert (path / 'old-results.json').read_text() == 'keep'


@pytest.mark.parametrize('gpus', ['', '0,0', '0;echo bad', '-1', '0,,1'])
def test_invalid_gpu_selection_is_rejected(repo, gpus):
    with pytest.raises(ValueError, match='GPU'):
        runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], gpus)


def test_dry_run_does_not_call_native_suite(repo, monkeypatch):
    plan = runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], '0,1')
    def forbidden(*args, **kwargs):
        pytest.fail('A dry run launched a subprocess')
    monkeypatch.setattr(runner.subprocess, 'run', forbidden)
    result = runner.execute_plan(plan, action='run', execute=False)
    assert result['status'] == 'dry_run'
    assert [c['action'] for c in result['commands']] == ['prepare','plan','check','run','summarize']
    assert result['commands'][3]['argv'][-2:] == ['--gpus','0,1']


def test_failure_stops_before_fit_and_records_failed_stage(repo, monkeypatch):
    plan = runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], '0')
    called = []
    def process(argv, **kwargs):
        action = argv[3]
        called.append(action)
        assert kwargs['cwd'] == str(repo)
        return subprocess.CompletedProcess(argv, 7 if action == 'check' else 0)
    monkeypatch.setattr(runner.subprocess, 'run', process)
    result = runner.execute_plan(plan, action='run', execute=True)
    assert called == ['prepare','plan','check']
    assert result['status'] == 'failed' and result['returncode'] == 7
    receipts = list((repo / 'runs/new/attempts').glob('*.json'))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())['stages'][-1]['action'] == 'check'


def test_summary_action_never_prepares_or_runs_training(repo, monkeypatch):
    plan = runner.freeze_plan(repo, repo / 'runs/new', list(runner.SUITES), '0')
    called = []
    def process(argv, **kwargs):
        called.append(argv[3])
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(runner.subprocess, 'run', process)
    result = runner.execute_plan(plan, action='summarize', execute=True)
    assert called == ['summarize', 'summarize']
    assert result['status'] == 'commands_completed'
    assert result['benchmark_results_validated'] is False


def test_root_must_not_overlap_historical_output_or_source(repo):
    for path in (repo, repo / 'src', repo / 'runs/old-medical', repo / 'runs/old-medical/retry', repo / 'runs'):
        with pytest.raises(ValueError, match='overlap'):
            runner.freeze_plan(repo, path, ['medical-multihospital'], '0')


def test_changed_plan_parameters_are_rejected(repo):
    runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], '0')
    with pytest.raises(ValueError, match='changed'):
        runner.freeze_plan(repo, repo / 'runs/new', ['medical-multihospital'], '1')
