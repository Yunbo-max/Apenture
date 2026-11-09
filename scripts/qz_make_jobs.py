#!/usr/bin/env python3
"""Generate per-card run roots + qz payloads for the 16-card full rebuttal run.

Each concurrent card gets its own run_root config (separate .gpu-0.lock).
Plans are frozen and units preflighted per root. Payloads are written to
scripts/qz_payloads/<job>.json for submission via `qz train CreateJob`.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / '.venv_rebuttal/bin/python')
BASE_CFG = REPO / 'configs/aperture_reviewer_cases_v1.json'
OUT = REPO / 'scripts/qz_payloads'
WS = 'ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6'
PROJECT = 'project-160ccb20-98ab-4538-a847-01d1f83d5b0f'
LCG = 'lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e'
SPEC = '79fe954a-be92-4772-ac0b-94ad8a79b7bb'
IMAGE = 'docker.sii.shaipower.online/inspire-studio/relay2:v2'

DOMAINS = [
    'bright-hawaii-wildfire', 'bright-libya-flood', 'bright-noto-earthquake',
    'bright-turkey-earthquake', 'camelyon17-hospital_0', 'camelyon17-hospital_1',
    'camelyon17-hospital_2', 'pathmnist', 'manipbench-q1-bridge_pick_place',
    'manipbench-q1-droid_arti', 'camelyon17-hospital_3', 'camelyon17-hospital_4',
    'manipbench-q1-droid_pick_place', 'manipbench-q2-bridge_pick_place-pick_point',
    'manipbench-q2-bridge_pick_place-ending_tile', 'manipbench-q2-droid_arti-pick_point',
    'manipbench-q2-droid_arti-ending_tile', 'manipbench-q2-droid_pick_place-pick_point',
    'manipbench-q2-droid_pick_place-ending_tile', 'guardian-ur5fail_test',
]
E01_SPLIT_A = DOMAINS[:10]
E01_SPLIT_B = DOMAINS[10:]
E01_MODELS = ['q25', 'q34', 'g34', 'internvl3', 'phi']


def make_root(job):
    cfg = json.loads(BASE_CFG.read_text())
    cfg['run_root'] = f'runs/aperture_rebuttal_cases_20261108_b16_{job}'
    path = REPO / f'configs/aperture_reviewer_cases_v1_{job}.json'
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    return str(path.relative_to(REPO))


def freeze_checks(cfg_rel, case, model=None, domains=None):
    cfg_flag = ['--config', cfg_rel]
    for c in ('E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09'):
        r = subprocess.run([PY, 'scripts/run_aperture_cases.py', 'plan', '--case', c] + cfg_flag,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'plan {c} failed: {r.stderr[-300:]}')
    if domains:
        for d in domains:
            r = subprocess.run([PY, 'scripts/run_aperture_cases.py', 'check', '--case', case,
                                '--domain', d] + cfg_flag, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f'check {case}/{d} failed: {r.stderr[-300:]}')
    elif model:
        r = subprocess.run([PY, 'scripts/run_aperture_cases.py', 'check', '--case', case,
                            '--model', model] + cfg_flag, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'check {case}/{model} failed: {r.stderr[-300:]}')
    else:
        r = subprocess.run([PY, 'scripts/run_aperture_cases.py', 'check', '--case', case] + cfg_flag,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'check {case} failed: {r.stderr[-300:]}')


def payload(job, command_lines, hours=24):
    body = '\n'.join(f'{PY} {line} 2>&1 | tee runs/qz_jobs/{job}_{i}.log'
                     for i, line in enumerate(command_lines))
    return {
        'workspace_id': WS, 'project_id': PROJECT, 'logic_compute_group_id': LCG,
        'name': f'aperture-{job}-20261108',
        'description': f'rebuttal job {job}; APERTURE_BATCH=16',
        'framework': 'pytorch',
        'command': 'set -euo pipefail\ncd /inspire/hdd/global_user/zhangjiaquan-253108540222/backup/APERTURE\n'
                   'export APERTURE_BATCH=16\n' + body + '\necho JOB_DONE',
        'max_running_time_ms': str(hours * 3600000), 'task_priority': 4,
        'framework_config': [{'image': IMAGE, 'image_type': 'SOURCE_PRIVATE',
                              'instance_count': 1, 'spec_id': SPEC, 'shm_gi': 100}],
    }


def main():
    OUT.mkdir(exist_ok=True)
    jobs = {}

    # E02: 2 cards (q25 / g34)
    for model in ('q25', 'g34'):
        job = f'e02_{model}'
        cfg = make_root(job)
        freeze_checks(cfg, 'E02', model=model)
        jobs[job] = payload(job, [
            f'scripts/run_aperture_cases.py run --case E02 --model {model} --gpu 0 --execute --config {cfg}',
            f'scripts/run_aperture_cases.py status --case E02 --config {cfg}',
        ], hours=24)

    # E07: 2 cards (q25 / g34; generation-heavy)
    for model in ('q25', 'g34'):
        job = f'e07_{model}'
        cfg = make_root(job)
        freeze_checks(cfg, 'E07', model=model)
        jobs[job] = payload(job, [
            f'scripts/run_aperture_cases.py run --case E07 --model {model} --gpu 0 --execute --config {cfg}',
        ], hours=24)

    # E04 + E06 + E09: 1 card (sequential, same root)
    job = 'e04e06e09'
    cfg = make_root(job)
    freeze_checks(cfg, 'E04'); freeze_checks(cfg, 'E06'); freeze_checks(cfg, 'E09')
    jobs[job] = payload(job, [
        'scripts/run_aperture_cases.py run --case E04 --gpu 0 --execute --config ' + cfg,
        'scripts/run_aperture_cases.py run --case E06 --gpu 0 --execute --config ' + cfg,
        'scripts/run_aperture_cases.py audit --case E09 --gpu 0 --execute --config ' + cfg,
    ], hours=24)

    # E05: 1 card (q25 then g34 sequential)
    job = 'e05'
    cfg = make_root(job)
    freeze_checks(cfg, 'E05')
    jobs[job] = payload(job, [
        f'scripts/run_aperture_cases.py run --case E05 --model q25 --gpu 0 --execute --config {cfg}',
        f'scripts/run_aperture_cases.py run --case E05 --model g34 --gpu 0 --execute --config {cfg}',
    ], hours=24)

    # E01: 10 cards (5 models x 2 domain splits)
    for model in E01_MODELS:
        for half, domains in (('a', E01_SPLIT_A), ('b', E01_SPLIT_B)):
            job = f'e01v2_{model}_{half}'
            cfg = make_root(job)
            freeze_checks(cfg, 'E01', domains=domains)
            commands = [f'scripts/run_aperture_cases.py run --case E01 --model {model} --domain {d} --gpu 0 --execute --config {cfg}'
                        for d in domains]
            jobs[job] = payload(job, commands, hours=36)

    for job, p in jobs.items():
        (OUT / f'{job}.json').write_text(json.dumps(p, indent=2) + '\n')
    print('generated', len(jobs), 'jobs:', ', '.join(sorted(jobs)))


if __name__ == '__main__':
    main()
