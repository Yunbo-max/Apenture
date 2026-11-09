#!/usr/bin/env python3
"""Generate one E01 per-model job: root config, plan freeze, per-domain checks, payload.
Usage: qz_gen_one.py <model> <half a|b>
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / '.venv_rebuttal/bin/python')
BASE = json.loads((REPO / 'configs/aperture_reviewer_cases_v1.json').read_text())
DOMAINS = ['bright-hawaii-wildfire', 'bright-libya-flood', 'bright-noto-earthquake',
           'bright-turkey-earthquake', 'camelyon17-hospital_0', 'camelyon17-hospital_1',
           'camelyon17-hospital_2', 'pathmnist', 'manipbench-q1-bridge_pick_place',
           'manipbench-q1-droid_arti', 'camelyon17-hospital_3', 'camelyon17-hospital_4',
           'manipbench-q1-droid_pick_place', 'manipbench-q2-bridge_pick_place-pick_point',
           'manipbench-q2-bridge_pick_place-ending_tile', 'manipbench-q2-droid_arti-pick_point',
           'manipbench-q2-droid_arti-ending_tile', 'manipbench-q2-droid_pick_place-pick_point',
           'manipbench-q2-droid_pick_place-ending_tile', 'guardian-ur5fail_test']
WS = 'ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6'
PROJECT = 'project-160ccb20-98ab-4538-a847-01d1f83d5b0f'
LCG = 'lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e'
SPEC = '79fe954a-be92-4772-ac0b-94ad8a79b7bb'
IMAGE = 'docker.sii.shaipower.online/inspire-studio/relay2:v2'
OUT = REPO / 'scripts/qz_payloads'


def main():
    model, half = sys.argv[1], sys.argv[2]
    domains = DOMAINS[:10] if half == 'a' else DOMAINS[10:]
    job = f'e01v2_{model}_{half}'
    cfg = dict(BASE)
    cfg['run_root'] = f'runs/aperture_rebuttal_cases_20261108_b16_{job}'
    cprel = f'configs/aperture_reviewer_cases_v1_{job}.json'
    (REPO / cprel).write_text(json.dumps(cfg, indent=2) + '\n')
    r = subprocess.run([PY, 'scripts/run_aperture_cases.py', 'plan', '--case', 'E01',
                        '--config', cprel], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'plan failed: {r.stderr[-300:]}')
    for d in domains:
        r = subprocess.run([PY, 'scripts/run_aperture_cases.py', 'check', '--case', 'E01',
                            '--domain', d, '--config', cprel], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f'check {d} failed: {r.stderr[-300:]}')
        print(f'{job} {d} checked', flush=True)
    lines = '\n'.join(f'{PY} scripts/run_aperture_cases.py run --case E01 --model {model} '
                      f'--domain {d} --gpu 0 --execute --config {cprel} 2>&1 | tee runs/qz_jobs/{job}_{i}.log'
                      for i, d in enumerate(domains))
    p = {'workspace_id': WS, 'project_id': PROJECT, 'logic_compute_group_id': LCG,
         'name': f'aperture-{job}-20261109', 'description': f'E01 {model} {half} APERTURE_BATCH=16',
         'framework': 'pytorch',
         'command': 'set -euo pipefail\ncd /inspire/hdd/global_user/zhangjiaquan-253108540222/backup/APERTURE\n'
                    'export APERTURE_BATCH=16\n' + lines + '\necho JOB_DONE',
         'max_running_time_ms': str(36 * 3600000), 'task_priority': 4,
         'framework_config': [{'image': IMAGE, 'image_type': 'SOURCE_PRIVATE',
                               'instance_count': 1, 'spec_id': SPEC, 'shm_gi': 100}]}
    (OUT / f'{job}.json').write_text(json.dumps(p, indent=2) + '\n')
    print(job, 'READY', flush=True)


if __name__ == '__main__':
    main()
