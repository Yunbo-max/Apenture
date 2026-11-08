#!/usr/bin/env python3
"""Debug wrapper: run the g34 frozen fit stage directly with full traceback.

Constructs the same task file the orchestrator writes, then calls
aperture_rebuttal_cases.cli.internal() without the CLI's exception swallowing.
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aperture_rebuttal_cases import store
from aperture_rebuttal_cases.cli import internal

REPO = Path(__file__).resolve().parents[1]
CASE = 'E03'
UNIT = 'g34--hospital_0--n16--s0'
VARIANT = 'frozen'
PLAN = REPO / f'runs/aperture_rebuttal_cases_20261108_b16/{CASE}/plan.json'


def main():
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTHONHASHSEED', '0')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    plan = json.loads(PLAN.read_text())
    unit = next(u for u in plan['units'] if u['key'] == UNIT)
    task = {'stage': 'fit', 'unit_key': UNIT, 'variant_name': VARIANT,
            'plan_path': str(PLAN), 'plan_sha256': store.sha(PLAN), 'repo': str(REPO)}
    task_path = Path(unit['output']) / 'logs' / 'fit--frozen.debug.job.json'
    store.atomic(task_path, task, immutable=False)
    try:
        result = internal(str(task_path))
        print('INTERNAL OK:', json.dumps(result, indent=2))
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
