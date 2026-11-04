#!/usr/bin/env python3
"""APERTURE-only rebuttal orchestration; default plan never launches experiments.

Reuse the existing audited medical and rank-16 suites. Do not import SPECTRUM,
change the learning algorithm, select on query labels, or overwrite old runs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('medical-multihospital', 'efficiency-rank16')
SOURCES = {
    'medical-multihospital': 'configs/aperture_medical_multimodel.json',
    'efficiency-rank16': 'configs/aperture_final_round.json',
}
ACTIONS = ('prepare', 'plan', 'check', 'run', 'summarize')


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def source_identity(repo: Path) -> dict:
    paths = set()
    for folder, suffixes in [('src', {'.py'}), ('scripts', {'.py', '.sh'})]:
        paths.update(p for p in (repo / folder).rglob('*') if p.is_file() and p.suffix in suffixes)
    paths.update(repo / p for p in SOURCES.values())
    return {str(p.relative_to(repo)): digest_file(p) for p in sorted(paths)}


def overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def freeze_plan(repo: Path, root: Path, suites: list[str], gpus: str) -> dict:
    """Create or verify a source/config-bound plan without model/data loading."""
    repo, root = Path(repo).resolve(), Path(root).resolve()
    if (not re.fullmatch(r'\d+(,\d+)*', gpus)
            or len(set(map(int, gpus.split(',')))) != len(gpus.split(','))):
        raise ValueError('Distinct nonnegative GPU indices are required, e.g. 0,1')
    if not suites or len(set(suites)) != len(suites) or any(s not in SUITES for s in suites):
        raise ValueError('Select unique APERTURE suites only')
    bases = {s: json.loads((repo / path).read_text()) for s, path in SOURCES.items()}
    protected = [repo / n for n in ('src', 'scripts', 'configs', 'paper', 'tests', 'results', 'reports', '.git')]
    for cfg in bases.values():
        for key in ('run_root', 'prepared_root'):
            protected.append((repo / cfg[key]).resolve())
    if root == repo or root in repo.parents or any(overlaps(root, p.resolve()) for p in protected):
        raise ValueError('New root must not overlap source files or historical run/prepared roots')
    entries = []
    for suite in suites:
        cfg = deepcopy(bases[suite])
        cfg['run_root'] = str(root / suite / 'run')
        cfg['prepared_root'] = str(root / suite / 'prepared')
        if suite == 'efficiency-rank16':
            names = ('hawaii-wildfire', 'hospital_2')
            domains = {d['name']: d for d in cfg['domains']}
            if any(n not in domains for n in names) or cfg.get('timing_samples', 12) != 12:
                raise ValueError('Efficiency source must contain Hawaii/H2 and twelve timing queries')
            cfg['domains'] = [domains[n] for n in names]
            cfg['support_seeds'] = [0]
        entries.append({'suite': suite, 'config': str(root / 'configs' / f'{suite}.json'),
                        'contents': cfg})
    plan = {'protocol': 'aperture_rebuttal_orchestration_v1', 'repo': str(repo),
            'root': str(root), 'gpus': gpus, 'python': sys.executable,
            'source_sha256': source_identity(repo), 'suites': entries,
            'status': 'planned_not_run', 'new_algorithm': False}
    marker = root / 'plan.json'
    if marker.exists():
        if json.loads(marker.read_text()) != plan:
            raise ValueError('Frozen plan/source changed; restore the pinned checkout or use a new root')
        for entry in entries:
            path = Path(entry['config'])
            if not path.is_file() or path.read_text() != json_text(entry['contents']):
                raise ValueError('Frozen suite configuration changed or is missing')
        return plan
    if root.exists() and any(root.iterdir()):
        raise ValueError('Refusing a nonempty root without its complete plan; use a new root')
    (root / 'configs').mkdir(parents=True, exist_ok=True)
    for entry in entries:
        with Path(entry['config']).open('x') as stream:
            stream.write(json_text(entry['contents']))
    with marker.open('x') as stream:
        stream.write(json_text(plan))
    return plan


def commands_for(plan: dict, action: str) -> list[dict]:
    if action not in ('run', 'summarize'):
        raise ValueError('Unknown execution action')
    stages = ACTIONS if action == 'run' else ('summarize',)
    result = []
    for entry in plan['suites']:
        for stage in stages:
            command = [plan['python'], '-m', 'aperture.cli', stage,
                       '--suite', entry['suite'], '--config', entry['config']]
            if stage == 'run':
                command.extend(['--gpus', plan['gpus']])
            result.append({'suite': entry['suite'], 'action': stage, 'argv': command})
    return result


def execute_plan(plan: dict, *, action: str, execute: bool) -> dict:
    """Run foreground stages, stop on failure, and leave native audit gates intact."""
    commands = commands_for(plan, action)
    if not execute:
        return {'status': 'dry_run', 'commands': commands, 'execute_required': True}
    repo, root = Path(plan['repo']), Path(plan['root'])
    lock = root / '.executor.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError('Run root is locked; confirm the earlier process stopped before removing its lock') from error
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(str(os.getpid()) + '\n')
    started = datetime.now(timezone.utc).isoformat()
    result = {'status': 'commands_running', 'started_at': started, 'stages': [],
              'benchmark_results_validated': False}
    env = {**os.environ, 'PYTHONPATH': str(repo / 'src') + (
        os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')}
    try:
        for command in commands:
            freeze_plan(repo, root, [entry['suite'] for entry in plan['suites']], plan['gpus'])
            print(f"[{command['suite']}] {command['action']}", flush=True)
            process = subprocess.run(command['argv'], cwd=str(repo), env=env, check=False)
            result['stages'].append({**command, 'returncode': process.returncode})
            if process.returncode:
                result.update(status='failed', returncode=process.returncode)
                break
        else:
            result.update(status='commands_completed', returncode=0)
    except KeyboardInterrupt:
        result.update(status='interrupted', returncode=130)
    except Exception as error:
        result.update(status='failed', returncode=1, error=str(error))
    finally:
        result['finished_at'] = datetime.now(timezone.utc).isoformat()
        try:
            (root / 'attempts').mkdir(exist_ok=True)
            receipt = root / 'attempts' / f'{uuid.uuid4().hex}.json'
            with receipt.open('x') as stream:
                stream.write(json_text(result))
        finally:
            lock.unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('plan', 'run', 'summarize'), default='plan')
    parser.add_argument('--root', default='runs/aperture_rebuttal_20261104')
    parser.add_argument('--suite', choices=('all', *SUITES), default='all')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    suites = list(SUITES) if args.suite == 'all' else [args.suite]
    root = Path(args.root).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    try:
        plan = freeze_plan(ROOT, root, suites, args.gpus)
        result = execute_plan(plan, action='run' if args.action == 'plan' else args.action,
                              execute=args.execute and args.action != 'plan')
        print(json_text(result), end='')
        return int(result.get('returncode', 0))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'APERTURE rebuttal blocked: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
