"""Immutable evidence, atomic writes and process-released exclusive locks."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def atomic(path, obj, *, immutable=False):
    path = Path(path)
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists() and immutable:
        if path.read_text(encoding='utf-8') != text: raise ValueError(f'Frozen file changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f: f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_rows(path, data):
    path = Path(path)
    text = ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in data)
    if path.exists():
        if path.read_text(encoding='utf-8') != text: raise ValueError(f'Frozen rows changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


@contextmanager
def lock(path):
    import fcntl  # Execution contract: Linux/WSL; no unsafe lock-file deletion.
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError(f'Another process owns {path}') from e
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def seal(folder, binding):
    folder = Path(folder)
    if (folder / 'DONE.json').exists(): return verify(folder, binding)
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name != 'DONE.json' and not p.name.endswith('.lock')}
    if not files: raise ValueError('Cannot seal empty evidence')
    state = {'binding': binding, 'files': files}
    atomic(folder / 'DONE.json', state, immutable=True)
    return state


def verify(folder, binding=None):
    folder = Path(folder); state = read(folder / 'DONE.json')
    if binding is not None and state['binding'] != binding: raise ValueError(f'Evidence identity changed: {folder}')
    if not state.get('files'): raise ValueError('Empty evidence receipt')
    for name, expected in state['files'].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or not p.is_file() or sha(p) != expected:
            raise ValueError(f'Evidence changed or missing: {p}')
    return state


def sources(repo):
    repo = Path(repo)
    files = [p for p in (repo / 'src').rglob('*.py')]
    files += [repo / 'scripts/run_aperture_cases.py']
    return {str(p.relative_to(repo)): sha(p) for p in sorted(files) if p.is_file()}
