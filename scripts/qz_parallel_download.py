#!/usr/bin/env python3
"""Multi-connection ranged downloader using curl workers (zenodo-friendly).

Usage: qz_parallel_download.py URL OUT [--chunks N] [--md5 HEX]
Downloads URL into OUT using N parallel curl --range requests, then concatenates
and (optionally) verifies the md5. Resumes existing complete chunks.
"""
import argparse
import hashlib
import os
import subprocess
import sys


def head_size(url):
    out = subprocess.run(
        ['curl', '-sSL', '--max-time', '30', '-r', '0-0', '-D', '-', '-o', '/dev/null', url],
        capture_output=True, text=True, timeout=60)
    ok = False
    for line in out.stdout.splitlines():
        low = line.lower()
        if low.startswith('http/'):
            ok = line.split()[1] in ('200', '206')
            continue
        if not ok:
            continue
        if low.startswith('content-range:') and '/' in line:
            total = line.split('/')[-1].strip()
            if total.isdigit():
                return int(total)
        if low.startswith('content-length:'):
            v = line.split(':')[1].strip()
            if v.isdigit() and int(v) > 1:
                return int(v)
    raise RuntimeError(f'no size from {url}: {out.stdout[-200:]}')


def fetch_range(url, start, end, path):
    expect = end - start + 1
    if os.path.exists(path):
        if os.path.getsize(path) == expect:
            return
        os.unlink(path)  # corrupt final chunk, redownload
    # per-process tmp name avoids collisions with other workers/runs
    tmp = f'{path}.tmp{os.getpid()}'
    r = subprocess.run([
        'curl', '-sSL', '--fail', '--retry', '10', '--retry-delay', '3',
        '--retry-all-errors', '--max-time', '1800', '-C', '-',
        '-r', f'{start}-{end}', '-o', tmp, url])
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.unlink(tmp)
        sys.exit(f'curl failed for {path}')
    if os.path.getsize(tmp) != expect:
        os.unlink(tmp)
        sys.exit(f'short chunk {path}')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url'); ap.add_argument('out')
    ap.add_argument('--chunks', type=int, default=16)
    ap.add_argument('--md5')
    a = ap.parse_args()
    total = head_size(a.url)
    part_dir = a.out + '.parts'
    os.makedirs(part_dir, exist_ok=True)
    for stale in os.listdir(part_dir):
        if '.tmp' in stale:
            os.unlink(os.path.join(part_dir, stale))
    size = total // a.chunks
    spans = []
    for i in range(a.chunks):
        start = i * size
        end = total - 1 if i == a.chunks - 1 else start + size - 1
        spans.append((start, end))
    procs = []
    for i, (s, e) in enumerate(spans):
        p = subprocess.Popen(
            [sys.executable, __file__, '--worker', a.url, str(s), str(e),
             os.path.join(part_dir, f'{i:02d}.bin')])
        procs.append(p)
    for p in procs:
        if p.wait() != 0:
            for q in procs:
                if q.poll() is None:
                    q.terminate()
            print('worker failed, aborting', file=sys.stderr)
            sys.exit(1)
    print(f'{a.out}: all {a.chunks} chunks done, concatenating...', flush=True)
    with open(a.out, 'wb') as out:
        for i in range(a.chunks):
            with open(os.path.join(part_dir, f'{i:02d}.bin'), 'rb') as f:
                while True:
                    b = f.read(4 << 20)
                    if not b:
                        break
                    out.write(b)
    if a.md5:
        h = hashlib.md5()
        with open(a.out, 'rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        ok = h.hexdigest() == a.md5
        print(f'md5: {h.hexdigest()} {"OK" if ok else "MISMATCH expected " + a.md5}', flush=True)
        sys.exit(0 if ok else 2)
    print(f'{a.out}: done ({total} bytes)', flush=True)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        flag, url, start, end, path = sys.argv[1:6]
        assert flag == '--worker'
        fetch_range(url, int(start), int(end), path)
    else:
        main()
