#!/usr/bin/env python3
"""Standalone data repair for E07 manifests with empty normalized target answers.

Row 4 of textvqa/seed_0/fit.jsonl has all ten human answers normalizing to ''
under the pinned EvalAIAnswerProcessor, which the candidate-scoring path cannot
tokenize. This script (run OUTSIDE src so frozen plans elsewhere stay valid)
re-derives the target as the most common NON-empty normalized answer, falling
back to the most common raw answer. Query rows are left untouched (the
generation path does not score the label).

After running, the E07 run roots must be re-frozen and re-checked.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aperture_rebuttal_cases.data import evaluator  # noqa: E402  (loads pinned evaluator)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'data/prepared/aperture_rebuttal_cases_20261107_v1/textvqa'
CFG = json.loads((REPO / 'configs/aperture_reviewer_cases_v1.json').read_text())
EVAL = evaluator(CFG, REPO)
NORMALIZE = EVAL.EvalAIAnswerProcessor()


def main():
    fixed = []
    for path in sorted(ROOT.glob('seed_*/fit.jsonl')) + sorted(ROOT.glob('seed_*/calibration.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        changed = False
        for r in rows:
            if r.get('label') == '':
                answers = r.get('answers') or []
                norm = [NORMALIZE(a) for a in answers]
                counts = Counter(a for a in norm if a)
                if counts:
                    target = min(counts, key=lambda a: (-counts[a], a))
                else:
                    raw = Counter(answers)
                    target = min(raw, key=lambda a: (-raw[a], a))
                r['label'] = target
                r['candidate_labels'] = [target]
                r['label_id'] = 0
                changed = True
                fixed.append({'file': str(path.relative_to(ROOT)), 'sample_id': r['sample_id'],
                              'new_label': target, 'answers': answers[:3]})
        if changed:
            text = ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows)
            path.write_text(text)
    print(json.dumps({'fixed_rows': fixed}, indent=2))


if __name__ == '__main__':
    main()
