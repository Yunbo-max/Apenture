"""Mask nonexistent per-example candidate positions for shared calibration code."""
import numpy as np


def score_arrays(rows, classes=None):
    if not rows:raise ValueError('empty candidate scores')
    width=classes if classes is not None else max(len(r['mean_log_scores']) for r in rows)
    # Finite log-zero mask, exactly zero after exp for the fixed T <= 20 grid.
    # Raw prediction files contain only the candidates actually scored.
    scores=np.full((len(rows),width),-1e30,dtype=np.float64)
    labels=np.array([r['label_id'] for r in rows],dtype=int)
    for i,row in enumerate(rows):
        values=np.asarray(row['mean_log_scores'],dtype=np.float64)
        if values.ndim!=1 or not 2<=len(values)<=width or not np.isfinite(values).all() or not 0<=labels[i]<len(values):
            raise ValueError('invalid per-example candidate scores')
        scores[i,:len(values)]=values
    return scores,labels
