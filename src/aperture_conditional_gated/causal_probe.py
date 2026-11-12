"""Audit answer CONTENT without changing native forward shapes or preprocessing.

This is a support-only diagnostic, never a training/scoring/generation path.
Both reference and probes use Backend.forward_batch on one prepared batch.
Only input_ids in the exact answer span change. Phi's negative image IDs,
images, attention/position/type tensors, suffix and masks remain untouched.
"""
from copy import deepcopy
from numbers import Integral

import torch

PROTOCOL = 'same-shape-answer-token-v1'
GATE_ATOL = .005  # Unchanged from the original audit; never a per-model override.


def _safe_text_ids(backend):
    processor = backend.processor
    tokenizer = getattr(processor, 'tokenizer', processor)
    vocab = tokenizer.get_vocab()
    special = set(getattr(tokenizer, 'all_special_ids', ()))
    configs = [getattr(backend, 'config', None)]
    base = getattr(backend, 'base', None)
    configs.append(getattr(getattr(base, 'language_model', None), 'config', None))
    for config in list(configs):
        if config is not None:
            configs.extend(getattr(config, name, None) for name in ('text_config', 'llm_config'))
    limits = []
    for config in configs:
        if config is None:
            continue
        size = getattr(config, 'vocab_size', None)
        if isinstance(size, Integral) and size > 0:
            limits.append(int(size))
        for name in ('image_token_id', 'image_token_index', 'img_context_token_id',
                     'video_token_id', 'vision_start_token_id', 'vision_end_token_id',
                     'bos_token_id', 'eos_token_id', 'pad_token_id'):
            value = getattr(config, name, None)
            if isinstance(value, Integral):
                special.add(int(value))
            elif isinstance(value, (list, tuple)):
                special.update(int(x) for x in value if isinstance(x, Integral))
    if not limits:
        raise ValueError('Causality probe requires the native text vocabulary size')
    size = min(limits)
    allowed = sorted({int(v) for v in vocab.values()
                      if isinstance(v, Integral) and 0 <= v < size and v not in special})
    if len(allowed) < 3:
        raise ValueError('Causality probe needs three distinct non-special text token IDs')
    return allowed[:3]


def _validate(batch, span, mask):
    ids = batch.get('input_ids')
    if (not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1
            or ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError('Causality probe requires one integer-token native input')
    if (len(span) != 2 or not all(isinstance(x, Integral) for x in span)
            or not 0 < span[0] < span[1] <= ids.shape[1]):
        raise ValueError('Invalid answer span for same-shape causality probe')
    start, end = map(int, span)
    if (not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != ids.shape
            or mask.device != ids.device or not mask.any() or mask[:, start:].any()):
        raise ValueError('Visual mask must be nonempty and wholly before the answer')
    attention = batch.get('attention_mask')
    if attention is not None and (attention.shape != ids.shape or not attention[:, start:end].bool().all()):
        raise ValueError('Answer span must contain unpadded active tokens')
    return start, end


def _copy_batch(batch):
    # Clone every tensor: a native image embedder may normalize input IDs in-place.
    return {k: v.detach().clone() if torch.is_tensor(v) else deepcopy(v) for k, v in batch.items()}


def _check_only_answer_changed(reference, candidate, start, end):
    if set(reference) != set(candidate):
        raise ValueError('Causality probe changed native fields')
    for key, left in reference.items():
        right = candidate[key]
        if torch.is_tensor(left):
            if (not torch.is_tensor(right) or right.shape != left.shape
                    or right.dtype != left.dtype or right.device != left.device):
                raise ValueError(f'Causality probe changed tensor shape/type/device: {key}')
            if key == 'input_ids':
                if not torch.equal(left[:, :start], right[:, :start]) or not torch.equal(left[:, end:], right[:, end:]):
                    raise ValueError('Causality probe changed non-answer token IDs')
            elif not torch.equal(left, right):
                raise ValueError(f'Causality probe changed non-answer native field: {key}')
        elif left != right:
            raise ValueError(f'Causality probe changed metadata: {key}')


def same_shape_gate_probe(backend, row, controller):
    """Return a receipt or fail closed. No padding, text retokenization or fallback.

    Two alternative token-ID sequences differ at EVERY answer position, from
    the reference and from each other. They need not spell candidate labels:
    this checks causal independence, not question accuracy. The reference is
    repeated once to distinguish within-shape instability from content effects.
    Full logits are discarded immediately; nothing is returned as a prediction.
    """
    try:
        if getattr(backend.model, 'training', False):
            raise ValueError('Causality probe requires eval mode')
        batch, span, mask, _ = backend.batch(row, row['label'])
        start, end = _validate(batch, span, mask)
        safe = _safe_text_ids(backend)
        answer = batch['input_ids'][:, start:end]
        a, b, c = safe
        # For each original token choose two different alternatives, never itself.
        first = torch.where(answer == a, b, a)
        second = torch.where((answer != a) & (answer != b), b, c)
        probes = [first, second]
        if any(not (x != answer).all() for x in probes) or not (first != second).all():
            raise ValueError('Causality probe became a no-op')
        expected = {f'{layer}:{kind}' for layer, kind, _ in controller.targets}
        expected_shape = (int(mask.sum()), 1)
        if not expected:
            raise ValueError('No registered visual gate sites to probe')
        controller.trace_enabled = True

        def forward(replacement=None):
            inputs = _copy_batch(batch)
            if replacement is not None:
                inputs['input_ids'][:, start:end] = replacement
            _check_only_answer_changed(batch, inputs, start, end)
            controller.trace = {}
            controller.set_mask(mask.clone())
            try:
                with torch.no_grad():
                    out = backend.forward_batch(inputs)
                del out  # Do not retain full-sequence logits between probes.
                if set(controller.trace) != expected:
                    raise RuntimeError('Missing/extra gate trace site in same-shape probe')
                result = {}
                for key, value in controller.trace.items():
                    if value.shape != expected_shape or not torch.isfinite(value).all():
                        raise RuntimeError('Invalid/nonfinite gate trace in same-shape probe')
                    result[key] = value.detach().float().cpu().clone()
                return result
            finally:
                controller.clear_mask()

        reference = forward()
        replay = forward()
        replay_delta = max(float((reference[k] - replay[k]).abs().max()) for k in expected)
        if replay_delta > GATE_ATOL:
            raise RuntimeError(f'Same-shape reference replay unstable: {replay_delta:.8g} > {GATE_ATOL}; audit blocked')
        results = []
        for replacement in probes:
            changed = forward(replacement)
            deltas = {k: float((reference[k] - changed[k]).abs().max()) for k in sorted(expected)}
            delta = max(deltas.values())
            if delta > GATE_ATOL:
                raise RuntimeError('Visual gate depends on future candidate content under identical shapes: '
                                   f'{delta:.8g} > {GATE_ATOL}; causal/prompt audit failed')
            results.append({'changed_answer_tokens': end-start, 'max_gate_delta': delta, 'per_site': deltas})
        return {'protocol': PROTOCOL, 'threshold': GATE_ATOL, 'probe_count': len(probes),
                'input_shape': list(batch['input_ids'].shape), 'answer_span': [start, end],
                'answer_tokens': end-start, 'reference_replay_delta': replay_delta,
                'max_gate_delta': max(r['max_gate_delta'] for r in results), 'probes': results,
                'forward_policy': 'native forward_batch, one row, eval/no_grad, identical full shapes',
                'unchanged': 'prefix/suffix IDs, image tensors, positions, attention/types, visual mask',
                'raw_text_retokenization': False, 'query_forwards': 0, 'performance_claim': False}
    finally:
        controller.trace_enabled = False
        controller.trace = {}
        controller.clear_mask()
