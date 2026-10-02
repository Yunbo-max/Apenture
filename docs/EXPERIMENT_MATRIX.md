# Experiment matrix

## Completed evidence retained without default reruns

| Block | Dataset / domains | Models | Status |
|---|---|---|---|
| Matched disaster adaptation | BRIGHT: Hawaii, Libya, Noto, Turkey; support24/query300; 3 support seeds | Qwen2.5-VL-7B | complete |
| Cross-backbone disaster transfer | same four BRIGHT events | Qwen3-VL-8B, InternVL3-8B | complete archived blocks |
| Cross-model pathology | CAMELYON17-H2, PathMNIST-224 | Qwen2.5-VL-7B, Qwen3-VL-4B/8B, Gemma-3-4B | complete |
| Calibration and robustness | BRIGHT, H2, ManipBench Q1 | Qwen2.5-VL-7B | complete 96-state round |
| Mechanism and boundaries | basis, layer, mask, objective, input controls, dose, RoboFail/ManipBench boundaries | recorded suite models | complete / exploratory as labeled |

## Required final validation

### 1. Five-hospital CAMELYON17

- Hospitals: 0--4; patient-disjoint fit/calibration/query.
- Seeds: 10, 11, 12.
- Budget: 24 fit + 8 calibration labels; 500 query patches per hospital.
- Models: `Qwen/Qwen2.5-VL-7B-Instruct` at 448 px and `OpenGVLab/InternVL3-8B-Instruct` at 224 px, BF16.
- Qwen arms: Frozen, LoRA-1E, LoRA-4E, Random, APERTURE.
- InternVL arms: Frozen, LoRA-1E, LoRA-4E, APERTURE.
- Total: 135 states, 105 trainable fits.
- Decision: negative or mixed hospitals narrow the cross-hospital claim; they do not trigger dataset or seed removal.

### 2. Rank-16 resource measurement

- Model: Qwen2.5-VL-7B-Instruct, BF16, 448 px.
- Domains: BRIGHT Hawaii seed0 and CAMELYON17-H2 seed0.
- Arms: Frozen, LoRA-1E, LoRA-4E, APERTURE.
- Two warm-ups, 12 complete candidate-scoring queries, three repeats.
- Report extraction, fitting, steady-state latency, allocated/reserved peak memory, optimized and stored scalars, serialized bytes, visits and optimizer updates.

## Optional, not submission-blocking

- Qwen five-hospital 16/32/64-label curve.
- Official LoReFT reproduction on H2 and one BRIGHT event if the paper makes a direct performance claim against representation fine-tuning.
