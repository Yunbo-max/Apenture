# APERTURE conditional-gated v1 — explicit new implementation

## Provenance
Base remote: Yunbo-max/Apenture, 43cb0ebb2d1dc4d29bb1ab5fb8eba75d4d47dd88.
The available history contains no recoverable conditional-gated formula or
checkpoint. This is a NEW, explicitly specified CG-v1 candidate, NOT recovery
of a historical best-performing implementation. No performance claim is made.
The submitted static method and every historical result remain unchanged.

## Definition
For each declared K/V projection site, preserve the support-gradient basis U
and bounded Full matrix A = alpha R / (1 + ||R||_F). Let h = z U and
h_norm = h / sqrt(mean(h**2) + 1e-6), computed in float32. CG-v1 uses a scalar
per-visual-token gate g = sigmoid(h_norm @ w + b) and
z_new = z + m * g * (h A U^T). It is conditioned on that token's current visual
state, NOT labels, candidate answers, query gradients, dataset identifiers,
previous image activations or calibration outcomes. It does not claim to see
question text that occurs causally after the visual tokens.
R,w,b are trainable; U and the backbone are frozen; alpha remains a
hyperparameter. R=w=b=0 initially: the residual is zero and the gate is 0.5.
Gate data gradients are zero on the first step when R=0; they become available
after R changes. A four-site rank16 model has 4*(256+16+1)=1092 trainable
scalars, not the legacy 1024. Store the basis, matrices, gates and metadata.

## Controls and scope
Keep the original token-mean answer loss and task-specific support reduction.
Use the original L2 on R plus the same explicit coefficient on trainable gate
parameters (both logged). No RL, loss mixing or controller ensembles.
Controls: frozen, legacy Full, fixed gate=0.5, learned input-independent scalar,
conditional Random/PCA bases, and strong support-selected LoRA/DoRA. A fixed
unit gate must reproduce Full exactly. An all-zero visual mask must be identity.
No gate implies a guarantee of lower overfitting, no forgetting or higher F1.

## Experiments and evidence
Use independent CG00–CG09 case roots, explicit GPU execution, frozen source /
model / manifests, per-unit file locks, complete-case aggregation with budget,
method family, seed and raw/scaled view retained. CG01 is the gated primary;
CG02 strong baselines; CG03 support16/32/64; CG04 mechanism; CG05 alpha/LR;
CG06 full storage/cost; CG07 free VQA; CG08 predeclared training budgets;
CG09 paired error analysis. CG00 is support-only native audit.
No reuse of legacy trained states as gated results. Existing data may be
reused, and comparisons on previously examined query sets are exploratory.
Official LoReFT remains optional and must pass a behavioral support audit;
the known frozen-identical historical row is not a validated strong baseline.
Use the same label/selection budgets; report actual compute, do not equate
optimizer steps or parameters. No query-based selection or hidden failures.
