# Claim freeze

APERTURE is a support-fitted, domain-level visual lens for frozen VLMs. It estimates task-sensitive directions from labeled support, then learns a bounded dense transformation within those fixed directions and applies it only to selected visual-token K/V projection outputs. The fitted operator is reused on each query's fresh activations without per-query optimization or support-activation storage.

The paper does **not** claim that gradient eigenspaces, K/V intervention, representation fine-tuning, or low-rank control are individually new. The defensible contribution is their visual-row-selective, bounded, support-fitted, query-reusable combination and the empirical account of when it works.
