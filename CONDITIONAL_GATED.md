# Conditional-gated APERTURE — CG-v1

This branch adds an explicitly defined conditional-gated candidate and independent
reviewer cases. It does NOT recover an unlocated historical best method and does
NOT relabel any old result. No native VLM/GPU result is included in this delivery.

- Implementation: `src/aperture_conditional_gated/controller.py`
- Entry point: `scripts/run_aperture_gated.py`
- Runbook + reviewer mapping: `docs/APERTURE_CONDITIONAL_GATED_RUNBOOK.md`
- Exact design: `docs/APERTURE_CONDITIONAL_GATED_DESIGN.md`
- Validation boundaries: `docs/APERTURE_CONDITIONAL_GATED_VERIFICATION.md`

Begin with `python scripts/run_aperture_gated.py list`. Do not run the old
`run_aperture_cases.py` expecting a conditional gate.
