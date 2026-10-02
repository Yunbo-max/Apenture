# APERTURE Canonical Migration Implementation Plan

> **For agentic workers:** execute source locking, additive result import, facade creation, final-suite registration, manuscript import, static validation, CPU tests, and archive-ref publication in that order. Do not launch paid GPU runs during migration.

**Goal:** Produce one auditable, runnable APERTURE repository containing all public code, compact evidence, manuscript sources, provenance, and exact remaining experiment definitions.

**Architecture:** Build canonical main from the pinned StateAdaptation commit; import branch-specific measured exports as additive directories; retain source refs as archive branches; expose old packages through a new `aperture` facade.

**Verification:** `python3 scripts/validate_release.py`, `pytest -q tests_release`, source-lock checks, and exact archive-ref checks must pass before publication.
