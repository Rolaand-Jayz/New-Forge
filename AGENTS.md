# AGENTS.md

## Repository phase

This repository is in the **research, falsification, and architecture-selection phase**.

Do not treat the absence of an implementation as permission to invent one.

## Mandatory rules

1. **Architecture is unresolved.** Do not select optical flow, recurrence, transformers, deformable alignment, neural reconstruction, fixed temporal windows, or other major mechanisms without an explicit evidence-backed decision.
2. **Do not import FSR semantics.** The earlier FSR-centered Temporal Forge repository is a historical evidence and performance source, not an architecture specification.
3. **Preserve source-faithfulness.** Distinguish measurement-supported reconstruction from prior-driven synthesis.
4. **Prefer falsifiable experiments.** Change one major unknown at a time whenever practical.
5. **Keep oracle/reference paths.** Optimizations must be compared against transparent references instead of silently replacing them.
6. **Record negative results.** Failed mechanisms and regressions are evidence.
7. **Separate quality from performance.** First establish reconstruction behavior; then optimize while checking against the reference.
8. **Use explicit evidence labels** in research and decision documents: measured fact, documented result, observation, inference, hypothesis, speculation, unresolved.
9. **No architecture drift through convenience.** A library, model, GPU API, or available implementation is not automatically the correct design.
10. **Document architecture decisions.** Any change that commits the project to a major assumption belongs under `docs/decisions/`.

## Performance inheritance

General GPU/video engineering lessons may be reused only under `docs/principles/PERFORMANCE_INHERITANCE.md`.

The long-term target remains at least 30 FPS real-time playback and ideally 60 FPS on a broad GPU range. First-stage oracle experiments are not required to meet those numbers.

## Documentation authority

- `README.md` — project mission and current public status.
- `docs/principles/ARCHITECTURE_NEUTRALITY.md` — rules preventing assumption contamination.
- `docs/principles/PERFORMANCE_INHERITANCE.md` — boundary for reuse from earlier Forge work.
- `docs/evaluation/EVIDENCE_STANDARD.md` — evidence and promotion standard.
- `docs/research/*` — research evidence; not automatic architecture authority.
- `docs/decisions/*` — explicit project decisions.
