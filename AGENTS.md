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
11. **Keep the repository human-readable.** A future reader should be able to understand structure, rationale, evidence, and current status without access to project chats.
12. **Do not duplicate authority.** Update or link the canonical source document instead of creating parallel instructions.

## Git behavior

All Git and GitHub activity is governed by `docs/governance/GIT_GOVERNANCE.md`.

Unless a prompt explicitly grants narrower Git authority, worker agents are **edit/test only**: do not create or switch branches, commit, push, open/modify PRs, merge, rebase, force-push, tag, release, or delete refs.

Chat Mode coordinates branch/commit/PR integration and reviews the actual diff before presenting a PR for approval.

**Nothing is merged into `main` without explicit user approval.** If a reviewed PR changes after approval, it must be re-reviewed and approved again.

## Performance inheritance

General GPU/video engineering lessons may be reused only under `docs/principles/PERFORMANCE_INHERITANCE.md`.

The long-term target remains at least 30 FPS real-time playback and ideally 60 FPS on a broad GPU range. First-stage oracle experiments are not required to meet those numbers.

## Research context

For routine project work, prefer `docs/research/NEW_FORGE_RESEARCH_BASELINE.md` over loading the complete research corpus.

Consult `docs/research/foundation/` when primary-source detail, quantitative nuance, citation verification, or unresolved alternatives are required.

## Documentation authority

- `README.md` — project mission and public navigation.
- `docs/STATUS.md` — canonical current project/decision state.
- `docs/governance/*` — Git, documentation, prompting, and instruction-maintenance rules.
- `docs/principles/ARCHITECTURE_NEUTRALITY.md` — rules preventing assumption contamination.
- `docs/principles/PERFORMANCE_INHERITANCE.md` — boundary for reuse from earlier Forge work.
- `docs/evaluation/EVIDENCE_STANDARD.md` — evidence and promotion standard.
- `docs/research/NEW_FORGE_RESEARCH_BASELINE.md` — condensed current research context.
- `docs/research/foundation/*` — detailed research evidence; not automatic architecture authority.
- `docs/decisions/*` — explicit project decisions.

If authoritative documents conflict, stop and surface the conflict rather than choosing a convenient rule.
