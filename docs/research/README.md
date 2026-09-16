# Research

This directory preserves the research evidence used to select the architecture.

## Routine research context

Use [`NEW_FORGE_RESEARCH_BASELINE.md`](NEW_FORGE_RESEARCH_BASELINE.md) as the default condensed research context for chats, prompts, experiments, reviews, and architecture discussions.

It contains only the findings that currently matter to New Forge and intentionally avoids loading the entire research corpus into ordinary work.

## Current artifacts

### Foundation / targeted architecture-selection research

1. [`foundation/01-recoverable-information-limits.md`](foundation/01-recoverable-information-limits.md) — physical, sampling, blur, and codec limits on true recoverability.
2. [`foundation/02-correspondence-visibility-uncertainty.md`](foundation/02-correspondence-visibility-uncertainty.md) — what reconstruction needs to know about mapping, validity, occlusion, and confidence.
3. [`foundation/03-source-faithful-reconstruction-evaluation.md`](foundation/03-source-faithful-reconstruction-evaluation.md) — deterministic/hybrid reconstruction, data consistency, hallucination testing, and the experimental ladder.

These reports are preserved as research artifacts. They do not automatically define the implementation architecture.

Consult them when a task requires primary-source detail, quantitative nuance, citation verification, or unresolved alternatives that the condensed baseline intentionally omits.

## Current synthesis

See [`CURRENT_SYNTHESIS.md`](CURRENT_SYNTHESIS.md) for the broader cross-report interpretation.

## Research rules

- Preserve source artifacts instead of rewriting them into architecture documents.
- Put routine condensed context in `NEW_FORGE_RESEARCH_BASELINE.md`.
- Put cross-report interpretation in synthesis documents.
- Put actual engineering commitments in `../decisions/`.
- When a later experiment contradicts a research expectation, record the contradiction rather than editing history.
- When the condensed baseline becomes stale because evidence or decisions change, update it under `../governance/INSTRUCTION_MAINTENANCE.md`.
