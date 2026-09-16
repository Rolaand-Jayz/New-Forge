# Decisions

Research findings do not become architecture decisions automatically.

Create a decision record when the project commits to a significant choice such as:

- correspondence representation;
- temporal support strategy;
- reconstruction formulation;
- learned vs deterministic responsibility split;
- forward-model assumptions;
- production GPU/API strategy;
- performance/quality trade-off that changes behavior.

## Decision-record template

```markdown
# YYYY-MM-DD — Decision title

## Decision

## Status
Proposed | Accepted | Rejected | Superseded

## Evidence basis

## Alternatives considered

## Consequences

## Unresolved risks

## Reopen conditions
```

## Current decisions

### 2026-09-16 — Clean reconstruction architecture

**Accepted.** This repository does not inherit the FSR-centered reconstruction architecture or FSR-specific semantic assumptions from the earlier Temporal Forge project.

General GPU/video performance knowledge may be reused under `docs/principles/PERFORMANCE_INHERITANCE.md`.

### 2026-09-16 — Remain in research/decision phase

**Accepted.** Do not scaffold a production engine merely for momentum. The first implementation work should be a controlled reconstruction research harness whose components remain replaceable until experiments justify architectural commitments.
