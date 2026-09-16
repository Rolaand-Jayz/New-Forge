# Documentation

This repository separates **research evidence**, **project principles**, **evaluation rules**, **governance**, and **engineering decisions** so that one cannot silently become another.

## Authority map

### `STATUS.md`
Canonical snapshot of the current project phase, established constraints, unresolved decisions, and next allowed work.

### `governance/`
Durable operating rules for Git/GitHub, documentation and traceability, prompt construction, and instruction maintenance. It also contains the compact ChatGPT project-folder instruction router.

### `research/`
Research artifacts, condensed research context, and project syntheses.

Research informs decisions but does not automatically define architecture. Routine work should start with `research/NEW_FORGE_RESEARCH_BASELINE.md`; consult the full foundation reports only when deeper evidence is needed.

### `principles/`
Rules that constrain how architecture is selected and what may be inherited from prior work.

### `evaluation/`
The evidence standard used to decide whether a mechanism, result, or optimization is actually supported.

### `decisions/`
Explicit choices the project has made, along with evidence, alternatives, unresolved risks, and reopen conditions.

## Current phase

The project is still in **decision making**. Implementation directories should not be created merely to make the repository look active. They should appear when experiments justify them.

## Key documents

- [`STATUS.md`](STATUS.md)
- [`governance/README.md`](governance/README.md)
- [`governance/GIT_GOVERNANCE.md`](governance/GIT_GOVERNANCE.md)
- [`governance/DOCUMENTATION_AND_TRACEABILITY.md`](governance/DOCUMENTATION_AND_TRACEABILITY.md)
- [`governance/PROMPTING_STANDARD.md`](governance/PROMPTING_STANDARD.md)
- [`governance/INSTRUCTION_MAINTENANCE.md`](governance/INSTRUCTION_MAINTENANCE.md)
- [`governance/PROJECT_FOLDER_INSTRUCTIONS.md`](governance/PROJECT_FOLDER_INSTRUCTIONS.md)
- [`principles/ARCHITECTURE_NEUTRALITY.md`](principles/ARCHITECTURE_NEUTRALITY.md)
- [`principles/PERFORMANCE_INHERITANCE.md`](principles/PERFORMANCE_INHERITANCE.md)
- [`evaluation/EVIDENCE_STANDARD.md`](evaluation/EVIDENCE_STANDARD.md)
- [`research/NEW_FORGE_RESEARCH_BASELINE.md`](research/NEW_FORGE_RESEARCH_BASELINE.md)
- [`research/README.md`](research/README.md)
- [`decisions/README.md`](decisions/README.md)
