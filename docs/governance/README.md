# Governance

This directory contains the durable operating rules for New Forge.

These documents govern **how the project is maintained**, not what reconstruction architecture is selected.

## Files

- [`GIT_GOVERNANCE.md`](GIT_GOVERNANCE.md) — branches, commits, PRs, reviews, merge approval, agent Git scope, tags/releases, dependencies, and repository-history rules.
- [`DOCUMENTATION_AND_TRACEABILITY.md`](DOCUMENTATION_AND_TRACEABILITY.md) — human readability, authority, repository structure, decision/experiment traceability, and public-repository quality.
- [`PROMPTING_STANDARD.md`](PROMPTING_STANDARD.md) — how Chat Mode builds task-specific prompts and injects only the source instructions needed for each task.
- [`INSTRUCTION_MAINTENANCE.md`](INSTRUCTION_MAINTENANCE.md) — when source instructions must change and how replacements, amendments, supplements, and retirements are proposed and applied.
- [`PROJECT_FOLDER_INSTRUCTIONS.md`](PROJECT_FOLDER_INSTRUCTIONS.md) — compact router text for the ChatGPT project folder; detailed policy remains in the files above.

## Authority rule

Keep detailed policy here and keep higher-level routers short. If the same durable rule appears in multiple places, consolidate it under the narrowest appropriate authority and link to it elsewhere.

Governance changes use the same Git review process they define. They are not merged into `main` without explicit user approval.
