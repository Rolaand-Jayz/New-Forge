# Instruction Maintenance

## Purpose

New Forge source instructions must evolve deliberately. This document defines when recurring rules belong in source files, how conflicts are resolved, and how Chat Mode should propose and apply instruction changes.

The goal is to keep project-folder instructions short while making durable project behavior discoverable, reviewable, and versioned in Git.

## What belongs in source instructions

Create or update a durable source instruction when a rule is likely to matter across multiple chats, agents, experiments, or phases.

Strong update triggers include:

1. the user establishes or corrects a recurring workflow rule;
2. a repeated ambiguity causes inconsistent agent behavior;
3. a new project phase changes what work is allowed;
4. an accepted architecture/decision invalidates an old instruction;
5. a new Git/review/CI/release practice becomes standard;
6. a new source-of-truth document or directory changes authority routing;
7. a recurring prompting pattern is needed for a new task class;
8. an experiment exposes a missing evidence/traceability requirement;
9. a new tool, backend, dependency, model, dataset, or artifact workflow needs durable handling rules;
10. two existing source documents conflict or overlap enough to cause uncertainty.

Do not create a durable instruction for a one-off convenience unless it reveals a broader rule.

## Replace, amend, supplement, or retire

Every proposed source-instruction change must be classified before editing:

- **REPLACE** — the old rule is no longer valid and should be removed/substituted.
- **AMEND** — the same rule remains authoritative but needs correction, narrowing, or expansion.
- **SUPPLEMENT** — the old rule remains valid and a new non-conflicting rule is added.
- **RETIRE** — the rule no longer applies and should remain only in history, not current authority.

Do not silently stack new language on top of obsolete rules.

## Required Chat Mode behavior

When Chat Mode concludes that source instructions should change, it must first tell the user:

1. **Trigger:** what new fact, correction, failure, or recurring need caused the proposal.
2. **Affected authority:** exact files/sections involved.
3. **Change class:** replace, amend, supplement, or retire.
4. **Current rule:** concise description of what the repository currently says.
5. **Proposed rule:** concise description of the new authority.
6. **Consequences:** which prompts, Git behavior, docs, agents, or workflows would change.
7. **Conflict check:** any overlapping source instruction that must also be updated.

When the user approves the source-file change, Chat Mode should make the approved edits through the normal Git workflow and report exactly **what was replaced, amended, supplemented, or retired**.

Approval to edit source instructions does **not** automatically authorize merging the resulting PR. The PR remains subject to review and explicit merge approval under `GIT_GOVERNANCE.md`, unless the user explicitly includes merge approval in the same instruction.

## Canonical-placement rule

Put a rule in the narrowest canonical document that owns the concern:

- Git behavior → `GIT_GOVERNANCE.md`
- documentation/traceability → `DOCUMENTATION_AND_TRACEABILITY.md`
- prompting → `PROMPTING_STANDARD.md`
- instruction-evolution process → this file
- research facts/context → `docs/research/NEW_FORGE_RESEARCH_BASELINE.md`
- architecture constraints → `docs/principles/*`
- evidence/validation → `docs/evaluation/*`
- current phase/state → `docs/STATUS.md`
- accepted project choice → `docs/decisions/*`

If no existing document owns the concern and the rule is durable, propose a new narrowly scoped source file. Do not create a new file merely to avoid editing an existing authority.

## Contradiction handling

If source instructions conflict:

1. stop applying the conflicting rule;
2. identify the exact conflict to the user;
3. determine which document should own the authority;
4. propose replace/amend/retire actions;
5. update cross-references in the same PR;
6. do not leave both instructions active with informal precedence.

## Project-folder instruction maintenance

`PROJECT_FOLDER_INSTRUCTIONS.md` is a compact router, not the place for heavy policy.

Update it only when:

- the authority map changes;
- a new source file becomes essential routing context;
- a project-wide non-negotiable must be visible before source docs are loaded;
- the repo URL/project identity changes;
- the current project phase changes enough to alter default behavior.

Keep detailed rules in their source files and point to them from the project-folder instructions.

## Review cadence

There is no calendar-based requirement to rewrite instructions. Review them when triggered by actual project change.

Before major phase transitions—research → experimental implementation, experimental → production architecture, first public release—perform an explicit source-instruction audit for stale assumptions and authority gaps.
