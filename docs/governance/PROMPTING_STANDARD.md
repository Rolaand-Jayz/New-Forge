# Prompting Standard

## Purpose

Prompts for New Forge should be task-specific, evidence-aware, and scoped to the current project phase. Do not use one giant generic prompt for every task.

The prompt should carry only the source instructions needed for the assigned work. Link to repository authority instead of pasting the entire research corpus.

## Source-selection rule

Before writing a prompt, determine which source documents actually govern the task.

Common routing:

- Git/repository work → `docs/governance/GIT_GOVERNANCE.md`
- documentation/repo organization → `docs/governance/DOCUMENTATION_AND_TRACEABILITY.md`
- recurring instruction/process changes → `docs/governance/INSTRUCTION_MAINTENANCE.md`
- current project state → `docs/STATUS.md`
- architecture contamination boundaries → `docs/principles/ARCHITECTURE_NEUTRALITY.md`
- prior-Forge performance reuse → `docs/principles/PERFORMANCE_INHERITANCE.md`
- evidence requirements → `docs/evaluation/EVIDENCE_STANDARD.md`
- routine research context → `docs/research/NEW_FORGE_RESEARCH_BASELINE.md`
- detailed research/source nuance → specific files under `docs/research/foundation/`
- accepted decisions → relevant files under `docs/decisions/`

Do not load all source files by default. Use the smallest authoritative set that is sufficient.

## Common prompt header

When relevant, establish:

- repository: `Rolaand-Jayz/New-Forge`;
- current phase from `docs/STATUS.md`;
- task objective;
- authoritative source files to read first;
- explicit scope and exclusions;
- expected deliverables;
- evidence/validation requirements;
- Git instructions if repository state can change.

## Mandatory Git block for repo-modifying tasks

Every prompt that may alter repository files must include a Git section consistent with `GIT_GOVERNANCE.md`.

At minimum state:

- assigned repository and branch/worktree;
- allowed paths and scope;
- whether branch creation is allowed;
- whether commits are allowed;
- required commit convention if allowed;
- whether PR creation is allowed;
- **no merge without explicit user approval**;
- no force-push, history rewrite, rebase of shared branches, tag/release change, or branch deletion unless specifically authorized;
- final report requirements: changed files, commits, tests, failures, unresolved issues, repo status.

Default worker behavior is edit/test only. Chat Mode handles branch/commit/PR integration unless the prompt explicitly delegates a narrower Git action.

## Research prompts

A research prompt should define:

- the decision blocked by the research;
- architecture-neutral/non-contamination constraints;
- exact research questions;
- evidence labels and source hierarchy;
- time cutoff when current research matters;
- what would falsify attractive hypotheses;
- required comparison dimensions;
- required unresolved-question output;
- artifact format constraints.

Research must distinguish established result, documented result, engineering inference, hypothesis, speculation, and unknown.

Do not ask research to choose an implementation merely because an implementation is familiar or available.

For Deep Research, use one coherent report artifact when product constraints favor a single artifact, with internal sections/tables/appendices rather than fragmented outputs.

## Experiment / implementation prompts

An experiment prompt should include:

- hypothesis or mechanism under test;
- why it is being tested now;
- reference/oracle path;
- independent/control variables;
- baselines;
- exact success/falsification criteria;
- required metrics and human-review checks;
- artifact/log requirements;
- allowed implementation scope;
- explicit non-goals;
- performance measurement only when relevant to the stage.

Introduce one major unknown at a time where practical.

An implementation prompt must distinguish **approved architecture** from exploratory mechanism. Do not let a worker turn a prototype into architectural authority.

## Performance prompts

Read `PERFORMANCE_INHERITANCE.md` first.

Require:

- validated reference behavior to preserve;
- target bottleneck;
- before/after measurements;
- hardware, resolution, precision, frame support, and timing method;
- quality delta versus reference;
- one optimization at a time where possible;
- rejection criteria when speed damages reconstruction fidelity.

Prior Forge techniques may be reused as engineering ideas, not as semantic requirements.

## Review / audit prompts

Reviews should be adversarial and evidence-based.

Require the reviewer to inspect the actual diff/artifacts and classify findings by severity. Typical checks:

- scope correctness;
- hidden architecture assumptions;
- unsupported claims;
- regressions or missing tests;
- stale/contradictory docs;
- traceability gaps;
- Git-policy violations;
- performance claims without comparable measurements;
- source-faithfulness risks;
- security/licensing/dependency issues where relevant.

Review prompts should default to **report only** unless fixes are explicitly authorized.

## Documentation prompts

Require the agent to identify:

- which document is authoritative;
- whether the change replaces, amends, or supplements existing information;
- links that need updating;
- duplication or contradictions introduced;
- whether `STATUS.md`, README, AGENTS, decisions, or governance docs are affected.

Prefer updating canonical docs over creating overlapping new documents.

## Decision prompts

When a task may commit the project to a major mechanism, first ask for a decision analysis rather than silently implementing the choice.

A decision prompt should require:

- decision question;
- evidence and experiments supporting each viable alternative;
- constraints/trade-offs;
- unresolved evidence;
- falsifiers/reopen conditions;
- proposed decision-record content.

The user retains final authority for major project decisions.

## Prompt quality rules

- Be explicit about authoritative inputs versus contextual inputs.
- Do not smuggle conclusions into the task statement when the task is meant to test them.
- Avoid vague verbs such as `improve`, `optimize`, or `clean up` without acceptance criteria.
- Preserve separation between research evidence, project decisions, implementation, and performance optimization.
- Require final reports to identify exactly what changed, what was learned, and what remains unresolved.
- If the task discovers a durable recurring rule not covered by source instructions, flag it under `INSTRUCTION_MAINTENANCE.md` rather than silently embedding it only in one prompt.
