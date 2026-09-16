# Documentation and Traceability

## Purpose

New Forge documentation must let a technically competent reader understand **what exists, why it exists, what is authoritative, what changed, and what evidence justified the change** without reconstructing chat history.

Documentation is part of the engineering system, not a retrospective cleanup step.

## Documentation principles

1. Prefer one canonical source for each kind of truth.
2. Link to authority instead of copying the same rule into multiple files.
3. Keep documents human-readable before making them machine-convenient.
4. Preserve superseded decisions and negative evidence rather than rewriting history.
5. Update documentation in the same PR as the change that invalidates it.
6. Keep the repository sparse. Create a new directory or document only when it has a durable purpose.
7. Every non-obvious directory should have an index/README explaining its purpose, contents, and authority.
8. Every file name should make its role understandable without opening it.
9. Avoid `misc`, `old`, `temp`, `notes2`, and other ambiguous dumping grounds.

## Public-repository quality standard

Assume this repository may become public and may be read by engineers, collaborators, or employers who have no access to project chats.

Do not write documentation as a portfolio advertisement. Instead, let disciplined execution be visible through the repository itself:

- coherent structure;
- restrained file count;
- clear ownership of concepts;
- traceable decisions;
- reproducible experiments;
- explicit evidence and uncertainty;
- focused commits;
- reviewable PRs;
- honest negative results;
- clean separation between research, decisions, implementation, and optimization;
- documentation that explains intent without requiring tribal knowledge.

Automation or agent assistance should never make the repository less human-readable. Generated work must meet the same standard as manually authored work and should not contain generic filler, unnecessary ceremony, duplicated summaries, or tool-centric narration.

The repository should demonstrate orchestration quality implicitly through consistency and traceability, not through claims about orchestration.

## Authority map

- `README.md` — concise public project overview and navigation.
- `docs/STATUS.md` — canonical current phase, established constraints, unresolved decisions, and next allowed work.
- `AGENTS.md` — minimum repository-working rules every coding/research agent should obey.
- `docs/governance/*` — durable process rules for Git, prompts, documentation, and instruction maintenance.
- `docs/principles/*` — architectural/engineering principles that constrain choices.
- `docs/evaluation/*` — evidence and validation standards.
- `docs/research/NEW_FORGE_RESEARCH_BASELINE.md` — condensed research context for routine project work.
- `docs/research/foundation/*` — detailed research evidence and source material.
- `docs/research/CURRENT_SYNTHESIS.md` — broader research synthesis; not automatic architecture authority.
- `docs/decisions/*` — explicit accepted/rejected/superseded project decisions.

If two authoritative documents conflict, stop and resolve the authority conflict rather than choosing whichever text is convenient.

## Decision traceability

Major architecture, dependency, evaluation, performance, data, and workflow decisions should have a decision record when the reasoning will matter later.

Recommended decision file naming:

`NNNN-short-decision-title.md`

Each decision record should include:

- Status: `PROPOSED`, `ACCEPTED`, `REJECTED`, `SUPERSEDED`, or `REOPENED`;
- date;
- question/decision being made;
- context and constraints;
- evidence used, with links to research/experiments/PRs/commits;
- alternatives considered;
- decision and rationale;
- consequences/trade-offs;
- unresolved risks;
- falsifiers/reopen conditions;
- supersedes/superseded-by links when applicable.

Do not convert research suggestions into accepted decisions without an explicit decision step.

## Experiment traceability

Each controlled experiment should be reproducible from the repository record or from clearly referenced external artifacts. Record at minimum:

- hypothesis/claim;
- controlled and independent variables;
- code/commit identity;
- input/dataset identity;
- configuration;
- hardware/driver when performance is measured;
- baselines/controls;
- metrics and human-review protocol;
- output artifact locations/manifests;
- invalid runs and why they are invalid;
- result;
- evidence classification and disposition under `docs/evaluation/EVIDENCE_STANDARD.md`;
- decision impact, if any.

Raw large outputs do not need to live in Git. The repository should retain enough metadata/manifests to reproduce and audit them.

## Human readability

Prefer prose that explains causality and purpose over dense shorthand. Tables are useful for comparisons but should not replace the explanation of why a result matters.

A new contributor should be able to answer from repository documentation:

- What problem is New Forge solving?
- What phase is it in?
- Which assumptions are established, provisional, or prohibited?
- What is the current architecture decision state?
- Where are the supporting experiments/research?
- Why does this directory/file exist?
- What would invalidate or reopen a major decision?
- What is the next legitimate task?

## Repository design

Keep the top level intentionally small. Add top-level directories only for durable major concerns.

When implementation begins, prefer clear domain-oriented directories rather than technology-fashion names. The exact source layout should follow accepted architecture decisions, not precede them.

Do not create empty architecture scaffolding merely to make the repository appear mature.

## Documentation maintenance triggers

Review and update affected canonical docs when any of these occur:

- project phase changes;
- an architecture decision is accepted/rejected/reopened;
- an experiment changes an established assumption;
- a new major dependency/backend/model becomes authoritative;
- a repository directory changes purpose;
- a workflow or Git rule changes;
- a public claim in the README becomes inaccurate;
- performance targets or supported hardware scope changes;
- the condensed research baseline becomes stale.

Prefer modifying the existing canonical document over creating a parallel document with overlapping authority.

## PR documentation review

Before recommending a PR for merge, Chat Mode should ask:

1. Does this change make any existing documentation false or incomplete?
2. Does it create a new concept that needs a durable name/location?
3. Is a decision record required?
4. Is `docs/STATUS.md` affected?
5. Is the README affected?
6. Did the PR duplicate information that should instead be linked?
7. Can a future reader trace the change to evidence and rationale?
8. Would the change make sense to a public reader who has never seen the project chats?

A code-complete PR is not documentation-complete if its authoritative docs are stale.
