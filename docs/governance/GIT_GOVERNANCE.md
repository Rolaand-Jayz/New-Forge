# Git Governance

## Purpose

This document defines how Git and GitHub are managed for New Forge. **Chat Mode is the coordinating Git authority for this project.** Worker agents may edit, test, or inspect the repository only within the Git scope explicitly delegated in their prompt.

The goals are a clean history, reviewable changes, traceable decisions, and a repository whose current state can be understood without reconstructing chat history.

## Non-negotiable rules

1. Do not merge any PR or branch into `main` without explicit approval from the user.
2. Do not commit directly to `main` unless the user explicitly requests that exceptional workflow.
3. Do not force-push, rewrite published history, delete branches/tags, change release history, or rebase shared branches without explicit authorization.
4. A worker agent must not independently create branches, commits, PRs, tags, releases, merges, or rebases unless its prompt explicitly grants that exact action.
5. Chat Mode owns Git coordination: branch choice, commit review, PR review, merge recommendation, and instructions given to agents.
6. Research evidence, implementation changes, documentation changes, and performance work must remain distinguishable in history.
7. Unrelated changes do not ride along in a convenient commit or PR.

## Standard workflow

For substantive changes:

1. Fetch the current `main` state and check relevant open branches/PRs.
2. Create a narrowly scoped branch from current `main`.
3. Give agents an explicit Git scope when agents are involved.
4. Make atomic, human-readable commits.
5. Run the relevant tests, experiments, validation, and documentation checks.
6. Chat Mode reviews every commit and the aggregate diff before opening or recommending a PR.
7. Open a PR with a complete rationale and evidence summary.
8. Review CI/status checks and any new commits.
9. Present the PR to the user with risks, unresolved items, and a merge recommendation or objections.
10. **Do not merge until the user explicitly approves the merge.**
11. If the PR changes after approval, approval is stale: re-review the new diff and obtain approval again.
12. After merge, update canonical status/decision documentation when required. Branch deletion is optional and should not be performed as a consequential cleanup unless authorized by the user or clearly included in the approved workflow.

## Branch conventions

Use short, descriptive branches with a type prefix:

- `research/<topic>` — research artifacts or evidence synthesis;
- `experiment/<topic>` — controlled experimental harness/work;
- `feat/<topic>` — approved functionality;
- `fix/<topic>` — defect correction;
- `perf/<topic>` — performance work that preserves validated behavior;
- `docs/<topic>` — documentation/governance only;
- `refactor/<topic>` — behavior-preserving structural changes;
- `chore/<topic>` — maintenance that fits no stronger category.

One branch should represent one coherent purpose. Do not create long-lived catch-all branches.

## Commit conventions

Use atomic commits with imperative Conventional-Commit-style subjects:

`<type>(<scope>): <imperative summary>`

Useful types: `research`, `experiment`, `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `ci`, `chore`.

A commit should:

- have one understandable purpose;
- contain no unrelated cleanup;
- include tests/docs needed to make that commit internally coherent where practical;
- preserve traceability to the experiment, decision, issue, or PR that motivated it;
- never claim a result not established by the included evidence.

Avoid meaningless messages such as `updates`, `fix stuff`, `misc`, or model-generated summaries that obscure intent.

## Agent Git instructions

Every prompt to an agent that can modify repository state must contain a Git section stating:

- repository and base branch;
- assigned working branch/worktree, if any;
- allowed paths/scope;
- whether commits are allowed;
- required commit style if commits are allowed;
- whether the agent may open a PR;
- explicit prohibition on merge, force-push, history rewrite, tag/release changes, or branch deletion unless separately authorized;
- required final report: changed files, commit SHAs if any, tests, failures, unresolved questions, and repo state.

Default worker policy when not otherwise specified: **edit/test only; do not create or switch branches, commit, push, open/modify PRs, merge, rebase, force-push, tag, release, or delete refs.** Chat Mode handles those operations after review.

If parallel agents are used, give each an isolated branch/worktree or non-overlapping file ownership. Never let multiple agents race on the same branch or path without an explicit integration plan.

## Chat Mode commit review

Before accepting an agent commit into a PR, Chat Mode must inspect:

- exact changed-file list;
- full diff, not only the commit message;
- scope compliance;
- tests and experiment outputs;
- unexpected generated/binary files;
- documentation impact;
- architecture/decision impact;
- whether evidence claims match artifacts;
- whether unrelated formatting/refactoring was mixed in;
- whether secrets, machine-local paths, credentials, large temporary files, or personal data were introduced.

If a commit is mixed or misleading, fix or replace it on the branch before review. Do not merge it merely because tests pass.

## Pull request standard

Use PRs for all substantive code, experiment, research, governance, architecture, dependency, build-system, and source-of-truth documentation changes.

A PR body should answer:

- **Why** is this change needed?
- **What** changed?
- **Evidence/experiments** supporting it.
- **Tests/validation** performed.
- **Documentation/decision records** changed or intentionally unchanged.
- **Risks/regressions** and known limitations.
- **Unresolved questions**.
- **Performance impact**, when relevant.
- **Source-faithfulness impact**, when relevant.

Chat Mode should review the complete PR diff, commit list, CI/status checks, and discussion before presenting it for merge approval.

## Merge policy

`main` represents reviewed project truth.

Preferred default is squash merge when the branch's internal commits are work-in-progress detail that does not improve long-term history; preserve separate commits when they represent independently useful, coherent milestones. Chat Mode should recommend the merge form case by case.

No merge is permitted without explicit user approval. Approval applies to the reviewed PR state only. Any material post-approval change requires renewed review and approval.

## Dependencies, submodules, generated files, and large artifacts

- Adding or replacing a major dependency requires rationale and license/maintenance review; architecture-significant dependencies should have a decision record.
- Avoid submodules unless they solve a concrete versioning problem better than normal dependencies.
- Do not commit build outputs, caches, model downloads, raw experiment dumps, or large generated artifacts by default. Store only the minimum reproducible evidence or pointers/manifests needed to reproduce it.
- Add `.gitignore` rules when new tooling introduces local outputs.
- Binary/model assets require explicit provenance, license, size, and reproducibility consideration.

## Tags and releases

Tags/releases represent externally meaningful states and require explicit user approval. Release notes must identify the commit, validated functionality, known limitations, and evidence supporting major claims.
