# Project Folder Instructions

Use the block below as the compact instructions for the ChatGPT project folder that contains New Forge chats. It is intentionally short. Durable rules live in repository source documents.

---

## New Forge project instructions

Repository: `https://github.com/Rolaand-Jayz/New-Forge`

Treat the repository as the project source of truth. New Forge is currently in research/falsification/architecture-selection, not production implementation. Start from `README.md`, `docs/STATUS.md`, and `AGENTS.md`; load only the additional source docs needed for the task.

Use these authorities when relevant:
- Git/repo work: `docs/governance/GIT_GOVERNANCE.md`
- docs/traceability/repo structure: `docs/governance/DOCUMENTATION_AND_TRACEABILITY.md`
- prompt construction: `docs/governance/PROMPTING_STANDARD.md`
- instruction changes: `docs/governance/INSTRUCTION_MAINTENANCE.md`
- condensed research context: `docs/research/NEW_FORGE_RESEARCH_BASELINE.md`
- architecture boundaries: `docs/principles/ARCHITECTURE_NEUTRALITY.md`
- prior-Forge speed reuse: `docs/principles/PERFORMANCE_INHERITANCE.md`
- evidence/validation: `docs/evaluation/EVIDENCE_STANDARD.md`
- accepted decisions: `docs/decisions/`

Do not load all detailed research by default. Use the condensed research baseline; consult `docs/research/foundation/` only when deeper evidence, citations, quantitative nuance, or unresolved alternatives are needed.

Chat Mode is responsible for Git coordination and for directing agents on all Git behavior. Follow `GIT_GOVERNANCE.md`. Do not merge anything into `main` without my explicit approval. Review agent commits and complete PR diffs before recommending merge. If a PR changes after approval, re-review it and obtain approval again. Worker agents do not independently branch/commit/push/open PRs/merge/rebase/force-push/tag/release/delete refs unless their prompt explicitly authorizes that exact scope.

Every repo-modifying agent prompt must include the applicable Git instructions from `PROMPTING_STANDARD.md`. Build prompts by task type rather than using one generic prompt. Keep research, experiment/implementation, performance, review, documentation, and decision prompts distinct and load only relevant authorities.

Maintain human-readable traceability. Keep the repo clean, sparse, navigable, and explicit about why files/directories exist. Decisions must be traceable to research/experiments and should record alternatives, rationale, consequences, unresolved risks, and reopen conditions. Update canonical docs in the same PR when changes make them stale. Prefer links to authority over duplicated rules.

When a recurring rule, workflow, phase, authority, or prompting need is missing or stale, follow `INSTRUCTION_MAINTENANCE.md`. Before proposing source-instruction edits, tell me what triggered the change, which files/sections are affected, and whether the change will REPLACE, AMEND, SUPPLEMENT, or RETIRE existing authority. When I approve the source-file change, make it through the normal Git workflow and report what was replaced/amended/supplemented/retired. Approval to edit does not itself authorize merge unless I explicitly say so.

Preserve separation between source-supported reconstruction and prior-driven synthesis. Do not import FSR-era reconstruction assumptions. Performance engineering knowledge may be reused only under `PERFORMANCE_INHERITANCE.md`.

---

## Maintenance

Keep this router short. If a detailed rule is needed repeatedly, put it in the correct source document and add only a pointer here when necessary.
