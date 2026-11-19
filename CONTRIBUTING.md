# Contributing to New Forge

New Forge is a research project. The most useful contributions are evidence: measurements, failed experiments, counter-examples, and corrections to existing records.

## Before you start

Read these first. They are the project's rules, and contributions are reviewed against them:

- [`AGENTS.md`](AGENTS.md): mandatory rules for any contributor, human or AI agent.
- [`docs/evaluation/EVIDENCE_STANDARD.md`](docs/evaluation/EVIDENCE_STANDARD.md): evidence labels and the minimum evidence chain.
- [`docs/principles/ARCHITECTURE_NEUTRALITY.md`](docs/principles/ARCHITECTURE_NEUTRALITY.md): why no mechanism is assumed.
- [`docs/STATUS.md`](docs/STATUS.md): what is established, what is open, and what is not yet authorized.

## Ways to contribute

- **Report evidence.** Open an *Evidence report* issue with a measurement, a reproduction, or a counter-example.
- **Propose an experiment.** Open an *Experiment proposal* issue. State the claim, the controlled variable, and what result would falsify it, before running it.
- **Correct a record.** If a document overstates a result or mislabels evidence, open an issue or a pull request with the correction and its source.

## Pull request rules

1. **Label your evidence.** Use the labels in the evidence standard: measured fact, documented result, observation, engineering inference, hypothesis, speculation, unresolved.
2. **Keep negative results.** Do not delete or rewrite a failed result. Add the new evidence and change the disposition.
3. **One major unknown at a time.** Experiments that change several things at once are hard to attribute and may be returned.
4. **No architecture by convenience.** A library, model, or API being available is not evidence that it is the right design. Major commitments need a decision record under `docs/decisions/`.
5. **Separate quality from performance.** Report speed with resolution, hardware, driver, precision, and what the timing includes.
6. **Credit sources.** Cite external results and credit test media under its license.

## Commit messages

Use a short prefix that says what changed: `docs:`, `research:`, `results:`, `decision:`. Explain *why* in the body when it isn't obvious.

## Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
