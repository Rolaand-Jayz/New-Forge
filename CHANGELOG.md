# Changelog

Notable changes to the project record. Dates are UTC.

## 2026-11-21

- **Code:** published the FORGE implementation from ANVIL commit `41ba911` under `implementations/forge/`: GPU shaders, the host-side stage, the PyTorch replica, the weight constructor, the default weights, and the gate scripts with their original outputs and input checksums. Labeled as implementation evidence, not a selected architecture. FSR-derived material is excluded.
- **Results:** the FORGE record links the published code and outputs, and adds the Sintel cave scene (part of the tuning set, so not held out).

## 2026-11-19

- **Results:** independent re-run reproduces the whole-frame means exactly; added tile-level figures, including the worst tile, which shows likely temporal ghosting on noisy input (observation, not root-caused).
- **Results:** recorded the FORGE protected-scene confirmation (`docs/results/2026-11-17-forge-protected-scene-confirm.md`). Disposition: supported with qualification.
- **Status:** added a dated update to `docs/STATUS.md`; the 2026-09-16 research snapshot is unchanged.
- **Project hygiene:** added a contributing guide, code of conduct, issue and pull request templates, a changelog, and citation metadata.

## 2026-09-16

- Initial public record: mission, project rules, evidence standard, architecture-neutrality and performance-inheritance principles, three foundation research reports, the current synthesis, and the canonical status snapshot.
