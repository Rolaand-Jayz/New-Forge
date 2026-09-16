# New Forge

> **Decision-stage research project for source-faithful temporal video reconstruction.**

New Forge is an open research and engineering project investigating how ordinary decoded video can be reconstructed at higher spatial fidelity by combining evidence distributed across time.

The project begins from the reconstruction problem itself rather than from an existing game upscaler, vendor pipeline, or preselected machine-learning architecture.

## Current status

**Research and architecture decision phase.**

No production reconstruction architecture has been selected. The repository is intentionally being structured before implementation so that experiments, evidence, decisions, and later performance work remain traceable.

The current research supports a bounded premise: multiple frames can contain genuinely recoverable spatial information when they provide useful independent observations, but the gain is conditional on sampling diversity, blur, motion, visibility, alignment, compression, noise, and reconstruction method.

The current working direction is therefore **evidence-gated temporal reconstruction** rather than unconditional temporal accumulation.

See [`docs/STATUS.md`](docs/STATUS.md) for the canonical current decision state.

## Objective

Recover as much **genuine source-supported spatial detail** as practical from low-resolution, blurry, compressed, noisy, or bandwidth-constrained video by exploiting useful information distributed across multiple frames.

The project prioritizes:

- reconstruction over generative reinterpretation;
- measurement consistency over visual sharpness alone;
- explicit uncertainty and failure detection;
- controlled experiments over architecture-by-assumption;
- a clear distinction between recovered information and prior-driven synthesis.

The long-term deployment objective is real-time playback:

- **minimum target:** 30 reconstructed/upscaled frames per second;
- **desired target:** 60 FPS where achievable without compromising the reconstruction objective;
- **hardware goal:** a broad range of GPUs, including older but still practically useful hardware where feasible;
- **implementation goal:** vendor-neutral design where technically reasonable.

Early experiments are quality-first and are not required to run in real time.

## Research before architecture

The repository starts with a foundation research record and targeted follow-up studies covering:

1. recoverable information limits in real video;
2. correspondence, visibility, and uncertainty;
3. source-faithful reconstruction and evaluation.

See [`docs/research/`](docs/research/README.md).

These reports inform experiments and candidate mechanisms. They do **not** constitute an implementation specification.

## Core project rules

1. **Every architectural component must earn its inclusion through evidence.**
2. **Temporal evidence is conditional.** More frames are not automatically more information.
3. **Correspondence is not equivalent to validity.** A sample must also be visible and trustworthy.
4. **Optical flow is a candidate mechanism, not an architectural requirement.**
5. **Learned methods are allowed, but unsupported detail must not be mislabeled as recovered detail.**
6. **Reprojection/data consistency is a primary validation tool.**
7. **Negative results are retained.** A falsified idea is useful project evidence.
8. **Human visual review may reopen a metric-green result.**
9. **Performance knowledge may be inherited; reconstruction assumptions may not.** See [`PERFORMANCE_INHERITANCE.md`](docs/principles/PERFORMANCE_INHERITANCE.md).

## Repository map

```text
.
├── README.md
├── LICENSE
├── AGENTS.md
└── docs/
    ├── README.md
    ├── STATUS.md
    ├── decisions/
    │   └── README.md
    ├── evaluation/
    │   └── EVIDENCE_STANDARD.md
    ├── principles/
    │   ├── ARCHITECTURE_NEUTRALITY.md
    │   └── PERFORMANCE_INHERITANCE.md
    └── research/
        ├── README.md
        ├── CURRENT_SYNTHESIS.md
        └── foundation/
            ├── 01-recoverable-information-limits.md
            ├── 02-correspondence-visibility-uncertainty.md
            └── 03-source-faithful-reconstruction-evaluation.md
```

The structure will grow only when the work requires it. Implementation directories will be introduced after the first experimental architecture is justified.

## Relationship to earlier work

This project follows an earlier FSR-centered Temporal Forge research phase, but it does **not** inherit that project's reconstruction model or FSR-specific semantic assumptions.

Engineering lessons from that work—especially high-throughput native GPU video processing—remain valuable and may be reused when independently appropriate. Architectural assumptions do not transfer merely because they existed before.

## License

MIT. See [`LICENSE`](LICENSE).
