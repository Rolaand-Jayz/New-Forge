# Project Status

**Date:** 2026-09-16  
**Phase:** Research synthesis and architecture decision-making  
**Implementation status:** No production architecture selected

## Purpose

This file is the canonical short-form snapshot of the project's current state.

It is intentionally narrower than the research reports and should answer four questions:

1. What is established strongly enough to guide work?
2. What remains unresolved?
3. What must not be treated as decided?
4. What work is allowed next?

When a major project decision is made, record it under `docs/decisions/` and update this file.

## Established working conclusions

### Temporal recovery is real but conditional

Multiple decoded frames can contain additional source-supported spatial information when they provide genuinely independent observations, such as useful subpixel phase diversity or complementary blur information.

Frame count alone is not useful evidence. Independent, valid observations are.

### Information loss is bounded by the measurement chain

Information destroyed by optical nulls, destructive filtering/resampling, severe quantization, chroma subsampling, or other irreversible processing cannot be recovered merely by adding more frames.

A future system must distinguish recoverable information from denoising, inference, interpolation, and hallucination.

### Temporal processing should be evidence-gated

Temporal reconstruction should not be applied uniformly simply because neighboring frames exist.

The project currently expects that useful temporal support may vary by region and potentially by spatial-frequency band, but the exact gate has not been selected.

### Correspondence is not enough

A useful temporal observation requires at least:

- where the candidate evidence maps;
- whether it is visible/valid for the target;
- how trustworthy it is.

Occlusion/disocclusion handling is therefore a core requirement for any serious temporal fusion path.

### Optical flow is not an architectural requirement

Explicit flow remains a valid baseline and candidate mechanism, but deformable offsets, feature correspondence, splatting, attention, multi-hypothesis correspondence, and other approaches remain open.

### Deterministic reconstruction should anchor the scientific reference

Initial oracle/reference work should favor transparent measurement-consistent approaches such as:

- nonuniform sampling/resampling;
- weighted least squares;
- robust L1/Huber/M-estimator fusion;
- iterative back-projection;
- frequency-domain reconstruction/diagnostics.

These are reference mechanisms, not predetermined production architecture.

### Reprojection is a primary fidelity test

When a forward model is available, reconstructed HR candidates should be projected back through that model and compared with the actual observations.

Reprojection residuals do not prove uniqueness, but they directly test measurement compatibility.

### Learned mechanisms remain allowed

Machine learning is not excluded.

Learned components must earn a defined responsibility and should be tested against transparent references. Measurement-consistency constraints are preferred where applicable.

### Performance remains part of the project definition

Early oracle experiments may be slow.

The eventual target remains:

- **minimum:** 30 FPS real-time reconstructed/upscaled playback;
- **desired:** 60 FPS where achievable without compromising reconstruction fidelity;
- **hardware:** broad GPU relevance rather than flagship-only viability.

General GPU/video performance knowledge from the earlier Forge may be reused only under `principles/PERFORMANCE_INHERITANCE.md`.

## Unresolved architecture decisions

The following are deliberately **not selected**:

- explicit optical flow vs implicit/deformable/feature correspondence;
- backward warping vs forward splatting vs another evidence-mapping model;
- one correspondence vs multiple hypotheses/layers per target region;
- confidence representation and calibration method;
- temporal-memory model;
- fixed vs adaptive temporal reach;
- causal vs bidirectional processing;
- forward-model/PSF estimation strategy;
- pixel/grid vs frequency/multi-scale vs continuous reconstruction representation;
- deterministic-only vs learned vs hybrid production reconstruction;
- learned model type, size, precision, or training strategy;
- GPU API and final backend architecture;
- final real-time approximation strategy.

Available libraries, models, prior code, or convenient implementations must not silently settle these questions.

## Historical boundary

The earlier FSR-centered Temporal Forge is a historical evidence and performance source.

This repository does **not** inherit:

- FSR 4.1 as the reconstruction center;
- FSR-specific temporal semantics or input contracts;
- synthetic renderer-style inputs merely because FSR expected them;
- FSR history/jitter/motion assumptions;
- any prior topology as architectural authority.

It may inherit independently valid performance engineering knowledge as defined in `principles/PERFORMANCE_INHERITANCE.md`.

## Next allowed work

### 1. Freeze the research basis

Treat the three foundation reports and `research/CURRENT_SYNTHESIS.md` as the current evidence basis. New research should target a specific unresolved decision rather than restart the broad survey.

### 2. Specify the experimental harness before production architecture

The first implementation work should support controlled reconstruction experiments with swappable stages and complete evidence capture.

The experimental ladder is:

1. known HR + known sampling/motion/blur;
2. estimated correspondence, known degradation;
3. estimated/unknown degradation;
4. independent motion, occlusion, disocclusion, and parallax;
5. realistic compression and noise;
6. real video with the strongest available reference methodology.

### 3. Introduce one major unknown at a time

Early tests should isolate causes. Do not combine correspondence, learned reconstruction, unknown blur, codec artifacts, and performance optimization in the same first experiment.

### 4. Preserve oracle/reference implementations

Optimized paths should be judged against transparent references. The project should not discard an oracle merely because it is too slow for production.

### 5. Record actual decisions separately

When evidence supports a major commitment, create a decision record under `docs/decisions/` with:

- decision;
- evidence;
- alternatives considered;
- rejected explanations;
- trade-offs;
- unresolved risks;
- reopen conditions.

## Not yet authorized by evidence

The following would be premature today:

- declaring the production architecture;
- selecting a neural network because it leads a benchmark;
- treating RAFT or any other flow system as required;
- creating a fixed temporal window without experimental support;
- optimizing around one GPU vendor before the algorithmic requirement exists;
- claiming 30/60 FPS feasibility for the final system from literature timings alone;
- claiming a reconstruction gain from visual sharpness without source-faithfulness evidence.

## Current source of truth

For detailed evidence, see:

- `research/CURRENT_SYNTHESIS.md`
- `research/foundation/01-recoverable-information-limits.md`
- `research/foundation/02-correspondence-visibility-uncertainty.md`
- `research/foundation/03-source-faithful-reconstruction-evaluation.md`
- `evaluation/EVIDENCE_STANDARD.md`
- `principles/ARCHITECTURE_NEUTRALITY.md`
- `principles/PERFORMANCE_INHERITANCE.md`
