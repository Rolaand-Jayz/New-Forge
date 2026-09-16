# Current Research Synthesis

**Date:** 2026-09-16  
**Status:** Working synthesis — not an architecture specification

## What the three reports jointly establish

### 1. Temporal recovery is physically real but conditional

Multiple observations can recover genuine spatial information beyond a single decoded frame when they provide independent sampling evidence: useful subpixel phase differences, differing invertible blur information, or other complementary measurements.

Frame count alone is not the resource. **Independent evidence is.**

Information destroyed by common optical nulls, destructive resampling, severe quantization, or other irreversible processing cannot be recreated by temporal fusion without relying on priors.

### 2. The system should probably detect temporal opportunity

Temporal reconstruction should not be unconditional. Regions with useful diversity may benefit strongly; regions with redundant observations, destructive blur, poor SNR, occlusion, or unreliable correspondence may require reduced temporal support or spatial-only fallback.

This suggests a future **evidence-gated** reconstruction strategy, potentially local and frequency-aware. The exact gate remains unresolved and requires experiment.

### 3. Correspondence is a three-part evidence problem

A useful temporal observation needs at least:

1. **where** it maps;
2. whether it is **visible/valid** for the target;
3. how **confident** the system should be in it.

Explicit optical flow is therefore a candidate representation, not the architecture. Deformable offsets, feature correspondence, splatting, attention, and multi-hypothesis representations remain viable research paths.

### 4. Occlusion/disocclusion cannot be ignored

Naive fusion can combine unrelated surfaces and create ghosting or false structure. The reconstruction path needs a way to reject invalid observations and represent holes/newly visible regions.

Fine structures, transparency, boundaries, hair, foliage, and overlapping motion may require more than one candidate correspondence per target location.

### 5. Temporal reach should remain adaptive until evidence says otherwise

No fixed window has earned authority. Adjacent frames may be useless while a more distant observation is uniquely informative. Conversely, long-range samples may become redundant or unsafe.

The value of distant evidence must be measured rather than assumed.

### 6. Measurement-consistent deterministic reconstruction should be the scientific reference

The strongest first baselines are transparent methods such as:

- nonuniform resampling / shift-and-add;
- weighted least squares;
- robust L1/Huber/M-estimator fusion;
- iterative back-projection;
- frequency-aware reconstruction diagnostics.

These are not necessarily production methods. They are intended to establish what information is actually recoverable before faster approximations are judged.

### 7. Reprojection should be a primary fidelity check

Given a reconstructed HR candidate, reapply the best available forward model and compare the predicted LR observations against the actual source frames.

This does not solve every ambiguity, especially when the forward model is uncertain, but it provides a direct measurement-consistency test that conventional sharpness metrics cannot.

### 8. Learned components remain open, but their authority should be constrained

Learning may be valuable for correspondence, confidence, denoising, residual correction, learned proximal steps, or other specialist roles.

The most attractive learned directions are those that can remain bounded by explicit measurement/data-consistency checks. End-to-end perceptual sharpness alone is insufficient evidence of source-faithful recovery.

## Experimental ladder

The reports converge on a staged falsification program:

1. **Known everything** — HR ground truth, exact subpixel sampling, exact blur/downsampling.
2. **Estimated correspondence** — remove oracle motion while keeping degradation known.
3. **Estimated degradation** — introduce PSF/forward-model uncertainty.
4. **Dynamic scenes** — independent motion, occlusion, disocclusion, parallax.
5. **Compression and noise** — realistic codec and sensor degradation.
6. **Real video** — paired or strongest available pseudo-ground-truth evaluation.

The intent is to introduce one major unknown at a time so failures remain attributable.

## Performance position

The first oracle paths may be slow.

The production goal remains:

- at least **30 FPS** real-time playback;
- **60 FPS** if achievable without sacrificing the reconstruction objective;
- broad GPU relevance rather than flagship-only execution.

Earlier Temporal Forge work demonstrated valuable high-throughput engineering patterns. Those may be reused under the performance-inheritance rules, but they cannot dictate reconstruction semantics.

## Still unresolved

- exact temporal-opportunity gate;
- flow vs implicit/deformable/multi-hypothesis correspondence;
- best calibrated confidence representation;
- alignment-error tolerance by reconstruction family;
- adaptive temporal-memory policy;
- practical treatment of layered/transparent/fine-structure correspondence;
- forward-model/PSF estimation strategy;
- learned-component boundaries;
- exact route from oracle-quality reconstruction to broad-GPU 30/60 FPS.

## Immediate consequence

The next implementation should be a **research harness**, not a production upscaler. It should make oracle and candidate stages swappable, retain ground truth and intermediate evidence, and make quality, consistency, and performance measurable independently.
