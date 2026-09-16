# New Forge Research Baseline

**Status:** Canonical condensed research context for routine New Forge work  
**Source basis:** `docs/research/foundation/` and `docs/research/CURRENT_SYNTHESIS.md`  
**Authority:** Research context, not an implementation specification

## Purpose

This file condenses the research findings that materially constrain or inform New Forge. Use it as the default research context for chats, prompts, experiment design, reviews, and architecture discussions.

Read the full foundation reports when a task requires primary-source detail, quantitative nuance, citation verification, or deeper comparison. Do not load the entire research corpus into routine prompts.

## Core research position

New Forge is investigating **source-faithful temporal video reconstruction**: recovering spatial detail that is supported by ordinary decoded video observations across time, rather than generating merely plausible detail.

The central premise is **qualified but supported**: multiple frames can contain genuine recoverable information beyond a single frame when they provide independent observations. That opportunity is conditional rather than universal.

## What temporal evidence can provide

Useful additional information can arise from:

- subpixel sampling diversity from camera/object motion;
- different sampling phases;
- complementary blur/PSF information when blur nulls differ;
- repeated observations with different noise realization;
- changing visibility or viewpoint that reveals previously unavailable structure.

**Frame count alone is not information. Independent evidence is.**

Repeated identical samples can improve SNR but do not create new frequency support.

## Hard information boundaries

Information destroyed by common optical nulls, clipping, destructive resampling, severe codec quantization, or other irreversible stages cannot be reconstructed from measurements that no longer contain it.

New Forge must distinguish:

1. directly recoverable detail;
2. model-constrained recoverable detail;
3. statistically inferred detail;
4. prior-driven synthesis;
5. unsupported hallucination.

Output scale is not the same thing as recovered-information scale. Real-world burst evidence suggests genuine temporal resolution gain often saturates well before arbitrary upscale factors; favorable practical regimes around roughly 1.5–2x recovered spatial gain are documented, but this is not a universal project cap.

## Evidence-gated temporal reconstruction

Temporal processing should not run simply because neighboring frames exist.

The likely direction is local, potentially frequency-aware opportunity detection that can reduce or disable temporal contribution when observations are:

- redundant;
- too blurred;
- too noisy;
- destructively compressed;
- occluded/disoccluded;
- poorly matched;
- unsupported by surviving frequency content.

A region may have strong temporal evidence while another region in the same frame has none. The exact opportunity gate is unresolved and must be measured.

## Correspondence requirements

A usable temporal observation needs at least:

1. **where** candidate evidence maps;
2. whether it is **visible/valid** for the target;
3. how **confident** the system should be in it.

Subpixel precision matters for high-frequency recovery, but explicit optical flow is only one candidate representation.

Viable research families include:

- classical registration;
- learned optical flow;
- feature correspondence;
- deformable offsets;
- dynamic kernels;
- splatting;
- attention/correlation;
- multi-hypothesis or layered correspondence.

Better flow metrics do not automatically imply better reconstruction quality.

## Occlusion, disocclusion, and ownership

Naive fusion can combine unrelated foreground/background evidence and create ghosting or false structure. Occlusion/disocclusion must be handled explicitly or by an equivalent validated mechanism.

Thin structures, hair, foliage, fences, transparency, reflections, motion boundaries, and overlapping surfaces may require multiple candidate correspondences or soft/layered ownership rather than one motion vector per target location.

## Temporal extent

No fixed temporal window has earned authority.

Adjacent frames can be redundant or invalid while a more distant frame may contain unique evidence. Conversely, long-range samples may become unreliable or add nothing.

Past-only, bidirectional, recurrent, memory-bank, and adaptive-support designs remain hypotheses until experiments establish their benefit.

## Scientific reconstruction references

The first reconstruction references should be transparent and measurement-consistent rather than production-optimized. Useful baseline/oracle families include:

- nonuniform resampling / shift-and-add for controlled cases;
- weighted least squares / MLE-style reconstruction;
- robust L1/Huber/median/M-estimator fusion;
- iterative back-projection;
- frequency-aware reconstruction/diagnostics.

These are scientific reference paths, not presumed production architecture.

## Reprojection and data consistency

Given an HR reconstruction, apply the best available forward model—motion/warp, blur, downsampling, noise/compression approximation where appropriate—to predict the LR observations. Compare those predictions with the actual source frames.

Reprojection residual is a primary fidelity signal:

- low residual does not prove uniqueness;
- high residual may indicate either unsupported reconstruction or forward-model error;
- a candidate that cannot explain its measurements should not be treated as source-faithful without additional evidence.

## Learned components

Machine learning is permitted but does not receive automatic architectural authority.

Potentially well-bounded learned roles include:

- correspondence/offset estimation;
- visibility/confidence prediction;
- denoising;
- residual correction;
- learned proximal/unrolled solver steps;
- lightweight fusion constrained by measurement checks.

End-to-end perceptual sharpness is not proof of recovered information. Learned methods must be compared against deterministic/oracle references and evaluated for unsupported detail.

## Evaluation baseline

No single metric is sufficient. Depending on the stage, combine:

- known HR ground truth;
- PSNR/SSIM and other distortion measures;
- frequency and phase analysis;
- slanted edges, Siemens stars, zone plates, gratings, text, random high-frequency patterns, repeating textures;
- reprojection/data-consistency residuals;
- temporal stability/flicker/ghosting analysis;
- strong spatial-only baselines;
- controlled ablations/attribution;
- human visual review.

Visible artifacts may invalidate a metric-green result.

## Experimental ladder

Introduce one major unknown at a time:

1. **Known everything** — exact HR, sampling shifts, blur/downsampling, no occlusion.
2. **Estimated correspondence** — remove oracle motion while keeping degradation known.
3. **Estimated degradation** — introduce PSF/forward-model uncertainty.
4. **Dynamic scenes** — independent motion, occlusion, disocclusion, parallax.
5. **Compression and noise** — realistic codec and sensor degradation.
6. **Real video** — paired or strongest available pseudo-ground-truth evaluation.

Each stage should define controls, falsifiers, exit criteria, artifacts, and a disposition under `docs/evaluation/EVIDENCE_STANDARD.md`.

## Performance position

Oracle/reference implementations may be slow.

Production goals remain:

- **minimum:** 30 FPS real-time playback;
- **desired:** 60 FPS where achievable without compromising reconstruction fidelity;
- broad GPU relevance rather than flagship-only execution;
- vendor-neutral design where technically reasonable.

Earlier Temporal Forge work demonstrated extremely high-throughput native GPU-video engineering. General performance lessons may be inherited under `docs/principles/PERFORMANCE_INHERITANCE.md`; FSR-specific reconstruction semantics may not.

Likely production-friendly mechanism classes include cheap robust analytical fusion, very limited data-consistency/back-projection steps, and lightweight learned components, but none is selected yet.

## Still unresolved

- exact temporal-opportunity gate;
- explicit flow vs implicit/deformable/multi-hypothesis correspondence;
- calibrated confidence representation;
- correspondence-error tolerance by reconstruction family;
- adaptive temporal-memory policy;
- layered/transparent/fine-structure treatment;
- forward-model and PSF estimation;
- exact learned-component boundaries;
- best fusion-vs-selection policy;
- route from oracle-quality reconstruction to broad-GPU 30/60 FPS.

## Default project stance

Do not optimize for an impressive architecture. Optimize for discovering which information is genuinely present, which mechanism recovers it, how failure is detected, and how a proven mechanism can later be approximated or accelerated without losing source fidelity.
