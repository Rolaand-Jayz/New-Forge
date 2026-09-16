# Evidence Standard

## Purpose

This project must be able to distinguish a measured reconstruction gain from an attractive but unsupported output.

## Evidence labels

Use these labels explicitly when the distinction matters:

- **Measured fact** — directly produced by a controlled project measurement.
- **Documented result** — reported by a cited external source.
- **Observation** — directly seen but not yet established as causal.
- **Engineering inference** — reasoned consequence of available evidence.
- **Hypothesis** — testable claim not yet established.
- **Speculation** — plausible but weakly grounded idea.
- **Unresolved** — current evidence is insufficient.

## Minimum evidence chain for promoted mechanisms

A promoted result should identify:

1. claim being tested;
2. code/binary identity;
3. input and ground truth/reference identity;
4. controlled variable;
5. controls/baselines;
6. configuration;
7. output artifacts and logs;
8. quality metrics;
9. temporal metrics where relevant;
10. competing explanations;
11. invalid or excluded evidence;
12. disposition.

## Recommended dispositions

- **SUPPORTED**
- **SUPPORTED WITH QUALIFICATION**
- **NOT SUPPORTED**
- **CONTRADICTED IN TESTED SCOPE**
- **UNRESOLVED**
- **INVALIDATED EVIDENCE**
- **PROJECT DECISION**

## Source-faithfulness gates

Conventional metrics are not sufficient on their own. Depending on the experiment, validation should include:

- PSNR/SSIM and other distortion metrics;
- frequency-domain analysis;
- known-frequency or phase-sensitive targets;
- edge/text/fine-structure crops;
- reprojection/data-consistency residuals;
- temporal stability/flicker analysis;
- comparison against strong spatial baselines;
- human visual review.

## Reprojection

Where a forward model is available, take the reconstructed HR candidate and regenerate predicted LR observations. Compare those predictions to the actual source frames.

A low residual does not prove uniqueness, and a high residual may also indicate a wrong forward model, but reprojection is a primary test of whether the reconstruction is compatible with the measurements.

## Human review

Human review may reopen a metric-green result when visible artifacts, false textures, lattice patterns, ghosting, temporal instability, or other quality failures remain.

## Negative evidence

Preserve failed experiments. Do not rewrite history after an architecture changes.

## Performance evidence

Do not conflate algorithmic quality with implementation speed.

Performance claims should specify:

- resolution and frame format;
- temporal support/frame count;
- hardware and driver;
- precision;
- measured stage timing;
- synchronization/presentation inclusion;
- quality delta versus reference.

The eventual target is at least 30 FPS and ideally 60 FPS across a broad GPU range, but oracle experiments may intentionally ignore this constraint.
