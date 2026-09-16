# Architecture Neutrality

## Purpose

The project exists to discover the strongest evidence-supported approach to source-faithful temporal video reconstruction. It must not become a search for ways to justify a preselected architecture.

## Non-contamination rule

Do not assume the final system requires:

- AMD FSR, NVIDIA DLSS, Intel XeSS, or another game upscaler;
- optical flow or motion vectors;
- synthetic jitter;
- fixed temporal windows;
- recurrent state;
- causal-only or bidirectional-only processing;
- explicit frame warping;
- deformable convolution;
- attention or transformers;
- neural networks;
- deterministic-only reconstruction;
- depth, segmentation, reactive masks, or renderer metadata;
- any specific GPU vendor or API.

All of these may be studied. None has architectural authority until evidence earns it.

## Problem-first framing

The governing question is:

> What source-supported information exists in the observations, and what mechanism can recover it faithfully?

Do not instead ask:

> How can technology X be made to work for this project?

## Promotion rule

A major mechanism should not become architectural unless we can state:

1. the problem it solves;
2. evidence that the problem materially limits reconstruction;
3. evidence that the mechanism improves the target outcome;
4. known failure modes;
5. the smallest falsifying experiment;
6. performance implications;
7. alternatives that were considered.

## Negative evidence

A mechanism may be rejected in tested scope without claiming that it is universally useless. Preserve the exact scope of negative conclusions.

## Separation from earlier Forge work

The earlier FSR-centered Temporal Forge project is historical evidence. Its reconstruction assumptions do not transfer automatically.

General performance engineering may transfer only under [`PERFORMANCE_INHERITANCE.md`](PERFORMANCE_INHERITANCE.md).
