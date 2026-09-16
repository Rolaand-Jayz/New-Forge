# Performance Inheritance

## Purpose

This document defines the boundary between the earlier FSR-centered Temporal Forge work and this repository.

The new project is intentionally clean with respect to **reconstruction assumptions**, but it should not discard engineering knowledge already earned while building an extremely fast Linux-native GPU video path.

> **Do not inherit the old Forge's model of the reconstruction problem. Inherit proven knowledge about how to make GPU video processing fast.**

## Not inherited

The following are **not architectural authority** here:

- AMD FSR 4.1 as the reconstruction center;
- FSR-specific input contracts;
- synthetic game-renderer semantics created to satisfy those contracts;
- assumptions that motion vectors, jitter, reactive masks, exposure metadata, depth, or a particular history format are required because a game upscaler expected them;
- one-input-frame to one-output-frame topology as a requirement;
- fixed history or temporal support inherited from the prior implementation;
- quality heuristics whose only justification was compatibility with the FSR path.

Historical results may still be useful evidence, but they do not define this architecture.

## May be inherited

Engineering techniques may be reused when independently applicable and validated in the new system.

### GPU execution

- efficient Vulkan compute dispatch patterns;
- command-buffer organization;
- pipeline and descriptor reuse;
- avoidance of unnecessary synchronization;
- asynchronous execution where safe;
- minimizing CPU/GPU round trips;
- reducing dispatch overhead;
- occupancy-aware kernel design;
- workgroup and launch-shape tuning.

### Memory and data movement

- buffer/image reuse;
- persistent allocation rather than per-frame allocation;
- reduced host/device copies;
- zero-copy or low-copy paths where practical;
- compact intermediate representations;
- tiling for locality or memory-pressure control;
- explicit VRAM budgeting;
- reusable frame/resource pools.

### Precision

- FP16 where validated against reference quality;
- INT8 or other quantized execution for learned components where validated;
- mixed precision with bounded error;
- retained high-precision reference paths for comparison.

Precision is an optimization choice, not a reconstruction assumption.

### Pipeline behavior

- decode / process / display pipelining;
- buffering to hide latency;
- asynchronous stages;
- bounded startup buffering when future observations prove useful;
- separation of algorithm timing from presentation timing.

### Profiling and reproducibility

- per-stage GPU timing;
- end-to-end frame-time accounting;
- separating algorithm cost from synchronization or presentation cost;
- representative-resolution testing;
- hardware/driver/configuration capture;
- binary and code identity;
- checking whether quality survives optimization.

## Carry-forward test

A technique from the earlier project should be reused only when:

1. it solves a general GPU/video-processing problem rather than an FSR-specific semantic problem;
2. it does not force a reconstruction assumption into the new design;
3. it can be isolated and measured;
4. it preserves the reference reconstruction within an explicit tolerance;
5. it improves a relevant bottleneck.

## Reference-first optimization

For each validated reconstruction stage:

1. keep a transparent oracle/reference implementation;
2. profile it;
3. identify the dominant cost;
4. replace or approximate one expensive operation at a time;
5. compare the optimized result against the reference;
6. reject optimizations that materially degrade source-supported reconstruction;
7. record the trade-off.

## Historical source

Earlier performance work remains in:

`Rolaand-Jayz/Temporal-Forge-Player`

That repository is a historical evidence source, not an active architecture specification for this project.
