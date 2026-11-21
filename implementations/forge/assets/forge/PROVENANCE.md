# assets/forge — FORGE default weights

`merged-v2.forge` (+ `merged-v2.forge.gate`) is the player's default FORGE
model (since 2026-11-15; `merged-v1.forge` is kept for A/B). It is **hand-constructed, not trained**: every weight is written by
`tools/forge_construct.py --mode merged` (noise-gated denoiser + undershoot-only
lanczos3 de-ringing on four edge phases, ForgeDeltaNet C=32 / 12 blocks / 3x).
No AMD/FSR weights, shaders or outputs were used as init, teacher or data.

The `.gate` sidecar (`<block> <slot> <scale>`) names the feature channel that
holds the net's noise map; the temporal path reads it after that block and
uses it to gate the next frame's history fusion.

Regenerate:

    PYTHONPATH=tools:tools/lab python3 tools/forge_construct.py .scratch/forge/constructed/merged-v2.pt --mode merged
    python3 tools/forge_export.py .scratch/forge/constructed/merged-v2.pt assets/forge/merged-v2.forge
    cp .scratch/forge/constructed/merged-v2.forge.gate assets/forge/

merged-v2 (2026-11-15) changes to the merged constructor, found by the
weight-worker round on the non-gate temporal dev set (tools/forge_tdev.py;
J +0.96 -> +1.85): isotropic gate gradient veto (|Gx| and |Gy|, rho =
rho_y = 0.75); chroma gate evidence through the Immerkaer HH operator
instead of HF; capped luma evidence (hcap 0.5); per-colour second
differences (freed by merging the knot-1 Q/SK pairs around a constant
channel); de-ringing off where the gate is open (dgT 50), relative deadband
dr 0.001, blue de-ringing at half weight (dgB 0.5); wC 0.65, kY 5.0. Same
tensors and shapes as v1 (speed unchanged); gate moves to block 4 slot 15.
merged-v1 was built by the pre-v2 constructor, kept as
tools/lab/forge_construct_merged_v1.py (same CLI); its gate is block 4 slot 17.

Override at runtime with `ANVIL_FORGE_WEIGHTS=<file.forge>` (the `.gate`
sidecar is looked up next to it), or drop a file at
`~/.local/share/anvil/forge/default.forge`.
