# FORGE implementation (from ANVIL `41ba911`)

This directory holds the code behind the results record
[`docs/results/2026-11-17-forge-protected-scene-confirm.md`](../../docs/results/2026-11-17-forge-protected-scene-confirm.md).

**Status: implementation evidence, not the selected New Forge architecture.** FORGE was built inside ANVIL, a private Linux/Vulkan video player. No decision record adopts it (see [`AGENTS.md`](../../AGENTS.md) and [`docs/decisions/`](../../docs/decisions/README.md)). It is published so the recorded numbers can be checked against the code that produced them.

## What FORGE is

- `output = lanczos3(LR) + delta(LR)`: a lanczos3 3× base plus a small convolutional correction network (`ForgeDeltaNet`, 32 channels, 12 depthwise + pointwise blocks, 18,299 parameters).
- An evidence-gated temporal path: noise-gated history fusion, noise-gated back-projection, and output-resolution (4K) accumulation that contributes only sub-pixel phases the current frame cannot see.
- **The shipped weights are hand-constructed by a script, not trained.** `tools/forge_construct.py` writes every weight (a noise-gated denoiser plus undershoot-only lanczos3 de-ringing). A trainer exists (`tools/train_forge.py`) but did not produce these weights.
- No AMD/FSR weights, shaders, or outputs were used as initialization, teacher, or data. The FSR 4 material in ANVIL is deliberately not included here (see [`NOTICE.md`](NOTICE.md)).

## What was measured (summary of the record; read it for the limits)

On three protected scenes (`sintel_rooftop`, `tos_daylight`, `tos_debris`; frames 72/120/168; 1280×720 → 3840×2160), against lanczos3 alone, PSNR vs the 2160p lossless master:

| Arm | dB over lanczos3 |
|---|---|
| Clean (CRF 10) | +0.585 |
| CRF 23, clips re-cut from the master | +0.414 |
| CRF 23, original corpus clips (1.3–1.7 px off the master) | −0.030 |
| Noisy | +4.858 (mostly denoising, not recovered detail) |

- The temporal-stability check (M2 edge flicker) **fails on noisy input** (0.0792 vs lanczos3 0.0643). It passes on the other arms.
- Speed: 9.77 ms median for the whole FORGE stage per 4K frame on an AMD RX 7900 GRE (RADV), from GPU timestamps in the player. **Real 4K60 playback has not been measured.**
- Three scenes and nine frames per arm. Generalization is not established.

## Layout

```text
implementations/forge/
├── shaders/                 Vulkan GLSL compute shaders of the GPU stage
│   ├── forge_stem|block|tail.comp      the correction network
│   ├── forge_pyr|motion|gate|fuse.comp temporal path: pyramid, motion, gating, history fusion
│   ├── forge_bp.comp                   noise-gated back-projection
│   ├── forge_acc.comp                  4K sub-pixel accumulation
│   └── chroma_guided.comp              guided chroma front end (runs in the player before FORGE)
├── src/pipeline/ForgeStage.{hpp,cpp}   host-side Vulkan stage (weights, pipelines, dispatch, timing)
├── assets/forge/            default weights merged-v2 (+ merged-v1 for A/B) and PROVENANCE.md
├── tools/
│   ├── forge_model.py, forge_construct.py, lab/forge_construct_merged_v1.py, forge_export.py
│   │                        model definition, weight constructor, export to the .forge format
│   ├── forge_lanczos.py, forge_player_io.py, forge_accum.py, forge_temporal.py, forge_consistency.py
│   │                        PyTorch replica of the player's FORGE path (what the gate measures)
│   ├── forge_shader_parity.py, forge_temporal_parity.py
│   │                        replica vs GPU-capture parity checks (need player captures)
│   ├── forge_fquality.py    F-QUALITY gate definition: cells, ground truth, PSNR
│   ├── forge_tdev.py, forge_tsearch.py, forge_tq.sh
│   │                        tuning harness; runs on a non-gate dev set only
│   ├── forge_data.py, forge_fetch.py, forge/sources-v1.json, forge/build_loop.sh
│   │                        training-data pipeline (CC BY 4.0 / CC BY 3.0 sources, by URL)
│   ├── train_forge.py       trainer (not used for the shipped weights)
│   ├── metrics.py, lab/ref_transfer.py, lab/_repo_io.py   metric code
│   └── lab/cut-hq-corpus.sh re-cuts the CRF 10 test clips from the 2160p masters
└── gate/                    the scripts that produced the record's numbers
    ├── gate_confirm.py      protected cells, four configurations, M2 flicker
    ├── scene_confirm.py     same protocol on chosen clips (used for the aligned CRF 23 arm)
    ├── spatial_protected.py spatial-only model on the protected cells
    ├── history_use.py       how often temporal history is used, per arm
    ├── cave_confirm.py      extra scene (Sintel cave), not one of the three protected scenes
    ├── clip_align.py        sub-pixel geometry check of each clip against the master
    ├── gate_confirm_m2.py   the M2 flicker metric, shared by the scripts
    ├── INPUTS.sha256        SHA-256 of every input file the original runs read
    └── results-2026-11-17/  the original JSON and log outputs, unchanged
```

**Not buildable on its own:** the C++ stage and shaders compile only inside the private ANVIL player, which provides the `Stage`, `Pipeline`, and `gfx::Gfx` interfaces and the shader embedding. They are published for reading and review. The Python replica and the gate scripts run on their own.

## Re-running the gate (CPU is enough)

Requirements: Python 3, PyTorch, NumPy, Pillow, OpenCV (`cv2`), and `ffmpeg`/`ffprobe` on `PATH`. The original runs used ROCm on an RX 7900 GRE. `FORGE_DEVICE` picks the PyTorch device (default `cpu`).

All paths below are relative to `implementations/forge/`. Inputs and outputs go in `research/`, `assets/testmedia/`, and `.scratch/`, which are git-ignored.

1. **Build the weights** (takes seconds), and check they match the shipped file:

   ```bash
   mkdir -p .scratch/forge/constructed .scratch/forge/gate-confirm
   PYTHONPATH=tools:tools/lab python3 tools/forge_construct.py .scratch/forge/constructed/merged-v2.pt --mode merged
   python3 tools/forge_export.py .scratch/forge/constructed/merged-v2.pt .scratch/forge/constructed/merged-v2.forge
   sha256sum .scratch/forge/constructed/merged-v2.forge assets/forge/merged-v2.forge
   ```

2. **Get the test clips.** Clone [`Temporal-Forge-Player`](https://github.com/Rolaand-Jayz/Temporal-Forge-Player) into `research/Temporal-Forge-Player` and run `benchmarks/video_corpus/prepare_corpus.sh`. It downloads the Blender open-movie 4K sources and writes the 2160p lossless masters and the CRF 23 clips. Then run `tools/lab/cut-hq-corpus.sh` for the CRF 10 clips.
   - **Aligned CRF 23 clips** (they sit on the master grid): made from the 2160p master (the scaling command was not recorded; `gate/clip_align.py` measures them within 0.03 px of the master), encoded with libx264 `-preset medium -crf 23 -g 48`. The encoder settings stored in the files confirm `crf=23 keyint=48 bframes=3`. Put them in `.scratch/forge/aligned-crf23/`.
   - **Noisy twins:** the CRF 10 clip with ffmpeg `noise=alls=20:allf=t+u` added (as recorded in ANVIL's notes), re-encoded with libx264. The stored settings say `crf=18 keyint=250 bframes=3`. Put them in `assets/testmedia/`.
   - The full commands for these two clip sets were not saved, so their bytes can't be reproduced exactly. `gate/INPUTS.sha256` lists the hash of every input file the original runs read. Matching hashes mean identical inputs. Regenerated clips will give close but not bit-identical numbers.
   - `prepare_corpus.sh` itself depends on ffmpeg and x264 versions, so a fresh corpus may also differ at the byte level from the files in `INPUTS.sha256`.

3. **Run** (from `implementations/forge/`):

   ```bash
   python3 gate/gate_confirm.py
   ONLY=crf23a SCENE=sintel_rooftop OUT=sintel_rooftop_crf23a python3 gate/scene_confirm.py
   ```

   Run the second line for `tos_daylight` and `tos_debris` as well. Outputs land in `.scratch/forge/gate-confirm/`. Compare them with `gate/results-2026-11-17/`.

### Re-run from this copy (2026-11-21)

Before publishing, the steps above were run from this directory on the CPU (PyTorch 2.15.0), with weights rebuilt by step 1 and the original input files (all 18 inputs these runs read match `gate/INPUTS.sha256`).

- The rebuilt `merged-v2.forge` is byte-identical to the shipped file.
- `gate_confirm.py` and the three aligned CRF 23 `scene_confirm.py` runs reproduce every number in `gate/results-2026-11-17/`. The largest difference in any field is 0.0002 dB, from CPU vs GPU floating-point rounding. The arm means are unchanged: clean +0.585, aligned CRF 23 +0.414, noisy +4.858, and the M2 result (fails on noisy only).
- `spatial_protected.py`, `history_use.py`, `cave_confirm.py` and the `tos_daylight` extra-arm run were not repeated.

This shows the published code produces the published numbers from the same inputs. It is not an independent replication with freshly built clips.

## Changes from the ANVIL source

All files are byte-identical to ANVIL commit `41ba911` (2026-11-17), except the gate scripts. Those were untracked in ANVIL (`.scratch/forge/gate-confirm/`) and got two edits:

- the repository root is found from the script's own location instead of a hard-coded local path;
- the PyTorch device comes from `FORGE_DEVICE` (default `cpu`) instead of a hard-coded `cuda`.

## License

MIT, under the repository [`LICENSE`](../../LICENSE). See [`NOTICE.md`](NOTICE.md) for what was left out and for the test-scene credits.
