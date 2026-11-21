#!/usr/bin/env python3
"""F-QUALITY offline verdict (FORGE-NEW-MODEL.md §3) — forge vs lanczos3.

Same 24 cells as the C4 campaign (clean hq_crf10 / crf23 medium / noisy
twins x scenes x frames 72/120/168), same whole-frame RGB PSNR on sRGB8,
with the corrected ruler (§6): player-exact GT (gt_linear_from_yuv on the
2160p master) at pts-mapped ordinals. Both arms computed by the
consistency-gated torch replica of the player (front end + lanczos3 base,
max 1 code from player captures). The in-player re-confirmation follows
the Vulkan shader bring-up.

Gates applied VERBATIM from §3 (pre-recorded 2026-11-07):
  KILLED if mean gain on crf23+noisy < +0.05 dB
  KILLED if clean CRF10 regresses > 0.05 dB mean
  mic-drop tier: >= +0.3 dB on crf23+noisy with clean intact

Usage: forge_fquality.py <ckpt.pt> [--json out.json]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))
import forge_data as D  # noqa: E402
from _repo_io import repo_open_w  # noqa: E402
from forge_consistency import decode_yuv  # noqa: E402
from forge_model import ForgeDeltaNet  # noqa: E402
from forge_player_io import (linear_to_srgb8, player_base,  # noqa: E402
                             srgb_encode, yuv420_to_linear)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = "research/Temporal-Forge-Player/benchmarks/video_corpus/references/"
C = "research/Temporal-Forge-Player/benchmarks/video_corpus/clips/"
SCENES = ["sintel_rooftop", "tos_daylight", "tos_debris"]
FRAMES = [72, 120, 168]
CLIPS = {"clean": lambda s: C + f"{s}_1280x720_hq_crf10.mp4",
         "crf23": lambda s: C + f"{s}_1280x720_medium_crf23.mp4",
         "noisy": lambda s: f"assets/testmedia/{s}_1280x720_hq_crf10_noisy.mp4"}
CACHE = os.path.join(REPO, ".scratch/forge/fq-gt")


def cells():
    for arm in ("clean", "crf23", "noisy"):
        for s in SCENES:
            if os.path.exists(os.path.join(REPO, CLIPS[arm](s))):
                for f in FRAMES:
                    yield arm, s, f


def _start(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=start_time", "-of", "csv=p=0", p],
                       capture_output=True, text=True).stdout.strip()
    return float(o) if o not in ("", "N/A") else 0.0


def gt_frame(scene, n):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{scene}-{n}.npy")
    if os.path.exists(p):
        return np.load(p)
    src = {"path": os.path.join(REPO, R + f"{scene}_2160p_lossless.mkv")}
    it = D._gt_stream(src, stride=n, offset=0) if n > 0 else D._gt_stream(src)
    for k, pl in enumerate(it):
        if k == (1 if n > 0 else 0):
            g = linear_to_srgb8(D._gt_lin(pl)).numpy()
            np.save(p, g)
            return g


def psnr(a, b):
    return float(10 * np.log10(255.0 ** 2 / ((a.astype(np.float64) - b) ** 2).mean()))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ForgeDeltaNet().to(dev).eval()
    st = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    net.load_state_dict(st["net"])
    rows = []
    for arm, s, f in cells():
        clip = os.path.join(REPO, CLIPS[arm](s))
        n = f - round(_start(clip) * 24)     # displayed decode ordinal == master ordinal
        y, u, v = decode_yuv(clip, n)
        lin = yuv420_to_linear(y, u, v).to(dev)
        base = player_base(lin, 3)                                  # [H,W,3] linear
        delta = net((srgb_encode(lin) - 0.5).permute(2, 0, 1)[None])[0].permute(1, 2, 0)
        forge = (base + delta).clamp(0.0, 1.0)
        g = gt_frame(s, n).astype(np.float64)
        pb = psnr(linear_to_srgb8(base.cpu()).numpy(), g)
        pf = psnr(linear_to_srgb8(forge.cpu()).numpy(), g)
        rows.append({"arm": arm, "scene": s, "frame": f, "ordinal": n,
                     "base": pb, "forge": pf, "d": pf - pb})
        print(f"{arm:6s} {s:15s} f{f:<3d} base {pb:6.3f}  forge {pf:6.3f}  d {pf - pb:+.4f}",
              flush=True)
    m = lambda arms: float(np.mean([r["d"] for r in rows if r["arm"] in arms]))
    clean, cn = m({"clean"}), m({"crf23", "noisy"})
    k1 = cn < 0.05
    k2 = clean < -0.05
    verdict = "KILLED" if (k1 or k2) else ("MIC-DROP" if cn >= 0.3 else "PASS")
    summary = {"ckpt": a.ckpt, "step": st.get("step"), "clean_mean_d": clean,
               "crf23_mean_d": m({"crf23"}), "noisy_mean_d": m({"noisy"}),
               "gate_set_mean_d": cn, "kill_gain": k1, "kill_clean": k2,
               "verdict": verdict, "cells": rows}
    print(f"clean mean d {clean:+.4f} | crf23 {summary['crf23_mean_d']:+.4f} | noisy "
          f"{summary['noisy_mean_d']:+.4f} | GATE SET (crf23+noisy) {cn:+.4f} "
          f"(kill < +0.05, mic-drop >= +0.3) | clean regress-kill {k2} -> {verdict}")
    if a.json:
        with repo_open_w(a.json) as fo:
            json.dump(summary, fo, indent=1)


if __name__ == "__main__":
    main()
