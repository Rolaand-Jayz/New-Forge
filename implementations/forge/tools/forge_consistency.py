#!/usr/bin/env python3
"""F-CONSISTENCY gate (FORGE-NEW-MODEL.md §3): the torch base used in
training must equal the player's deployed base.

Usage: forge_consistency.py <clip.mp4> <capture_cell_dir> [--json out.json]
  The cell must hold 00-source.png + 01-scaler.png from the player's
  scaler arm (ANVIL_PIPELINE=scaler, lanczos3, ANVIL_SCALER_FACTOR=3) and a
  manifest.jsonl giving frame_index.

Checks, end to end from the SAME compressed bitstream the player decoded:
  A. front end : torch yuv420->linear->sRGB8   vs 00-source.png
  B. base      : torch lanczos3(linear) ->sRGB8 vs 01-scaler.png
Each reported over the whole frame and split interior / border band.

PASS thresholds — fixed 2026-11-12 BEFORE the first comparison was run:
  per check, over all channel values:  >= 99.9% within +-1 code  AND
  max |diff| <= 2 codes  AND  mean |diff| <= 0.05 codes.
(+-1 code is the honest floor: the GPU's bilinear chroma fetch uses
fixed-point weights and transcendentals are not bit-exact vs torch.)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_player_io import (default_matrix, linear_to_srgb8,  # noqa: E402
                             player_base, yuv420_to_linear)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))
from _repo_io import repo_open_w  # noqa: E402

WITHIN1_MIN, MAX_MAX, MEAN_MAX = 0.999, 2, 0.05


def decode_yuv(clip, n):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", clip],
        check=True, capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(out[0]), int(out[1])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip, "-vf", f"select=eq(n\\,{n})",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        check=True, capture_output=True).stdout
    ys, cs = w * h, (w // 2) * (h // 2)
    if len(raw) != ys + 2 * cs:
        raise RuntimeError(f"short decode: {len(raw)} bytes")
    a = np.frombuffer(raw, np.uint8)
    y = torch.from_numpy(a[:ys].reshape(h, w).copy())
    u = torch.from_numpy(a[ys:ys + cs].reshape(h // 2, w // 2).copy())
    v = torch.from_numpy(a[ys + cs:].reshape(h // 2, w // 2).copy())
    return y, u, v


def decode_ordinal(clip, pts_us):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,start_time", "-of", "csv=p=0", clip],
        check=True, capture_output=True, text=True).stdout.strip().split(",")
    num, den = (int(x) for x in out[0].split("/"))
    start = float(out[1]) if out[1] not in ("", "N/A") else 0.0
    return int(round((pts_us / 1e6 - start) * num / den))


def compare(ours, ref, border):
    d = (ours.astype(np.int16) - ref.astype(np.int16))
    ad = np.abs(d)

    def stats(x):
        return {"within1": float((x <= 1).mean()), "max": int(x.max()),
                "mean": float(x.mean()), "exact": float((x == 0).mean())}
    inner = ad[border:-border, border:-border]
    mask = np.ones(ad.shape[:2], bool)
    mask[border:-border, border:-border] = False
    res = {"all": stats(ad), "interior": stats(inner),
           "border": stats(ad[mask]), "bias": float(d.mean())}
    a = res["all"]
    res["pass"] = (a["within1"] >= WITHIN1_MIN and a["max"] <= MAX_MAX
                   and a["mean"] <= MEAN_MAX)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("cell")
    ap.add_argument("--json")
    a = ap.parse_args()

    man = [json.loads(l) for l in open(os.path.join(a.cell, "manifest.jsonl"))]
    # frame_index is a zero-epoch PTS label (round(pts*fps)), NOT a decode
    # ordinal: clips whose first pts != 0 (tos_daylight: 0.083 s) are offset.
    # Map pts into the stream's decode lattice, like verify_cell_cas gt-index.
    n = decode_ordinal(a.clip, man[0]["pts_us"])
    src = np.asarray(Image.open(os.path.join(a.cell, "00-source.png")).convert("RGB"))
    cap = np.asarray(Image.open(os.path.join(a.cell, "01-scaler.png")).convert("RGB"))
    scale = cap.shape[0] // src.shape[0]

    y, u, v = decode_yuv(a.clip, n)
    h, w = y.shape
    lin = yuv420_to_linear(y, u, v, matrix=default_matrix(w, h))
    rA = compare(linear_to_srgb8(lin).numpy(), src, 3)
    base = player_base(lin, scale)
    rB = compare(linear_to_srgb8(base).numpy(), cap, 3 * scale)

    out = {"clip": os.path.basename(a.clip), "cell": os.path.basename(a.cell.rstrip("/")),
           "frame_index": n, "scale": scale,
           "thresholds": {"within1_min": WITHIN1_MIN, "max_max": MAX_MAX,
                          "mean_max": MEAN_MAX},
           "A_frontend": rA, "B_base": rB, "pass": rA["pass"] and rB["pass"]}
    print(json.dumps(out, indent=1))
    if a.json:
        with repo_open_w(a.json) as f:
            json.dump(out, f, indent=1)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
