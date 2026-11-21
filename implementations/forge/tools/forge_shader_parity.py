#!/usr/bin/env python3
"""FORGE shader parity gate: the player's forge stage (shaders/forge_*.comp)
must compute the same image as the torch model for the SAME weights.

Usage:
  forge_shader_parity.py <clip.mp4> <cell_dir> --weights W.forge --log player.log
                         [--vs-capture other/01-scaler.png] [--json out.json]
  The cell holds 00-source.png + 01-forge.png + manifest.jsonl from
  ANVIL_PIPELINE=forge with ANVIL_FORGE_WEIGHTS=W.forge. --log must contain
  the player's tell-line "forge: weights loaded ... sha256=<hex16>", which is
  checked against W.forge (a missing/mismatched tell-line FAILS the cell).

Torch replica, end to end from the SAME compressed bitstream the player
decoded (forge_consistency front end, pts-mapped ordinal):
  lin   = yuv420_to_linear(...)                 (fp16-rounded working image)
  delta = ForgeDeltaNet(srgb_encode(lin) - 0.5) (weights read from W.forge)
  out   = clamp(player_base(lin, 3) + delta, 0, 1) -> linear_to_srgb8
compared with 01-forge.png over the whole frame (also split interior/border).
Also reported: the visible size of the delta (|srgb8(out) - srgb8(base)| in
codes) so a parity pass can't be an accidental zero-delta pass.

--vs-capture: additionally compare 01-forge.png against another capture of
the same frame (zero-tail case: forge must equal the scaler lanczos3 arm).

PASS thresholds — fixed 2026-11-12 BEFORE the first comparison was run
(same as F-CONSISTENCY), whole frame, all channel values:
  >= 99.9% within +-1 code AND max |diff| <= 2 codes AND mean |diff| <= 0.05.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_consistency import compare, decode_ordinal, decode_yuv  # noqa: E402
from forge_export import read_forge  # noqa: E402
from forge_model import SCALE, ForgeDeltaNet  # noqa: E402
from forge_player_io import (default_matrix, linear_to_srgb8,  # noqa: E402
                             forge_base, player_base, player_decode, srgb_encode, yuv420_to_linear)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))
from _repo_io import repo_open_w  # noqa: E402

TELL = re.compile(r"forge: weights loaded (\S+) sha256=([0-9a-f]{16}) params=(\d+)")


def tell_line(log_path, weights_sha):
    with open(os.path.realpath(log_path), errors="replace") as f:
        hits = TELL.findall(f.read())
    if not hits:
        return {"ok": False, "why": "no forge tell-line in log"}
    path, sha, n = hits[-1]
    return {"ok": sha == weights_sha[:16], "path": path, "sha256_16": sha,
            "params": int(n), "want_sha256_16": weights_sha[:16]}


@torch.no_grad()
def replica(clip, n, sd):
    y, u, v = decode_yuv(clip, n)
    h, w = y.shape
    lin = player_decode(y, u, v, matrix=default_matrix(w, h))     # [H,W,3]
    net = ForgeDeltaNet()
    net.load_state_dict(sd)
    net.eval()
    inp = (srgb_encode(lin) - 0.5).permute(2, 0, 1)[None]
    delta = net(inp)[0].permute(1, 2, 0)                              # [3H,3W,3]
    base = forge_base(lin, SCALE)
    out = (base + delta).clamp(0.0, 1.0)
    return lin, base, delta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("cell")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--vs-capture")
    ap.add_argument("--json")
    a = ap.parse_args()

    sd, wsha = read_forge(a.weights)
    tell = tell_line(a.log, wsha)
    with open(os.path.realpath(os.path.join(a.cell, "manifest.jsonl"))) as f:
        man = [json.loads(line) for line in f]
    n = decode_ordinal(a.clip, man[0]["pts_us"])
    caps = sorted(glob.glob(os.path.join(a.cell, "01-*.png")))
    if len(caps) != 1 or not caps[0].endswith("01-forge.png"):
        raise SystemExit(f"expected exactly 01-forge.png in {a.cell}, got {caps}")
    cap = np.asarray(Image.open(caps[0]).convert("RGB"))

    lin, base, delta, out = replica(a.clip, n, sd)
    ref = linear_to_srgb8(out).numpy()
    base8 = linear_to_srgb8(base).numpy().astype(np.int16)
    if ref.shape != cap.shape:
        raise SystemExit(f"shape mismatch: replica {ref.shape} vs capture {cap.shape}")
    r = compare(ref, cap, 3 * SCALE)
    vis = np.abs(ref.astype(np.int16) - base8)
    capvis = np.abs(cap.astype(np.int16) - base8)
    res = {"clip": os.path.basename(a.clip), "cell": os.path.basename(a.cell.rstrip("/")),
           "frame_ordinal": n, "weights": a.weights, "weights_sha256": wsha,
           "tell_line": tell,
           "delta_linear_mean_abs": float(delta.abs().mean()),
           "delta_linear_max_abs": float(delta.abs().max()),
           "delta_visible_codes": {"mean": float(vis.mean()), "max": int(vis.max()),
                                   "frac_nonzero": float((vis > 0).mean())},
           "capture_vs_base_codes": {"mean": float(capvis.mean()), "max": int(capvis.max())},
           "parity": r}
    ok = r["pass"] and tell["ok"]
    if a.vs_capture:
        other = np.asarray(Image.open(a.vs_capture).convert("RGB"))
        rv = compare(cap, other, 3 * SCALE)
        res["vs_capture"] = {"path": a.vs_capture, **rv}
        ok = ok and rv["all"]["max"] <= 1
    res["pass"] = bool(ok)
    print(json.dumps(res, indent=1))
    if a.json:
        with repo_open_w(a.json) as f:
            json.dump(res, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
