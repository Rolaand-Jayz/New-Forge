#!/usr/bin/env python3
"""FORGE temporal parity gate (front end + temporal path + back-projection + 4K accumulation): the player's forge stage with the temporal
path (and ANVIL_FORGE_BP) vs forge_temporal.Temporal run over the same frames.

  forge_temporal_parity.py <clip> <cell_dir> --weights W.forge --log player.log [--bp 1|0] [--json out.json]
  (FORGE_ACC=0 in the environment scores a capture taken with ANVIL_FORGE_ACC=0)

The cell holds 01-forge.png + manifest.jsonl (cell.sh forge capture). The replica
streams frames 0..n of the clip through the player front end (player_decode) and
Temporal.step (FORGE_BP=<bp>), then compares sRGB8 whole frame with the capture.
Same PASS thresholds as forge_shader_parity (>= 99.9% within +-1, max <= 2,
mean <= 0.05)."""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO + "/tools")
ap = argparse.ArgumentParser()
ap.add_argument("clip"); ap.add_argument("cell")
ap.add_argument("--weights", required=True); ap.add_argument("--bp", default="1")
ap.add_argument("--json")
ap.add_argument("--log", required=True, help="player log with ANVIL_FORGE_T_LOG=1 (history resets)")
a = ap.parse_args()
os.environ["FORGE_BP"] = a.bp
import forge_data as D, forge_temporal as T
from forge_consistency import compare, decode_ordinal
from forge_export import read_forge
from forge_model import ForgeDeltaNet
from forge_player_io import default_matrix, linear_to_srgb8, player_decode

dev = "cuda" if torch.cuda.is_available() else "cpu"
sd, _ = read_forge(a.weights)
net = ForgeDeltaNet().to(dev).eval(); net.load_state_dict(sd)
blk, slot, sc = open(a.weights + ".gate").read().split()
gate = (int(blk), int(slot), float(sc))
man = [json.loads(l) for l in open(os.path.join(a.cell, "manifest.jsonl"))]
n = decode_ordinal(a.clip, man[0]["pts_us"])
cap = np.asarray(Image.open(os.path.join(a.cell, "01-forge.png")).convert("RGB"))
H, W = cap.shape[0] // 3, cap.shape[1] // 3
import re, collections
ev = [(int(p), int(h)) for p, h in re.findall(r"temporal frame \d+ pts_ticks=(-?\d+) history=(\d)", open(a.log).read())]
pts = sorted({p for p, _ in ev})
tpf = collections.Counter(b - a for a, b in zip(pts, pts[1:])).most_common(1)[0][0]
resets = sorted({round(p / tpf) for p, h in ev if h == 0})
print("ticks/frame", tpf, "history resets at ordinals", resets, file=sys.stderr)
tp = T.Temporal(net, gate); tp.nclamp, tp.tdead = 0.0, 0.0
t = lambda x: torch.from_numpy(np.ascontiguousarray(x).astype(np.int32))
for k, (y, u, v) in enumerate(D._yuv_stream(a.clip, W, H)):
    if k in resets: tp.reset()
    lin = player_decode(t(y), t(u), t(v), matrix=default_matrix(W, H)).to(dev)
    out = tp.step(lin)
    if k == n: break
ref = linear_to_srgb8(out.cpu()).numpy()
r = compare(ref, cap, 9)
res = {"clip": os.path.basename(a.clip), "frame": n, "bp": a.bp, "fused_frac": tp.frac, "parity": r,
       "resets": resets, "pass": bool(r["pass"])}
print(json.dumps(res, indent=1))
if a.json: json.dump(res, open(a.json, "w"), indent=1)
