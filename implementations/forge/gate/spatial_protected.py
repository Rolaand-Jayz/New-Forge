#!/usr/bin/env python3
"""Spatial-only merged-v2 (old front end, no temporal) on the 24 protected cells, vs the ruler
base; + M2 on FRAME/FRAME+1 — the first step of the progression for gate_confirm.py."""
import os, sys, json, numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO + "/tools"); sys.path.insert(0, REPO + "/tools/lab"); os.chdir(REPO)
os.environ["FORGE_BASE"] = "linear"
import forge_fquality as Q, forge_temporal as FT
from forge_consistency import decode_yuv
from forge_player_io import linear_to_srgb8, player_base, yuv420_to_linear
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_confirm_m2 import m2
net, gate = FT.load(REPO + "/.scratch/forge/constructed/merged-v2.pt", os.environ.get("FORGE_DEVICE", "cpu"))
t = FT.Temporal(net, gate)
s8 = lambda x: linear_to_srgb8(x.clamp(0, 1).cpu()).numpy()
rows = []
for arm, s, f in Q.cells():
    clip = Q.CLIPS[arm](s); n = f - round(Q._start(clip) * 24)
    o = []
    for fr in (n, n + 1):
        y, u, v = decode_yuv(clip, fr)
        T = lambda x: torch.as_tensor(np.asarray(x)).to(torch.int32)
        o.append(s8(t.spatial(yuv420_to_linear(T(y), T(u), T(v), chroma="bilinear").cuda())[0]))
        if fr == n: b = s8(player_base(yuv420_to_linear(T(y), T(u), T(v), chroma="bilinear").cuda(), 3))
    g = Q.gt_frame(s, n).astype(np.float64)
    r = dict(arm=arm, scene=s, frame=f, spatial=Q.psnr(o[0], g) - Q.psnr(b, g), m2_spatial=m2(o[0], o[1])); rows.append(r)
    print(f"{arm:6s} {s:15s} f{f:<3d} spatial {r['spatial']:+.3f} M2 {r['m2_spatial']:.4f}", flush=True)
for a in ("clean", "crf23", "noisy"):
    print(f"spatial {a}: {np.mean([r['spatial'] for r in rows if r['arm'] == a]):+.4f}  M2 {np.mean([r['m2_spatial'] for r in rows if r['arm'] == a]):.4f}")
print(f"spatial M2 mean {np.mean([r['m2_spatial'] for r in rows]):.4f}")
json.dump(rows, open(REPO + "/.scratch/forge/gate-confirm/spatial_protected.json", "w"), indent=1)
