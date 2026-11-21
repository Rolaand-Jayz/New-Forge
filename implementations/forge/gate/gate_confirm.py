#!/usr/bin/env python3
"""Protected-scene confirm (F-QUALITY cells, never tuned on) + F-TEMPORAL M2 edge flicker,
for the player's FORGE path as built, configuration by configuration:
  old : 2026-11-15 player (bilinear chroma, linear base, no BP, no accumulation)
  fe  : + code-domain base + guided chroma
  bp  : + noise-gated back-projection
  acc : + 4K accumulation (the current default)
Per cell: the player capture's warm-up (history from FRAME-18), PSNR at FRAME vs the
player-exact GT, minus the fixed ruler base (lanczos3 of the bilinear decode, linear light).
M2 (Q-UPSCALE-AA-PROTOCOL §7.3, verbatim: tools/metrics decode + luma_np; top-decile
gradient mask on frame N; mean |Y(N+1) - Y(N)| on the mask) on FRAME / FRAME+1, each
configuration vs lanczos3 alone on the same pair."""
import os, sys, json, tempfile, numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO + "/tools"); sys.path.insert(0, REPO + "/tools/lab")
os.chdir(REPO)
import forge_fquality as Q, forge_temporal as FT
from forge_consistency import decode_yuv
from forge_player_io import linear_to_srgb8, player_base, yuv420_to_linear
from metrics import decode as mdecode, luma_np
from PIL import Image
dev = os.environ.get("FORGE_DEVICE", "cpu")
net, gate = FT.load(REPO + "/.scratch/forge/constructed/merged-v2.pt", dev)
CFG = {"old": dict(chroma="bilinear", base="linear", bp=0.0, acc=False),
       "fe": dict(chroma="guided", base="code", bp=0.0, acc=False),
       "bp": dict(chroma="guided", base="code", bp=1.0, acc=False),
       "acc": dict(chroma="guided", base="code", bp=1.0, acc=True)}
TMP = tempfile.mkdtemp(dir=REPO + "/.scratch/forge/gate-confirm")
def m2(a8, b8):
    pa, pb = os.path.join(TMP, "a.png"), os.path.join(TMP, "b.png")
    Image.fromarray(a8).save(pa); Image.fromarray(b8).save(pb)
    y0, y1 = luma_np(mdecode(pa)), luma_np(mdecode(pb))
    gy = np.zeros_like(y0); gx = np.zeros_like(y0)
    gy[1:-1, :] = (y0[2:, :] - y0[:-2, :]) * 0.5; gx[:, 1:-1] = (y0[:, 2:] - y0[:, :-2]) * 0.5
    g = np.hypot(gx, gy); g[:2, :] = g[-2:, :] = 0; g[:, :2] = g[:, -2:] = 0
    mask = g >= np.quantile(g, 0.9)
    return float(np.abs(y1 - y0)[mask].mean())
s8 = lambda x: linear_to_srgb8(x.clamp(0, 1).cpu()).numpy()
rows = []
for arm, s, f in Q.cells():
    clip = Q.CLIPS[arm](s); n = f - round(Q._start(clip) * 24)
    ts = {}
    for k, c in CFG.items():
        t = FT.Temporal(net, gate); t.nclamp, t.tdead, t.bp, t.acc = 0.0, 0.0, c["bp"], c["acc"]; ts[k] = t
    outs = {k: [] for k in CFG}; base = []
    for fr in range(n - 18, n + 2):
        y, u, v = decode_yuv(clip, fr)
        T = lambda x: torch.as_tensor(np.asarray(x)).to(torch.int32)
        for k, c in CFG.items():
            os.environ["FORGE_BASE"] = c["base"]
            lin = yuv420_to_linear(T(y), T(u), T(v), chroma=c["chroma"]).to(dev)
            o = ts[k].step(lin)
            if fr >= n: outs[k].append(s8(o))
        if fr >= n:
            base.append(s8(player_base(yuv420_to_linear(T(y), T(u), T(v), chroma="bilinear").to(dev), 3)))
    g = Q.gt_frame(s, n).astype(np.float64)
    r = dict(arm=arm, scene=s, frame=f, base=Q.psnr(base[0], g), m2_lanczos=m2(base[0], base[1]))
    for k in CFG:
        r[k] = Q.psnr(outs[k][0], g) - r["base"]; r["m2_" + k] = m2(outs[k][0], outs[k][1])
    rows.append(r)
    print(f"{arm:6s} {s:15s} f{f:<3d} " + "  ".join(f"{k} {r[k]:+.3f}" for k in CFG) +
          f"  | M2 lanczos {r['m2_lanczos']:.4f} " + " ".join(f"{k} {r['m2_'+k]:.4f}" for k in CFG), flush=True)
    json.dump(rows, open(REPO + "/.scratch/forge/gate-confirm/gate_confirm.json", "w"), indent=1)
print()
for k in CFG:
    m = lambda arms: float(np.mean([r[k] for r in rows if r["arm"] in arms]))
    print(f"{k:4s}: clean {m({'clean'}):+.4f} | crf23 {m({'crf23'}):+.4f} | noisy {m({'noisy'}):+.4f} | crf23+noisy {m({'crf23','noisy'}):+.4f}")
ml = float(np.mean([r["m2_lanczos"] for r in rows]))
print(f"M2 edge flicker mean over {len(rows)} pairs: lanczos {ml:.4f}  " + "  ".join(f"{k} {np.mean([r['m2_'+k] for r in rows]):.4f}" for k in CFG))
for arm in ("clean", "crf23", "noisy"):
    sel = [r for r in rows if r["arm"] == arm]
    print(f"  {arm:6s} M2: lanczos {np.mean([r['m2_lanczos'] for r in sel]):.4f}  " + "  ".join(f"{k} {np.mean([r['m2_'+k] for r in sel]):.4f}" for k in CFG))
