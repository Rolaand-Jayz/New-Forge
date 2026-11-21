#!/usr/bin/env python3
"""Does the player DIGEST or REJECT its history? Measurement only (no tuning), player as built.
Per cell: the capture warm-up (FRAME-18 .. FRAME), statistics over the last 10 frames:
  LR fusion (forge_fuse): used  = share of pixels where the warped gate Tw > 0 (history blended)
                          w     = mean history weight (1 - a) where used
                          clamp = share of used pixels where the history was cut back by the band
  4K accumulation:        acc   = share of pixels with motion confidence c > 0.5 (history kept)
                          c     = mean confidence (the weight of the history's hidden band)
                          new   = share of pixels whose sub-pixel phase was new (BP re-applied)
                          bp    = mean BP strength there (noise gate x new)"""
import os, sys, json, numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO + "/tools"); sys.path.insert(0, REPO + "/tools/lab"); os.chdir(REPO)
import torch.nn.functional as F
import forge_fquality as Q, forge_temporal as FT
from forge_temporal import pyramid, motion, flow_at_pixels, sample
from forge_consistency import decode_yuv
from forge_player_io import yuv420_to_linear
dev = os.environ.get("FORGE_DEVICE", "cpu")
net, gate = FT.load(REPO + "/.scratch/forge/constructed/merged-v2.pt", dev)

class Probe(FT.Temporal):
    def step(self, lin):
        st = None
        if self.hist is not None:  # recompute the fusion's decisions (same maths as Temporal.step)
            pc = pyramid(lin); fl = flow_at_pixels(motion(pc, self.pp), *lin.shape[:2])
            Tw = sample(self.T[..., None], fl, "bilinear")[..., 0]
            S = lin.clamp(min=0).sqrt(); Hw = sample(self.hist, fl, "bicubic")
            used = Tw > 0
            a = 1 - (1 - self.amin) * (Tw / self.tref).clamp(0, 1)
            cut = ((Hw - S).abs() > (self.kappa * Tw)[..., None]).any(-1)
            st = dict(used=float(used.float().mean()),
                      w=float((1 - a)[used].mean()) if used.any() else 0.0,
                      clamp=float(cut[used].float().mean()) if used.any() else 0.0)
        out = super().step(lin)
        d = getattr(self.accum, "dbg", None)
        if st is not None and d is not None:
            c, s = d["c"], d["s"]
            st.update(acc=float((c > 0.5).float().mean()), c=float(c.mean()),
                      new=float((s > 0).float().mean()), bp=float(s.mean()))
        self.st = st
        return out

CAVE = "research/Temporal-Forge-Player/benchmarks/video_corpus/clips/"; CC = ".scratch/forge/cave-clips/"
CELLS = [(arm, s, f, Q.CLIPS[arm](s)) for arm, s, f in Q.cells()]
CELLS += [(arm, "sintel_cave", f, p) for arm, p in (("clean", CAVE + "sintel_cave_1280x720_hq_crf10.mp4"),
          ("crf23", CAVE + "sintel_cave_1280x720_medium_crf23.mp4"), ("crf27", CC + "sintel_cave_1280x720_medium_crf27.mp4"),
          ("noisy", CC + "sintel_cave_1280x720_hq_crf10_noisy.mp4")) for f in (72, 120, 168)]
TD = ".scratch/forge/tos-daylight-clips/"
N0TD = round(Q._start(Q.CLIPS["clean"]("tos_daylight")) * 24)
CELLS += [(arm, "tos_daylight", f, p, N0TD) for arm, p in (("crf27", TD + "tos_daylight_1280x720_medium_crf27.mp4"),
          ("noisy", TD + "tos_daylight_1280x720_hq_crf10_noisy.mp4")) for f in (72, 120, 168)]
K = ["used", "w", "clamp", "acc", "c", "new", "bp"]
rows = []
for cell in CELLS:
    arm, s, f, clip = cell[:4]
    n = f - (cell[4] if len(cell) > 4 else round(Q._start(clip) * 24))
    t = Probe(net, gate); t.nclamp, t.tdead = 0.0, 0.0
    acc = {k: [] for k in K}
    for fr in range(n - 18, n + 1):
        y, u, v = decode_yuv(clip, fr)
        T = lambda x: torch.as_tensor(np.asarray(x)).to(torch.int32)
        t.step(yuv420_to_linear(T(y), T(u), T(v), chroma="guided").to(dev))
        if fr > n - 10 and t.st:
            for k in K: acc[k].append(t.st[k])
    r = dict(arm=arm, scene=s, frame=f, **{k: float(np.mean(acc[k])) for k in K}); rows.append(r)
    print(f"{arm:6s} {s:15s} f{f:<3d}  LR fusion: used {100*r['used']:5.1f}%  w {r['w']:.3f}  clamped {100*r['clamp']:5.1f}%"
          f"  | 4K: kept {100*r['acc']:5.1f}%  c {r['c']:.3f}  new-phase {100*r['new']:5.1f}%  bp {r['bp']:.3f}", flush=True)
json.dump(rows, open(REPO + "/.scratch/forge/gate-confirm/history_use.json", "w"), indent=1)
print()
for grp, sel in (("protected", lambda r: r["scene"] != "sintel_cave"), ("cave", lambda r: r["scene"] == "sintel_cave"),
                 ("tos_dayl.", lambda r: r["scene"] == "tos_daylight")):
    for arm in ("clean", "crf23", "crf27", "noisy"):
        R = [r for r in rows if sel(r) and r["arm"] == arm]
        if R:
            m = lambda k: np.mean([r[k] for r in R])
            print(f"{grp:9s} {arm:6s} LR fusion used {100*m('used'):5.1f}% (w {m('w'):.3f}, clamped {100*m('clamp'):4.1f}%) | "
                  f"4K history kept {100*m('acc'):5.1f}% (c {m('c'):.3f}), new-phase {100*m('new'):5.1f}%")
