#!/usr/bin/env python3
"""FORGE temporal DEV set — non-gate sequences for tuning the temporal path.

Gate scenes (sintel_rooftop, tos_daylight, tos_debris) are never used here:
temporal knobs are tuned on this set only and confirmed on the gate set once.

Sources: the 2160p lossless masters that are NOT F-QUALITY scenes
(sintel_cave + the synthetic edges/text and motion masters), first
SECONDS seconds, degraded with forge_data chains (c10_lanczos, c23_lanczos,
noisy10) into .scratch/forge/tdev/enc/.

  forge_tdev.py build
  forge_tdev.py eval CKPT.pt [--frames 48] [--score-from 12] [--every 4]
                    [--kappa .. --amin .. --tref .. --hold .. --tmin ..] [--json OUT]
      per (source, chain): mean whole-frame sRGB8 PSNR vs GT over the scored
      frames for lanczos3 / spatial / temporal, and mean |frame-to-frame
      change| error vs GT (flicker proxy, "tflk": lower is better).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("FORGE_TPATH"):  # candidate dir whose modules shadow tools/
    sys.path.insert(0, os.environ["FORGE_TPATH"])
    # forge_temporal re-inserts tools/ at the front when it loads, so import the
    # candidate's constructor FIRST: the module cache then serves it to everyone.
    import forge_construct  # noqa: E402,F401
import forge_data as D  # noqa: E402
import forge_temporal as T  # noqa: E402
from forge_player_io import linear_to_srgb8, player_base, yuv420_to_linear  # noqa: E402

REPO = D.REPO
R = "research/Temporal-Forge-Player/benchmarks/video_corpus/references/"
OUT = os.path.join(REPO, ".scratch/forge/tdev")
SECONDS = 2.5
SOURCES = {
    "sintel_cave": R + "sintel_cave_2160p_lossless.mkv",
    "synth_edges": R + "synthetic_edges_text_2160p_lossless.mkv",
    "synth_motion": R + "synthetic_motion_2160p_lossless.mkv",
}
CHAINS = ["c10_lanczos", "c23_lanczos", "noisy10"]


def src(sid):
    return {"id": sid, "path": os.path.join(REPO, SOURCES[sid]), "max_seconds": SECONDS}


def cmd_build(a):
    for sid in SOURCES:
        for ch in CHAINS:
            print(D.encode(src(sid), ch, OUT), flush=True)


def _need(n, a):
    s = n >= a.score_from and (n - a.score_from) % a.every == 0
    return s or (n >= a.score_from - 1 and (n + 1 - a.score_from) % a.every == 0)


def frames(sid, ch, a):
    """-> (linear LR frame, lazy GT sRGB8) for n < a.frames; both cached in OUT."""
    cdir = os.path.join(OUT, "cache")
    os.makedirs(cdir, exist_ok=True)
    lp = os.path.join(cdir, f"{sid}__{ch}__lr{a.frames}.npy")
    if os.path.exists(lp):
        lrs = np.load(lp)
    else:
        enc = os.path.join(OUT, "enc", f"{sid}__{ch}.mkv")
        lrs = []
        for n, (y, u, v) in enumerate(D._yuv_stream(enc)):
            if n >= a.frames:
                break
            t = lambda x: torch.from_numpy(x.astype(np.int32))
            lrs.append(yuv420_to_linear(t(y), t(u), t(v)).half().numpy())
        lrs = np.stack(lrs)
        np.save(lp, lrs)
    need = [n for n in range(len(lrs)) if _need(n, a)]
    gpath = lambda n: os.path.join(cdir, f"{sid}__gt{n}.npy")
    if not all(os.path.exists(gpath(n)) for n in need):
        for n, gp in enumerate(D._gt_stream(src(sid))):
            if n > max(need):
                break
            if n in need and not os.path.exists(gpath(n)):
                np.save(gpath(n), linear_to_srgb8(D._gt_lin(gp)).numpy())
    for n in range(len(lrs)):
        yield torch.from_numpy(lrs[n].astype(np.float32)), (lambda n=n: np.load(gpath(n)))


def psnr(a, b):
    return float(10 * np.log10(255.0 ** 2 / max(((a - b) ** 2).mean(), 1e-12)))


@torch.no_grad()
def cmd_eval(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, gate = T.load(a.ckpt, dev)
    res = []
    for sid in SOURCES:
        if a.only and sid not in a.only.split(","):
            continue
        for ch in CHAINS:
            if a.chains and ch not in a.chains.split(","):
                continue
            t = T.Temporal(net, gate, a.kappa, a.amin, a.tref, a.hold)
            t.tmin = a.tmin
            t.nclamp = a.nclamp
            t.tdead = a.tdead
            for kv in a.set or []:  # candidate knobs: --set name=float
                k, v = kv.split("=", 1)
                setattr(t, k, float(v))
            acc = {"base": [], "spatial": [], "temporal": []}
            flk = {"spatial": [], "temporal": []}
            fr = []
            prev = {}
            for n, (lin, gt_of) in enumerate(frames(sid, ch, a)):
                lin = lin.to(dev)
                out = t.step(lin)
                if n < a.score_from - 1:
                    continue
                score = n >= a.score_from and (n - a.score_from) % a.every == 0
                need_prev = n >= a.score_from - 1 and ((n + 1 - a.score_from) % a.every == 0)
                if not (score or need_prev):
                    continue
                g = gt_of().astype(np.float64)
                o = {"temporal": linear_to_srgb8(out.cpu()).numpy().astype(np.float64)}
                o["spatial"] = linear_to_srgb8(t.spatial(lin)[0].cpu()).numpy().astype(np.float64)
                o["base"] = linear_to_srgb8(player_base(lin, 3).cpu()).numpy().astype(np.float64)
                if score:
                    fr.append(t.frac)
                    for k in acc:
                        acc[k].append(psnr(o[k], g))
                    if prev.get("n") == n - 1:
                        dg = g - prev["g"]
                        for k in flk:
                            flk[k].append(float(np.abs((o[k] - prev[k]) - dg).mean()))
                prev = {"n": n, "g": g, **o}
            r = {"src": sid, "chain": ch, **{k: float(np.mean(v)) for k, v in acc.items()},
                 **{"tflk_" + k: float(np.mean(v)) if v else None for k, v in flk.items()},
                 "fused_frac": float(np.mean(fr))}
            res.append(r)
            print(f"{sid:13s} {ch:12s} base {r['base']:.3f}  spatial {r['spatial'] - r['base']:+.3f}  "
                  f"temporal {r['temporal'] - r['base']:+.3f}  (vs spatial {r['temporal'] - r['spatial']:+.3f})  "
                  f"tflk S {r['tflk_spatial']:.3f} T {r['tflk_temporal']:.3f}  fused {r['fused_frac'] * 100:.1f}%", flush=True)
    for ch in CHAINS:
        rs = [r for r in res if r["chain"] == ch]
        if rs:
            print(f"MEAN {ch:12s} spatial {np.mean([r['spatial'] - r['base'] for r in rs]):+.3f}  "
                  f"temporal {np.mean([r['temporal'] - r['base'] for r in rs]):+.3f}")
    print(f"MEAN all temporal-vs-base {np.mean([r['temporal'] - r['base'] for r in res]):+.4f}  "
          f"temporal-vs-spatial {np.mean([r['temporal'] - r['spatial'] for r in res]):+.4f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    e = sp.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--frames", type=int, default=48)
    e.add_argument("--score-from", type=int, default=12)
    e.add_argument("--every", type=int, default=4)
    e.add_argument("--kappa", type=float, default=32.0)
    e.add_argument("--amin", type=float, default=0.25)
    e.add_argument("--tref", type=float, default=0.004)
    e.add_argument("--hold", type=float, default=0.9)
    e.add_argument("--tmin", type=float, default=0.0)
    e.add_argument("--nclamp", type=float, default=-1.0)
    e.add_argument("--tdead", type=float, default=0.0)
    e.add_argument("--set", action="append", help="extra Temporal attribute, name=float (repeatable)")
    e.add_argument("--only")
    e.add_argument("--chains")
    e.add_argument("--json")
    a = ap.parse_args()
    {"build": cmd_build, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
