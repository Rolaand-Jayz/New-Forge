#!/usr/bin/env python3
"""FORGE temporal path — offline replica of the player's ForgeStage with
ANVIL_FORGE_TEMPORAL=1 (shaders/forge_pyr.comp, forge_motion.comp,
forge_fuse.comp, forge_gate.comp), frame by frame from the start of a clip:

  pyramid   L0 = sqrt(luma(linear)), L1/L2 = 2x2 / 4x4 means       (fp16)
  motion    8x8 blocks, 3-level block matching, current -> previous (fp16)
  fusion    at INPUT resolution, before the net:
              Hc = S + clamp(Hw - S, +-kappa*Tw),  S = sqrt(cur)
              F  = a*S + (1-a)*Hc,  a = 1 - (1-amin)*clamp(Tw/tref, 0, 1)
            Hw = previous fused frame (gamma-2, bicubic at p + flow),
            Tw = previous frame's noise gate (bilinear at p + flow)
  net+tail  on the fused frame (player_base + ForgeDeltaNet)
  gate      T' = max(scale * max3x3(feat[slot] after block b),
                     hold * T_prev(p + flow))

Usage:
  forge_temporal.py CKPT.pt CLIP FRAME [--start N] [--out PNG] [--vs-capture CELLDIR]
                    [--gt-scene SCENE] [--kappa 32 --amin 0.25 --tref 0.004 --hold 0.9]
      run frames START..FRAME, report the sRGB8 result of FRAME: PSNR vs the
      lossless master (if --gt-scene) for lanczos3 / spatial / temporal, and
      the code difference vs a player capture (CELLDIR/01-forge.png).
The gate channel comes from the checkpoint's constructor (merged mode).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))
import forge_construct as FC  # noqa: E402
import forge_fquality as Q  # noqa: E402
from forge_consistency import decode_yuv  # noqa: E402
from forge_model import ForgeDeltaNet  # noqa: E402
from forge_accum import Accum  # noqa: E402
from forge_player_io import (BP_TREF, back_project, forge_base, linear_to_srgb8, player_base, srgb_encode,  # noqa: E402
                             yuv420_to_linear)

LW = (0.2126, 0.7152, 0.0722)
LAM = 0.004
h16 = lambda t: t.half().float()


def pyramid(lin):
    l0 = (lin * torch.tensor(LW, device=lin.device)).sum(-1).clamp(min=0).sqrt()
    l1 = F.avg_pool2d(l0[None, None], 2)[0, 0]
    l2 = F.avg_pool2d(l0[None, None], 4)[0, 0]
    return h16(l0), h16(l1), h16(l2)


def _sad(cur, prev, x0, y0, n, dx, dy):
    H, W = cur.shape
    ii = torch.arange(n, device=cur.device)
    px = x0[..., None, None] + ii[None, None, None, :]
    py = y0[..., None, None] + ii[None, None, :, None]
    a = cur[py.clamp(0, H - 1), px.clamp(0, W - 1)]
    b = prev[(py + dy[..., None, None]).clamp(0, H - 1), (px + dx[..., None, None]).clamp(0, W - 1)]
    return (a - b).abs().sum((-1, -2))


def motion(pc, pp):
    """block motion [H/8, W/8, 2] (dx, dy), current -> previous, input px."""
    H, W = pc[0].shape
    dev = pc[0].device
    by, bx = torch.meshgrid(torch.arange(H // 8, device=dev), torch.arange(W // 8, device=dev),
                            indexing="ij")
    z = torch.zeros_like(bx)

    def search(cur, prev, x0, y0, n, mx, my, R, sc):
        best, allc = None, []
        for ey in range(-R, R + 1):
            for ex in range(-R, R + 1):
                dx, dy = mx + ex, my + ey
                c = _sad(cur, prev, x0, y0, n, dx, dy) / float(n * n) + LAM * (dx.abs() + dy.abs()).float() * sc
                allc.append(c)
                if best is None:
                    best = (c, dx.clone(), dy.clone())
                else:
                    m = c < best[0]
                    best = (torch.where(m, c, best[0]), torch.where(m, dx, best[1]),
                            torch.where(m, dy, best[2]))
        return best, allc

    (_, dx, dy), _ = search(pc[2], pp[2], 2 * bx - 2, 2 * by - 2, 6, z, z, 2, 0.25)
    (_, dx, dy), _ = search(pc[1], pp[1], 4 * bx - 2, 4 * by - 2, 8, 2 * dx, 2 * dy, 1, 0.5)
    mx, my = 2 * dx, 2 * dy
    (_, dx, dy), allc = search(pc[0], pp[0], 8 * bx - 4, 8 * by - 4, 16, mx, my, 1, 1.0)
    cs = torch.stack(allc).view(3, 3, H // 8, W // 8)
    ex, ey = dx - mx, dy - my
    fx, fy = dx.float(), dy.float()
    g = lambda yy, xx: cs[yy, xx, by, bx]
    cm, c0, cp = g(ey + 1, z), g(ey + 1, z + 1), g(ey + 1, z + 2)
    den = cm + cp - 2 * c0
    fx = fx + torch.where((ex == 0) & (den > 1e-6),
                          (0.5 * (cm - cp) / den.clamp(min=1e-6)).clamp(-0.5, 0.5), torch.zeros_like(den))
    cm, c0, cp = g(z, ex + 1), g(z + 1, ex + 1), g(z + 2, ex + 1)
    den = cm + cp - 2 * c0
    fy = fy + torch.where((ey == 0) & (den > 1e-6),
                          (0.5 * (cm - cp) / den.clamp(min=1e-6)).clamp(-0.5, 0.5), torch.zeros_like(den))
    return h16(torch.stack([fx, fy], -1))


def flow_at_pixels(fl, H, W):
    return F.interpolate(fl.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
                         align_corners=False)[0].permute(1, 2, 0)


def sample(img, f, mode):
    """img [H,W,C] at pixel p + f (input px), clamp-to-edge."""
    H, W = img.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device),
                            indexing="ij")
    gx = (xs + f[..., 0] + 0.5) / W * 2 - 1
    gy = (ys + f[..., 1] + 0.5) / H * 2 - 1
    return F.grid_sample(img.permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None], mode=mode,
                         padding_mode="border", align_corners=False)[0].permute(1, 2, 0)


class Temporal:
    def __init__(self, net, gate, kappa=32.0, amin=0.25, tref=0.004, hold=0.9):
        self.net, self.gate = net, gate
        self.kappa, self.amin, self.tref, self.hold = kappa, amin, tref, hold
        self.tmin = 0.0  # experiment: floor on the warped gate (0 = player behaviour)
        self.nclamp = -1.0  # neighbourhood clamp margin (gamma-2 units); < 0 = off
        self.tdead = 0.0    # stored gate below this snaps to 0 (ends the hold tail)
        # noise-gated back-projection after the tail (player: ANVIL_FORGE_BP, default 1)
        self.bp = float(os.environ.get("FORGE_BP", "1"))
        self.bp_tref = float(os.environ.get("FORGE_BP_TREF", str(BP_TREF)))
        # 4K accumulation after BP (player: ANVIL_FORGE_ACC, default 1; needs BP on)
        self.acc = int(os.environ.get("FORGE_ACC", "1")) != 0
        self.accum = Accum()
        self.reset()

    def reset(self):
        self.pp = self.hist = self.T = None
        self.frac = 0.0
        if hasattr(self, "accum"):
            self.accum.reset()

    @torch.no_grad()
    def spatial(self, lin):
        """-> (final linear [3H,3W,3], raw gate T [H,W]) with fp16 feature storage."""
        blk, slot, scale = self.gate
        h = h16(F.relu(self.net.stem((srgb_encode(lin) - 0.5).permute(2, 0, 1)[None])))
        T = None
        for i, b in enumerate(self.net.body):
            h = h16(b(h))
            if i == blk:
                T = scale * F.max_pool2d(h[:, slot:slot + 1], 3, 1, 1)[0, 0]
        d = F.pixel_shuffle(self.net.tail(h), 3)[0].permute(1, 2, 0)
        return (forge_base(lin, 3) + d).clamp(0, 1), T

    @torch.no_grad()
    def step(self, lin):
        """lin [H,W,3] linear (the player's rgba16f input) -> final linear."""
        pc = pyramid(lin)
        pp = self.pp
        fused, fl = lin, None
        S = lin.clamp(min=0).sqrt()
        if self.hist is not None:
            fl = flow_at_pixels(motion(pc, self.pp), *lin.shape[:2])
            Tw = sample(self.T[..., None], fl, "bilinear")
            if self.tmin > 0:
                Tw = Tw.clamp(min=self.tmin)
            Hw = sample(self.hist, fl, "bicubic")
            if self.nclamp >= 0:
                Sc = S.permute(2, 0, 1)[None]
                hi = F.max_pool2d(Sc, 3, 1, 1)[0].permute(1, 2, 0)
                lo = -F.max_pool2d(-Sc, 3, 1, 1)[0].permute(1, 2, 0)
                Hw = torch.minimum(torch.maximum(Hw, lo - self.nclamp), hi + self.nclamp)
            band = self.kappa * Tw
            Hc = S + torch.minimum(torch.maximum(Hw - S, -band), band)
            a = 1 - (1 - self.amin) * (Tw / self.tref).clamp(0, 1)
            Fg = a * S + (1 - a) * Hc
            self.frac = float((Tw > 0).float().mean())
            S = torch.where(Tw > 0, Fg, S)
            fused = torch.where(Tw > 0, h16(Fg * Fg), lin)
        out, T = self.spatial(fused)
        if fl is not None and self.hold > 0:
            T = torch.maximum(T, self.hold * sample(self.T[..., None], fl, "bilinear")[..., 0])
        if self.tdead > 0:
            T = torch.where(T < self.tdead, torch.zeros_like(T), T)
        self.pp, self.hist, self.T = pc, h16(S), h16(T)
        if self.bp > 0:
            out = back_project(out.half().float(), fused, self.T, self.bp, self.bp_tref).half().float()
            if self.acc:
                out = self.accum.step(out, fused, self.T, pc[0], pp[0] if pp is not None else None,
                                      fl).half().float()
        return out


def load(ckpt, dev):
    st = torch.load(ckpt, weights_only=False, map_location="cpu")
    net = ForgeDeltaNet().to(dev).eval()
    net.load_state_dict(st["net"])
    mode = st.get("constructed")
    if mode != "merged":
        raise SystemExit("temporal path needs a checkpoint that names its gate channel (merged mode)")
    gate = FC.CONSTRUCTORS[mode](st.get("knobs"))[2]["__gate__"]
    return net, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("clip")
    ap.add_argument("frame", type=int)
    ap.add_argument("--start", type=int, default=0,
                    help="first frame (history starts empty there); the player's capture "
                         "seeks to 0.75 s before the target, i.e. FRAME - 18 at 24 fps")
    ap.add_argument("--out")
    ap.add_argument("--vs-capture")
    ap.add_argument("--gt-scene")
    ap.add_argument("--kappa", type=float, default=32.0)
    ap.add_argument("--amin", type=float, default=0.25)
    ap.add_argument("--tref", type=float, default=0.004)
    ap.add_argument("--hold", type=float, default=0.9)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, gate = load(a.ckpt, dev)
    t = Temporal(net, gate, a.kappa, a.amin, a.tref, a.hold)
    for n in range(a.start, a.frame + 1):
        y, u, v = decode_yuv(a.clip, n)
        lin = yuv420_to_linear(y, u, v).to(dev)
        out = t.step(lin)
    o8 = linear_to_srgb8(out.cpu()).numpy()
    if a.out:
        from PIL import Image
        Image.fromarray(o8).save(a.out)
    if a.gt_scene:
        g = Q.gt_frame(a.gt_scene, a.frame).astype(np.float64)
        sp = linear_to_srgb8(t.spatial(lin)[0].cpu()).numpy()
        ba = linear_to_srgb8(player_base(lin, 3).cpu()).numpy()
        print(f"frame {a.frame}: lanczos3 {Q.psnr(ba, g):.3f}  spatial {Q.psnr(sp, g):.3f}  "
              f"temporal {Q.psnr(o8, g):.3f} dB")
    if a.vs_capture:
        from PIL import Image
        c = np.asarray(Image.open(os.path.join(a.vs_capture, "01-forge.png")).convert("RGB")).astype(np.int32)
        d = np.abs(c - o8.astype(np.int32))
        print(f"vs capture: within1={(d <= 1).mean() * 100:.4f}%  max={d.max()}  mean={d.mean():.4f}  "
              f">2 codes: {(d > 2).mean() * 100:.4f}%")


if __name__ == "__main__":
    main()
