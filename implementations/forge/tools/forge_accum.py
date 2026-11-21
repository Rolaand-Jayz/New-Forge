#!/usr/bin/env python3
"""FORGE 4K accumulation — torch reference of shaders/forge_acc.comp (parity gate).

Temporal study 2026-11-16 (.scratch/forge/temporal-study/PLAN.md §6b, variant acc_hbn).
Runs after the noise-gated back-projection (forge_player_io.back_project), all in the
code domain (BT.709 OETF or sRGB of linear), state H at output resolution:

  f    = motion cur -> prev at input px: pyramid Lucas-Kanade from zero on the gamma-2 luma
         (pyr_lk: levels 3..0, each a 2^l box-average of level 0, 3 passes per level, flow
         carried down by bilinear x2 * 2)
  c    = exp(-(|box3(Y - warp(Yprev, f))| / (S0 + K*T))^2)    motion confidence (LR)
  Hw   = bicubic warp of H by 3f                                   (4K)
  E    = Hw - Xc;  Hc = Xc + up(c) * (E - U(D(E)))                 history keeps only what
                                                                   the LR frame cannot see
  M    = 3x3 sub-pixel phase coverage (9 bits, LR): nearest-warped, bins rotated by
         roundEven(-3 f) mod 3, cleared where c <= 0.5; nov = bin 0 unseen; then set
  H    = Hc + up(g * nov) * U(yf - D(Hc)),  g = clamp(1 - T/tref)  new phases only, noise-gated
  out  = dec(H)
D = forge_player_io.down3_code (lanczos3 x1/3), U = lanczos3 x3, up = bilinear x3.
Intermediates E, Hc and the state H are fp16 (the player's rgba16f images)."""
import torch
import torch.nn.functional as F

from forge_lanczos import lanczos3
from forge_player_io import (BP_TREF, bt709_inv_oetf, bt709_oetf, down3_code, srgb_encode)

ACC_S0, ACC_K = 0.006, 8.0
LK_ITERS, LK_RAD, LK_LAM, LK_SMOOTH = 2, 3, 1e-4, 2


def h16(x):
    return x.half().float()


def _sample(img, f, mode):  # img [H,W,C] at p + f (px), clamp-to-edge (forge_temporal.sample)
    H, W = img.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device),
                            indexing="ij")
    gx = (xs + f[..., 0] + 0.5) / W * 2 - 1
    gy = (ys + f[..., 1] + 0.5) / H * 2 - 1
    return F.grid_sample(img.permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None], mode=mode,
                         padding_mode="border", align_corners=False)[0].permute(1, 2, 0)


def _box(x, r):  # [H,W,C] (2r+1)^2 mean, replicate edges
    k = 2 * r + 1
    y = F.pad(x.permute(2, 0, 1)[None], (r, r, r, r), mode="replicate")
    return F.avg_pool2d(y, k, 1)[0].permute(1, 2, 0)


def _up(x, H, W):  # [h,w,C] -> [H,W,C] bilinear, align_corners=False
    return F.interpolate(x.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
                         align_corners=False)[0].permute(1, 2, 0)


def lk_refine(f, A, B, it=LK_ITERS, rad=LK_RAD, lam=LK_LAM, fs=LK_SMOOTH):
    """Dense Lucas-Kanade refinement of f (cur->prev, [H,W,2]) so that A(p) ~ B(p + f).
    Per pass: Bw = bilinear B(p+f); central-difference gradients of Bw (clamped);
    (2r+1)^2 box structure tensor + lam; f -= clip(solve, -1, 1); then (2fs+1)^2 box of f."""
    A, B = A[..., None], B[..., None]
    for _ in range(it):
        Bw = _sample(B, f, "bilinear")
        P = F.pad(Bw.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0, 0]
        Ix = (P[1:-1, 2:] - P[1:-1, :-2]) * 0.5
        Iy = (P[2:, 1:-1] - P[:-2, 1:-1]) * 0.5
        e = Bw[..., 0] - A[..., 0]
        S = _box(torch.stack([Ix * Ix, Ix * Iy, Iy * Iy, Ix * e, Iy * e], -1), rad)
        sxx, sxy, syy, sxe, sye = S.unbind(-1)
        sxx, syy = sxx + lam, syy + lam
        det = sxx * syy - sxy * sxy
        du = ((syy * sxe - sxy * sye) / det).clamp(-1, 1)
        dv = ((sxx * sye - sxy * sxe) / det).clamp(-1, 1)
        f = f - torch.stack([du, dv], -1)
        if fs:
            f = _box(f, fs)
    return f


LK_LEVELS, LK_PASSES = 4, 3


def lk_pyramid(Y, levels=LK_LEVELS):
    """[Y, Y/2, Y/4, Y/8]: each level a box average of level 0 over 2^l x 2^l (remainder dropped)."""
    out = [Y]
    for l in range(1, levels):
        s = 2 ** l
        out.append(F.avg_pool2d(Y[None, None], s, s)[0, 0])
    return out


def pyr_lk(PA, PB, passes=LK_PASSES):
    """Pyramid LK cur (PA) -> prev (PB) from zero motion; returns [H,W,2] at level 0."""
    f = None
    for l in range(len(PA) - 1, -1, -1):
        h, w = PA[l].shape
        f = (torch.zeros(h, w, 2, device=PA[l].device) if f is None else _up(f, h, w) * 2)
        f = lk_refine(f, PA[l], PB[l], passes)
    return f


_BX = None


def _rot_mask(M, f):  # M [h,w,9] float 0/1, bin = bx*3 + by
    global _BX
    if _BX is None or _BX[0].device != M.device:
        _BX = (torch.arange(3, device=M.device).view(1, 1, 3, 1),
               torch.arange(3, device=M.device).view(1, 1, 1, 3))
    q = torch.round(-3 * f).long() % 3  # torch.round = round half to even (GLSL roundEven)
    sx = (_BX[0] - q[..., 0, None, None]) % 3
    sy = (_BX[1] - q[..., 1, None, None]) % 3
    return torch.gather(M, 2, (sx * 3 + sy).reshape(*M.shape[:2], 9))


class Accum:
    def __init__(self, transfer=3, s0=ACC_S0, k=ACC_K, tref=BP_TREF, flow="plk", it=LK_ITERS):
        self.transfer, self.s0, self.k, self.tref, self.flow, self.it = transfer, s0, k, tref, flow, it
        self.reset()

    def reset(self):
        self.H = self.M = self.P = None

    def enc(self, x):
        return srgb_encode(x) if self.transfer == 0 else bt709_oetf(x)

    def dec(self, p):
        if self.transfer == 0:
            p = p.clamp(0.0, 1.0)
            return torch.where(p > 0.04045, ((p + 0.055) / 1.055).pow(2.4), p / 12.92)
        return bt709_inv_oetf(p)

    @torch.no_grad()
    def step(self, X, fused, T, Y, Yprev, fbm):
        """X 4K linear (after BP), fused LR linear, T stored gate [h,w], Y/Yprev gamma-2 luma
        [h,w] (the player's pyramid level 0), fbm block motion at pixels [h,w,2] or None
        (no history: reset). flow "plk" ignores fbm's values (pyramid LK from zero); "bm"
        refines fbm by `it` LK passes; "given" uses fbm as is. Returns the 4K linear output."""
        Xc = self.enc(X)
        Hh, Wh = Xc.shape[:2]
        Pc = lk_pyramid(Y) if self.flow == "plk" else None
        Pp, self.P = self.P, Pc
        if fbm is None or self.H is None or (self.flow == "plk" and Pp is None):
            self.H = h16(Xc.clamp(-0.1, 1.1))
            self.M = torch.zeros(*Y.shape, 9, device=Y.device)
            self.M[..., 0] = 1
            return self.dec(self.H)
        if self.flow == "plk":
            f = pyr_lk(Pc, Pp)
        elif self.flow == "bm" and self.it > 0:
            f = lk_refine(fbm, Y, Yprev, self.it)
        else:
            f = fbm
        Yw = _sample(Yprev[..., None], f, "bilinear")[..., 0]
        R = _box((Y - Yw)[..., None], 1)[..., 0].abs()
        c = torch.exp(-(R / (self.s0 + self.k * T)) ** 2)
        Hw = _sample(self.H, _up(f, Hh, Wh) * 3, "bicubic")
        Mw = (_sample(self.M, f, "nearest") > 0.5).float()
        Mw = _rot_mask(Mw, f) * (c > 0.5).float()[..., None]
        nov = 1 - Mw[..., 0]
        Mw[..., 0] = 1
        self.M = Mw
        E = h16(Hw - Xc)
        Hc = h16(Xc + _up(c[..., None], Hh, Wh) * (E - lanczos3(down3_code(E).contiguous(), 3)))
        g = (1 - T / self.tref).clamp(0, 1)
        s = _up((g * nov)[..., None], Hh, Wh)
        yf = self.enc(fused)
        H = Hc + s * lanczos3((yf - down3_code(Hc)).contiguous(), 3)
        self.H = h16(H.clamp(-0.1, 1.1))
        self.last_nov = float(nov.mean())
        self.dbg = {"c": c, "s": g * nov, "f": f}
        return self.dec(self.H)
