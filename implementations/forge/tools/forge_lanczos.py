#!/usr/bin/env python3
"""FORGE lanczos3 — the torch port of the player's upscaling base.

out = lanczos3(degraded LR) is the BASE the delta net corrects (design:
docs/design/FORGE-NEW-MODEL.md §1). Kernel: L(x) = sinc(x)*sinc(x/a),
a=3, 6 taps, SEPARABLE; sample convention matches the player's scaler
stage (source_pos = -0.5 + (out + 0.5)/scale — the same convention as the
FSR4 postpass base in tools/postpass_torch.py).

F-CONSISTENCY gate (doc §3): this port must be verified against the
PLAYER'S shader output on a real capture before training begins; a
train/deploy kernel mismatch silently invalidates F-QUALITY. The PIL
self-test here checks the math, not the player parity.

Memory law: row-chunked; never a full-frame [taps, P, C] blowup.
"""
import math

import torch

_A = 3  # lobes


def _lanczos1d(x):
    """L(x) = sinc(x)*sinc(x/a) for |x|<a else 0; L(0)=1."""
    ax = x.abs()
    out = torch.where(
        ax < 1e-8,
        torch.ones_like(x),
        torch.where(
            ax < _A,
            (torch.sin(math.pi * x) / (math.pi * x))
            * (torch.sin(math.pi * x / _A) / (math.pi * x / _A)),
            torch.zeros_like(x)))
    return out


def lanczos3(img, scale, rows_per_chunk=128):
    """img [H, W, C] float tensor -> [scale*H, scale*W, C] (separable,
    row-chunked; horizontal pass then vertical pass)."""
    h, w, c = img.shape
    oh, ow = h * scale, w * scale

    def axis_pass(src, out_len, in_len, dim):
        # output px -> source pos with the player's -0.5 convention
        p = -0.5 + (torch.arange(out_len, dtype=torch.float32,
                                 device=src.device) + 0.5) / scale
        base = torch.floor(p)
        # 6 taps at base-2..base+3 (kernel support |x|<3 around p)
        offs = torch.arange(-2, 4, dtype=torch.float32, device=src.device)
        taps = base.unsqueeze(-1) + offs.unsqueeze(0)        # [out, 6]
        wgt = _lanczos1d(p.unsqueeze(-1) - taps)             # [out, 6]
        wgt = wgt / wgt.sum(-1, keepdim=True).clamp(min=1e-8)
        ti = taps.long().clamp(0, in_len - 1)
        out = torch.zeros(*src.shape[:dim], out_len,
                          *src.shape[dim + 1:], device=src.device)
        # apply along dim in slices of the OTHER axis to bound memory
        free = src.shape[1] if dim == 0 else src.shape[0]
        for f0 in range(0, free, 512):
            f1 = min(f0 + 512, free)
            if dim == 0:      # vertical: src [H, W, C] -> out [oh, W, C]
                s = src[:, f0:f1]                           # [H, s, C]
                g = s[ti]                                    # [oh, 6, s, C]
                g = (g * wgt[:, :, None, None]).sum(1)
                out[:, f0:f1] = g
            else:            # horizontal: src [H, W, C] -> out [H, ow, C]
                s = src[f0:f1]                               # [s, W, C]
                g = s[:, ti]                                 # [s, ow, 6, C]
                g = (g * wgt[None, :, :, None]).sum(2)
                out[f0:f1] = g
        return out

    x = axis_pass(img, ow, w, dim=1)      # horizontal
    y = axis_pass(x, oh, h, dim=0)        # vertical
    return y


def _axis_taps(in_len, scale, device):
    p = -0.5 + (torch.arange(in_len * scale, dtype=torch.float32,
                             device=device) + 0.5) / scale
    taps = torch.floor(p).unsqueeze(-1) + torch.arange(-2, 4, dtype=torch.float32,
                                                       device=device)
    w = _lanczos1d(p.unsqueeze(-1) - taps)
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)
    return taps.long().clamp(0, in_len - 1), w            # [out,6] each


def lanczos3_nchw(x, scale):
    """Batched lanczos3 for training: x [B,C,H,W] -> [B,C,sH,sW]; identical
    math to lanczos3() (same taps/phase/normalization/clamp-to-edge)."""
    b, c, h, w = x.shape
    ti, wt = _axis_taps(w, scale, x.device)
    x = (x[..., ti] * wt).sum(-1)                          # [B,C,H,sW]
    ti, wt = _axis_taps(h, scale, x.device)
    x = (x[:, :, ti, :] * wt[:, :, None]).sum(-2)          # [B,C,sH,sW]
    return x


def _selftest():
    # (0) batched == per-image, and a margin-cropped tile base equals the
    #     full-frame base crop (the training tile contract: margin >= 3)
    img = torch.rand(40, 56, 3)
    full = lanczos3(img, 3)
    bat = lanczos3_nchw(img.permute(2, 0, 1)[None], 3)[0].permute(1, 2, 0)
    e1 = (full - bat).abs().max().item()
    m, y0, x0, t = 4, 10, 12, 16
    tile = img[y0 - m:y0 + t + m, x0 - m:x0 + t + m].permute(2, 0, 1)[None]
    tb = lanczos3_nchw(tile, 3)[0].permute(1, 2, 0)[3 * m:3 * (m + t), 3 * m:3 * (m + t)]
    e2 = (tb - full[3 * y0:3 * (y0 + t), 3 * x0:3 * (x0 + t)]).abs().max().item()
    print(f"batched==per-image {e1:.1e}, tile-with-margin==frame crop {e2:.1e} "
          f"({'PASS' if max(e1, e2) < 1e-5 else 'FAIL'})")
    # (1) DC preservation: flat image upscales to the same flat value
    flat = torch.full((16, 16, 3), 0.37)
    up = lanczos3(flat, 3)
    err = (up - 0.37).abs().max().item()
    print(f"DC preservation max err: {err:.2e} "
          f"({'PASS' if err < 1e-5 else 'FAIL'})")
    # (2) identity at scale=1 (6 taps: only center tap nonzero at exact
    #     integer alignment)
    img = torch.rand(9, 11, 3)
    idn = lanczos3(img, 1)
    err = (idn - img).abs().max().item()
    print(f"scale=1 identity max err: {err:.2e} "
          f"({'PASS' if err < 1e-5 else 'FAIL'})")
    # (3) reference: PIL LANCZOS in FLOAT mode ('F' — PIL's uint8 output
    #     clamps the kernel's negative lobes, which corrupted the first
    #     version of this test; the delta-impulse probe proved the port's
    #     taps match the reference). INTERIOR-only on noise: the port
    #     clamps/replicates at edges, PIL truncates+renormalizes — edge
    #     POLICY differs by design; F-CONSISTENCY (player capture) is the
    #     authority on edges, not PIL.
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    a = rng.random((40, 48)).astype(np.float32) * 255
    pil = np.asarray(Image.fromarray(a, mode="F").resize(
        (144, 120), Image.LANCZOS), dtype=np.float32)
    ours = lanczos3(torch.from_numpy(a[..., None]), 3)[..., 0].numpy()
    err = np.abs(ours[18:-18, 18:-18] - pil[18:-18, 18:-18]).max()
    print(f"vs PIL LANCZOS float interior (noise): {err:.3f}/255 "
          f"({'PASS' if err < 0.5 else 'FAIL'})")


if __name__ == "__main__":
    _selftest()
