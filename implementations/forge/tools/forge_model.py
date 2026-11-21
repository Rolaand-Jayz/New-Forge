#!/usr/bin/env python3
"""FORGE model definition — the lanczos3-conditioned correction net.

Design contract: docs/design/FORGE-NEW-MODEL.md (falsifiers pre-recorded
in §3 BEFORE any training — do not amend without a tracker note).

    out = lanczos3(degraded_LR) + delta(LR)

- Body: 12 ConvNeXt-lite blocks (dw3x3 + pw1x1), C=32, at INPUT res.
- Tail: pointwise C -> 3*k*k channels + pixel-shuffle k -> delta at kx.
- ZERO-INIT tail: delta == 0 at init, so out == lanczos3 exactly (the
  floor-by-construction property; F-QUALITY's worst case is a tie).

Smoke test (this file, run directly): random input -> delta is exactly
zero at init; param/FLOP report vs the §3 F-SPEED budget (<= 40 GFLOP/frame
at 720p->4K 3x).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

C = 32          # body channels
BLOCKS = 12     # ConvNeXt-lite blocks
SCALE = 3       # upscale factor (sizing case: 720p -> 4K)


class Block(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 (the cheap ConvNeXt core)."""

    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return F.relu(self.pw(self.dw(x))) + x


class ForgeDeltaNet(nn.Module):
    """delta(LR): [B,3,H,W] -> [B,3,SCALE*H,SCALE*W], zero at init."""

    def __init__(self, ch=C, blocks=BLOCKS, scale=SCALE):
        super().__init__()
        self.scale = scale
        self.stem = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        # ZERO-INIT tail: the floor-by-construction law (F-QUALITY guard)
        self.tail = nn.Conv2d(ch, 3 * scale * scale, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr):
        x = F.relu(self.stem(lr))
        x = self.body(x)
        d = self.tail(x)                              # [B,3k²,H,W]
        d = F.pixel_shuffle(d, self.scale)            # [B,3,kH,kW]
        return d                                      # zero at init

    def composite(self, lr, base):
        """base = lanczos3(lr) at output res (tools/forge_lanczos.py).
        Composite is elementwise — the addition is free at any res."""
        return base + self.forward(lr)


def _conv3_nhwc(x, w, b):
    """3x3 conv (zero pad 1) in NHWC as 9 shifted matmuls; w [Co,Ci,3,3]."""
    h, wd = x.shape[1:3]
    xp = F.pad(x, (0, 0, 1, 1, 1, 1))
    acc = b
    for i in range(3):
        for j in range(3):
            acc = acc + xp[:, i:i + h, j:j + wd, :] @ w[:, :, i, j].t()
    return acc


def _dw3_nhwc(x, w, b):
    """Depthwise 3x3 (zero pad 1) in NHWC as 9 shifted multiply-adds."""
    h, wd = x.shape[1:3]
    xp = F.pad(x, (0, 0, 1, 1, 1, 1))
    acc = b
    for i in range(3):
        for j in range(3):
            acc = acc + xp[:, i:i + h, j:j + wd, :] * w[:, 0, i, j]
    return acc


def forward_fast(net, lr):
    """TRAINING path: same parameters, same math as net(lr) (verified in
    report(): fwd 1e-7, grads 2e-5 rel), written as NHWC shifts + matmuls so
    torch.compile fuses it. On ROCm 7.2 MIOpen's fp32 depthwise conv costs
    ~55 ms fwd+bwd per layer at [32,32,160,160]; compiled, this path is
    3.2x faster per step (883 -> 272 ms). Deployment math is unaffected —
    the shader implements the conv definition, not this decomposition."""
    x = lr.permute(0, 2, 3, 1)
    x = F.relu(_conv3_nhwc(x, net.stem.weight, net.stem.bias))
    for blk in net.body:
        h = _dw3_nhwc(x, blk.dw.weight, blk.dw.bias)
        h = h @ blk.pw.weight[:, :, 0, 0].t() + blk.pw.bias
        x = F.relu(h) + x
    d = x @ net.tail.weight[:, :, 0, 0].t() + net.tail.bias
    return F.pixel_shuffle(d.permute(0, 3, 1, 2), net.scale)


def report():
    """F-SPEED budget check: params + MACs/pixel at the sizing case."""
    net = ForgeDeltaNet()
    n_par = sum(p.numel() for p in net.parameters())
    macs_per_px = (3 * C * 9                    # stem
                   + BLOCKS * (C * 9 + C * C)   # dw + pw per block
                   + C * 3 * SCALE * SCALE)     # tail
    in_px = 1280 * 720                          # sizing case input
    gflop = 2 * macs_per_px * in_px / 1e9
    print(f"params: {n_par:,}")
    print(f"MACs/px: {macs_per_px:,}  input px: {in_px:,}")
    print(f"frame cost @720p->4K 3x: {gflop:.1f} GFLOP "
          f"(budget <= 40: {'PASS' if gflop <= 40 else 'FAIL'})")
    # zero-init proof: delta must be EXACTLY zero on random input
    lr = torch.rand(1, 3, 64, 64)
    d = net(lr)
    print(f"zero-init delta: max|d| = {d.abs().max().item():.1e} "
          f"({'PASS' if d.abs().max().item() == 0.0 else 'FAIL'})")
    print(f"output shape: {tuple(d.shape)} (want (1,3,{64*SCALE},{64*SCALE}))")
    # training fast path == reference module (outputs and gradients)
    torch.manual_seed(0)
    nn.init.normal_(net.tail.weight, std=0.01)
    x = torch.rand(2, 3, 24, 24)
    e = (net(x) - forward_fast(net, x)).abs().max().item()
    g1 = torch.autograd.grad(net(x).square().mean(), list(net.parameters()))
    g2 = torch.autograd.grad(forward_fast(net, x).square().mean(), list(net.parameters()))
    ge = max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(g1, g2))
    print(f"forward_fast == module: out {e:.1e}, grad rel {ge:.1e} "
          f"({'PASS' if e < 1e-5 and ge < 1e-3 else 'FAIL'})")


if __name__ == "__main__":
    report()
