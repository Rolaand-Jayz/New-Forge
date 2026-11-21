#!/usr/bin/env python3
"""FORGE constructed weights — a ForgeDeltaNet whose 18,299 weights are
WRITTEN, not trained: a classical noise-gated denoiser compiled into the
stem / 12 dw+pw blocks / tail. No gradient step, no AMD lineage, no data in
the weights beyond a handful of scalar knobs (KNOBS below).

Algorithm (all at input res, in sRGB-coded YCbCr; x = sRGB - 0.5 is the
net input, the delta leaves in LINEAR light at 3x):

  stem      Y, Cb, Cr; relu(s_c - k_i) per-colour EOTF knots; noise-gate
            ingredients |HH(Y)|, |Gx(Y)|, |HF(Cb)|, |HF(Cr)| (8 channels,
            recycled after block 0 for the output differences)
  blocks    gate:  g = blur^m(wHH*|HH|/4.79 - rho*|Gx| + wC*|HF chroma|),
                   blurred in place; T = relu(g - t0)
                   (Immerkaer noise estimate + chroma evidence minus
                   gradient energy: white noise passes, edges/texture
                   cancel -> T == 0 on clean)
            diffusion x n: s <- s - a*clip(s - blur(s), +-k*T)  per component
                   (clipped-flux smoothing; P/N channels accumulate the two
                   relu halves of the clip, so the estimate stays a linear
                   read-out of the residual stream)
            gamma: linear-light correction E_c = F(s_c + D_c) - F(s_c), F = a
                   convex piecewise-linear EOTF (relu knots) -> the sRGB
                   correction D_c is scaled by the local EOTF slope
            differences of E: d/dx, d/dy per colour; d2/dx2, d2/dy2 of green
  tail      delta[c, phase] = E_c + beta*(ox*Ecx + oy*Ecy)
                              + w*(ox^2*Egxx + oy^2*Egyy), pixel-shuffle

ReLU is made transparent where a signed value must survive by offset
encoding (store v + B, subtract B at the consumer).

Usage:
  forge_construct.py OUT.pt [--mode denoise|dering|merged] [--knobs '{"kY": 2.5}']
                                  write a checkpoint
                                  ({"net": state_dict, "knobs": ...}) usable
                                  by forge_fquality.py and forge_export.py
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_model import BLOCKS, C, SCALE, ForgeDeltaNet  # noqa: E402

# Defaults = coordinate search on the aligned training shards (9 chains,
# sources disjoint from the F-QUALITY scenes), objective = noisy gain with a
# hard penalty on any clean-chain regression. Never tuned on the gate cells.
KNOBS = {
    "rho": 1.5,      # gradient-energy weight in the noise gate
    "t0": 0.00042,     # gate deadband (sRGB units)
    "wHH": 1.0,      # |HH(Y)| weight in the noise gate (Immerkaer-normalised)
    "wC": 0.30,      # chroma |HF| weight in the noise gate
    "gf": 0.02,      # gate floor: g0 is clipped below -gf before blurring
    "m": 5,          # gate blur passes (2..5)
    "gb": 1.0,       # gate blur kernel: 0 = binomial, 1 = 3x3 box
    "kY": 4.0,       # luma clip threshold per unit gate
    "kC": 9.8,       # chroma clip threshold per unit gate
    "a": 1.07,        # diffusion step
    "n": 2,          # diffusion iterations (blocks m .. m+n-1; m+n <= 7)
    "k1": 0.15,      # EOTF knots (sRGB code value)
    "k2": 0.425,
    "beta": 0.40,     # sub-pixel first-difference weight
    "w": 0.085,       # sub-pixel second-difference weight (off-centre phases)
    "gain": 0.83,     # overall delta strength
}

# fp16 feature storage: a value stored as (offset + payload) keeps only
# ~2^-11 of the OFFSET as absolute precision, so offsets are kept small.
EPS = 1.0 / 64       # storage scale of the de-ringing model's recyclables
EPS_ING = 1.0 / 1024  # storage scale of the denoiser's gate ingredients
BI = 0.03            # per-pass offset of the in-place gate blur
BE = 1.0 / 32   # offset for the signed difference channels

# BT.709 RGB <-> YCbCr on sRGB-coded values
A = torch.tensor([[0.2126, 0.7152, 0.0722],
                  [-0.1146, -0.3854, 0.5],
                  [0.5, -0.4542, -0.0458]], dtype=torch.float64)
AI = torch.linalg.inv(A)

ID = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float64)
BLUR = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float64) / 16
HF = ID - BLUR
HH = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=torch.float64)
GX = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float64) / 2
GY = GX.t().contiguous()
DXX = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=torch.float64)


def eotf(s):
    s = torch.as_tensor(s, dtype=torch.float64)
    return torch.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def eotf_segments(k1, k2):
    """Chord slopes of the sRGB EOTF on [0,k1], [k1,k2], [k2,1] as the
    relu-knot coefficients (c0, c1, c2): slope = c0, c0+c1, c0+c1+c2."""
    s0 = float(eotf(k1) / k1)
    s1 = float((eotf(k2) - eotf(k1)) / (k2 - k1))
    s2 = float((eotf(1.0) - eotf(k2)) / (1.0 - k2))
    return s0, s1 - s0, s2 - s1


class Asm:
    """Tiny assembler over the residual stream: named channels, per-block
    depthwise kernels and pointwise rows, with constant-offset tracking.

    A channel can only grow (x += relu(.)), but it can be RECYCLED: raise it
    to a known constant B >= max(x) with x += relu(B - x), then reuse it as
    a fresh channel that carries the offset B. Readers subtract the offsets
    of their sources automatically (write / tail_add)."""

    def __init__(self):
        self.names = {}
        self.off = {}
        self.sc = {}        # storage scale: stored = sc * value + off
        self.stem_w = torch.zeros(C, 3, 3, 3, dtype=torch.float64)
        self.stem_b = torch.zeros(C, dtype=torch.float64)
        self.dw = torch.zeros(BLOCKS, C, 3, 3, dtype=torch.float64)
        self.dw[:, :] = ID
        self.pw = torch.zeros(BLOCKS, C, C, dtype=torch.float64)
        self.pb = torch.zeros(BLOCKS, C, dtype=torch.float64)
        self.tail_w = torch.zeros(3 * SCALE * SCALE, C, dtype=torch.float64)
        self.tail_b = torch.zeros(3 * SCALE * SCALE, dtype=torch.float64)

    def ch(self, name):
        if name not in self.names:
            used = set(self.names.values())
            if len(used) >= C:
                raise ValueError(f"out of channels allocating {name}")
            self.names[name] = min(set(range(C)) - used)
            self.off[name] = 0.0
            self.sc.setdefault(name, 1.0)
        return self.names[name]

    def stem(self, name, kernel, rgb, bias=0.0, sc=1.0):
        """channel = sc * relu(sum_c rgb[c] * (kernel (*) x_c) + bias)."""
        self.sc[name] = sc
        j = self.ch(name)
        for c in range(3):
            self.stem_w[j, c] += sc * float(rgb[c]) * kernel
        self.stem_b[j] += sc * bias

    def kern(self, blk, names, kernel):
        for n in names:
            self.dw[blk, self.ch(n)] = kernel

    def write(self, blk, name, terms, bias=0.0, add_off=0.0):
        """channel += relu(sum coeff * (dw(src) - offset(src)) + bias).
        Set the block's kernels (kern) BEFORE writing. add_off records a
        constant the relu is known to pass through (offset encoding)."""
        j = self.ch(name)
        s = self.sc[name]
        bias = s * bias
        for src, co in terms.items():
            i = self.names[src]
            co = s * co / self.sc[src]
            self.pw[blk, j, i] += co
            bias -= co * self.off[src] * float(self.dw[blk, i].sum())
        self.pb[blk, j] += bias
        self.off[name] += s * add_off

    def recycle(self, blk, old, new, bound, sc=1.0):
        """Raise channel `old` (stored value <= bound everywhere, identity
        kernel in blk) to the constant `bound`; it becomes `new` (storage
        scale sc) with that offset."""
        import math
        # power of two: exactly representable in the player's fp16 feature
        # storage, so the reader's subtraction leaves no constant bias
        bound = 2.0 ** math.ceil(math.log2(bound))
        j = self.names.pop(old)
        assert torch.equal(self.dw[blk, j], ID)
        self.pw[blk, j, j] += -1.0
        self.pb[blk, j] += bound
        assert float(self.pw[blk, j].abs().sum()) == 1.0, f"row of {old} busy in block {blk}"
        self.off.pop(old, None)
        self.sc.pop(old, None)
        self.names[new] = j
        self.off[new] = bound
        self.sc[new] = sc

    def tail_add(self, row, name, co):
        co = co / self.sc[name]
        self.tail_w[row, self.names[name]] += co
        self.tail_b[row] -= co * self.off[name]

    def state_dict(self):
        sd = {"stem.weight": self.stem_w, "stem.bias": self.stem_b,
              "tail.weight": self.tail_w[:, :, None, None], "tail.bias": self.tail_b}
        for i in range(BLOCKS):
            sd[f"body.{i}.dw.weight"] = self.dw[i][:, None]
            sd[f"body.{i}.dw.bias"] = torch.zeros(C, dtype=torch.float64)
            sd[f"body.{i}.pw.weight"] = self.pw[i][:, :, None, None]
            sd[f"body.{i}.pw.bias"] = self.pb[i]
        return {k: v.float().contiguous() for k, v in sd.items()}


def _add(dst, src, scale=1.0):
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + scale * v
    return dst


def construct(knobs=None):
    k = dict(KNOBS)
    k.update(knobs or {})
    m, n = int(k["m"]), int(k["n"])
    if not (2 <= m <= 5 and n >= 1 and m + n <= BLOCKS - 5):
        raise ValueError("need 2 <= m <= 5 and m + n <= 7")
    gblur = (1 - k["gb"]) * BLUR + k["gb"] * torch.full((3, 3), 1 / 9, dtype=torch.float64)
    a = Asm()
    comps = ("Y", "CB", "CR")
    kc = {"Y": k["kY"], "CB": k["kC"], "CR": k["kC"]}

    # ---- stem ---------------------------------------------------------
    a.stem("Y", ID, A[0], 0.5)
    a.stem("CB", ID, A[1], 0.5)          # Cb + 0.5
    a.stem("CR", ID, A[2], 0.5)          # Cr + 0.5
    for c in range(3):                   # per-colour EOTF knots relu(s_c - k_i)
        e = [1.0 if j == c else 0.0 for j in range(3)]
        a.stem(f"SK{c}1", ID, e, 0.5 - k["k1"])
        a.stem(f"SK{c}2", ID, e, 0.5 - k["k2"])
    # channels that must stay offset-free (they are read through HF / d/dx
    # kernels later, where a constant would leak at the zero-padded border)
    for nm in ("G", "TH", "T", "PY", "NY", "PCB", "NCB", "PCR", "NCR",
               "Q01", "Q02", "Q11", "Q12", "Q21", "Q22"):
        a.ch(nm)
    # gate ingredients: |HH(Y)|, |Gx(Y)|, |HF(Cb)|, |HF(Cr)| as relu halves,
    # stored scaled by EPS (floating point: no precision cost) so the
    # recycling bound stays small. Read once (block 0), recycled in block 1.
    ing, bounds = {}, {}
    for nm, kern, row, gain, bound in (
            ("HH", HH, A[0], k["wHH"] / 4.79, 8.0),
            ("GX", GX, A[0], -k["rho"], 0.5),
            ("CBH", HF, A[1], k["wC"], 0.75),
            ("CRH", HF, A[2], k["wC"], 0.75)):
        for sgn, tag in ((1.0, "p"), (-1.0, "n")):
            a.stem(nm + tag, EPS_ING * sgn * kern, row)
            ing[nm + tag] = gain / EPS_ING
            bounds[nm + tag] = EPS_ING * bound

    # ---- noise gate: blocks 0 .. m-1 -----------------------------------
    # block 0: G = max(blur(g0), -gf) + gf; blocks 1..m-2 blur G IN PLACE
    # (x += relu((K - I)x + BI) == Kx + BI); block m-1: T = relu(blur(G) - t0).
    a.kern(0, ing, gblur)
    a.write(0, "G", ing, k["gf"], add_off=k["gf"])
    for b in range(1, m - 1):
        a.dw[b, a.names["G"]] = gblur - ID
        a.pw[b, a.names["G"], a.names["G"]] += 1.0
        a.pb[b, a.names["G"]] += BI
        a.off["G"] += BI
    a.kern(m - 1, ["G"], gblur)
    for t in ("T", "TH"):
        a.write(m - 1, t, {"G": 1.0}, -k["t0"])
    spare = [nm for nm in ing]           # recycle the 8 ingredient channels
    roles = ["EX0", "EY0", "EX1", "EY1", "EX2", "EY2", "EXX", "EYY"]
    for old, new in zip(spare, roles):
        a.recycle(1, old, new, bounds[old])

    # ---- clipped-flux diffusion: blocks m .. m+n-1 ---------------------
    for b in range(m, m + n):
        a.kern(b, ["TH"], HF)
        for c in comps:
            a.kern(b, [c, "P" + c, "N" + c], HF)
            # HF of the running estimate s - a*(P - N - j*k*T), j = b - m
            h = {c: 1.0, "P" + c: -k["a"], "N" + c: k["a"],
                 "TH": k["a"] * (b - m) * kc[c]}
            a.write(b, "P" + c, dict(h, T=kc[c]))
            a.write(b, "N" + c, dict(h, T=-kc[c]))

    # sRGB-domain correction per YCC component, then per colour:
    #   D_comp = -a * (P - N - n*k*T);   D_c = sum_comp AI[c, comp] * D_comp
    def d_rgb(c):
        t = {}
        for j, cp in enumerate(comps):
            w = float(AI[c, j]) * -k["a"]
            _add(t, {"P" + cp: w, "N" + cp: -w, "T": -w * n * kc[cp]})
        return t

    # ---- gamma: block m+n ----------------------------------------------
    c0, c1, c2 = eotf_segments(k["k1"], k["k2"])
    bg = m + n
    e_terms = []
    for c in range(3):
        d = d_rgb(c)
        # s_c = AI[c] . (Y, Cb, Cr); the CB/CR channels carry a +0.5 offset
        sc = _add(dict(d), {"Y": float(AI[c, 0]), "CB": float(AI[c, 1]), "CR": float(AI[c, 2])})
        sb = -0.5 * float(AI[c, 1] + AI[c, 2])
        for i, kn in ((1, k["k1"]), (2, k["k2"])):
            a.write(bg, f"Q{c}{i}", sc, sb - kn)
        e = _add({}, d, c0)
        _add(e, {f"Q{c}1": c1, f"SK{c}1": -c1, f"Q{c}2": c2, f"SK{c}2": -c2})
        e_terms.append(e)

    # ---- differences of E: four blocks ---------------------------------
    # first differences per colour, second differences of green (shared).
    ed = [{("TH" if s == "T" else s): v for s, v in e.items()} for e in e_terms]
    allsrc = sorted({s for e in ed for s in e})
    for blk, kern, tag in ((bg + 1, GX, "EX"), (bg + 2, GY, "EY")):
        a.kern(blk, allsrc, kern)
        for c in range(3):
            a.write(blk, f"{tag}{c}", ed[c], BE, add_off=BE)
    for blk, kern, tag in ((bg + 3, DXX, "EXX"), (bg + 4, DXX.t().contiguous(), "EYY")):
        a.kern(blk, sorted(ed[1]), kern)
        a.write(blk, tag, ed[1], BE, add_off=BE)

    # ---- tail ----------------------------------------------------------
    off = (-1.0, 0.0, 1.0)
    for c in range(3):
        for py in range(SCALE):
            for px in range(SCALE):
                r = c * SCALE * SCALE + py * SCALE + px
                for s, v in e_terms[c].items():
                    a.tail_add(r, s, k["gain"] * v)
                for nm, v in ((f"EX{c}", k["beta"] * off[px]), (f"EY{c}", k["beta"] * off[py]),
                              ("EXX", k["w"] * off[px] ** 2), ("EYY", k["w"] * off[py] ** 2)):
                    a.tail_add(r, nm, k["gain"] * v)
    return a.state_dict(), k, a.names


# ======================================================================
# Model B: lanczos3 de-ringing (helps clean AND compressed footage)
# ======================================================================
DERING_KNOBS = {
    "k1": 0.15,      # EOTF knots for the linear-light luma estimate
    "k2": 0.425,
    "dg": 0.85,      # strength of the correction
    "dt": 0.0,       # deadband (linear light): ringing below this is left alone
}


def _lanczos_taps(o, a=3):
    """Normalised lanczos-a weights of LR taps i = -2..2 for an output
    sample at LR offset o (|o| < 1). The 6th tap of the player's kernel
    (|d| = 2.67, weight ~0.013) is folded into its neighbour."""
    import math

    def L(d):
        if d == 0:
            return 1.0
        if abs(d) >= a:
            return 0.0
        return a * math.sin(math.pi * d) * math.sin(math.pi * d / a) / (math.pi * d) ** 2
    w = [L(i - o) for i in range(-3, 4)]
    w[1] += w[0]
    w[5] += w[6]
    w = w[1:6]
    t = sum(w)
    return [x / t for x in w]


def _factor5(w):
    """5-tap filter -> (pos, rest): two 3-tap filters whose convolution is
    w. `pos` is the factor with the least-negative taps: lanczos3 at +-1/3
    gives (0.755, 0.314, -0.069), so relu(pos * L) only clips where a
    neighbour is >10x brighter than the pixel itself -- there the
    undershoot is under-detected (the safe direction), never invented."""
    import numpy as np
    roots = np.roots(w)
    best = None
    for i in range(4):
        for j in range(i + 1, 4):
            q1 = np.poly([roots[i], roots[j]])
            if abs(q1.imag).max() > 1e-9:
                continue
            q1 = q1.real
            if abs(q1.sum()) > 1e-9:
                q1 = q1 / q1.sum()
                rest = [roots[m] for m in range(4) if m not in (i, j)]
                q2 = np.poly(rest).real
                q2 = q2 * (w[0] / (q1[0] * q2[0]))
                err = abs(np.convolve(q1, q2) - np.array(w)).max()
                score = q1.min()
                if err < 1e-9 and (best is None or score > best[0]):
                    best = (score, q1, q2)
    if best is None:
        raise ValueError("no real quadratic factorisation")
    return torch.tensor(best[1], dtype=torch.float64), torch.tensor(best[2], dtype=torch.float64)


def construct_dering(knobs=None):
    """delta = clamp(v, min3x3, max3x3) - v, with v = lanczos3(L) at each
    of the 8 non-centre sub-pixel phases and L = F(Y): the piecewise-linear
    EOTF of the sRGB luma (monotone, so min L = F(min Y)). The same delta
    is added to R, G and B. Exactly zero wherever lanczos3 does not ring.

    Layout (EPS-scaled where a channel is later recycled):
      stem     Y, relu(Y - k1), relu(Y - k2), ONE
      block 0  8 copies of L (one per phase), M = L, Mn = 1 - Y (sRGB
               domain: uniform fp16 precision in code values), border mask
      block 1  U_p = relu(Apos_p * copy_p)   (lanczos factor 1, near-positive)
      1..4     M <- max over 3x3 (in place, one shift per block); same for Mn
      block 5  v_p = Brest_p * U_p;  Z_p = relu(v_p)  (the player's base is
               clamped at 0);  W+_p = relu(v_p - M)  (overshoot);
               EOTF knots of min Y
      block 7  W-_p = relu(F(min Y) - Z_p)            (undershoot)
      tail     delta[phase p] = -W+_p + W-_p
    A border mask (zero padding vs the player's clamp-to-edge) zeroes both
    on the 2-px frame ring.
    """
    k = dict(DERING_KNOBS)
    k.update(knobs or {})
    a = Asm()
    c0, c1, c2 = eotf_segments(k["k1"], k["k2"])
    a.stem("Y", EPS * ID, A[0], EPS * 0.5)
    a.stem("YK1", EPS * ID, A[0], EPS * (0.5 - k["k1"]))
    a.stem("YK2", EPS * ID, A[0], EPS * (0.5 - k["k2"]))
    a.stem("ONE", torch.zeros(3, 3, dtype=torch.float64), [0, 0, 0], EPS)
    lum = {"Y": c0, "YK1": c1, "YK2": c2}
    stem_ch = ["Y", "YK1", "YK2", "ONE"]

    phases = [(py, px) for py in range(SCALE) for px in range(SCALE) if (py, px) != (1, 1)]
    fac = {}
    for j in range(SCALE):
        if j == 1:
            d = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
            fac[j] = (d, d)
        else:
            fac[j] = _factor5(_lanczos_taps((j - 1) / 3.0))

    # block 0: copies, range seeds, border mask
    for i, _ in enumerate(phases):
        a.write(0, f"V{i}", lum)
    a.write(0, "M", lum)
    a.write(0, "Mn", {"Y": -1.0}, EPS)             # EPS * (1 - Y)
    a.kern(0, ["ONE"], torch.full((3, 3), 1 / 9, dtype=torch.float64))
    a.write(0, "BD", {"ONE": -1.0}, EPS)           # > 0 on the outermost ring
    j = a.names["BD"]                              # block 1: dilate to 2 px
    a.dw[1, j] = torch.ones(3, 3, dtype=torch.float64) - ID
    a.pw[1, j, j] += 1.0
    # block 1: first lanczos factor; recycle the stem channels
    for i, (py, px) in enumerate(phases):
        a.kern(1, [f"V{i}"], torch.outer(fac[py][0], fac[px][0]))
        a.write(1, f"U{i}", {f"V{i}": 1.0})
    for nm in stem_ch:
        a.recycle(1, nm, "r_" + nm, EPS)
    # blocks 1..4: in-place 3x3 max (left, right, up, down)
    for b, (dy, dx) in zip(range(1, 5), ((1, 0), (1, 2), (0, 1), (2, 1))):
        sh = torch.zeros(3, 3, dtype=torch.float64)
        sh[dy, dx] = 1.0
        for nm in ("M", "Mn"):
            j = a.names[nm]
            a.dw[b, j] = sh - ID
            a.pw[b, j, j] += 1.0
    # block 2: recycle the copies -> Z_p
    for i, _ in enumerate(phases):
        a.recycle(2, f"V{i}", f"Z{i}", EPS)
    # block 5: Z_p, W+_p (payloads UNSCALED: full precision above the small
    # recycling offset) and the EOTF knots of min Y
    BIG = 64.0
    wp_slots = ["r_" + nm for nm in stem_ch]
    for i, (py, px) in enumerate(phases):
        a.kern(5, [f"U{i}"], torch.outer(fac[py][1], fac[px][1]))
        if wp_slots:
            old = wp_slots.pop()
            a.names[f"WP{i}"] = a.names.pop(old)
            a.off[f"WP{i}"] = a.off.pop(old)
            a.sc[f"WP{i}"] = a.sc.pop(old)
        a.write(5, f"Z{i}", {f"U{i}": 1.0 / EPS})
        a.write(5, f"WP{i}", {f"U{i}": 1.0 / EPS, "M": -1.0 / EPS, "BD": -BIG / EPS}, -k["dt"])
    a.write(5, "N1", {"Mn": -1.0}, EPS * (1.0 - k["k1"]))   # EPS*relu(minY - k1)
    a.write(5, "N2", {"Mn": -1.0}, EPS * (1.0 - k["k2"]))
    # block 6: recycle U_p; block 7: undershoot against F(min Y)
    mn = {"Mn": -c0 / EPS, "N1": c1 / EPS, "N2": c2 / EPS, "BD": -BIG / EPS}
    for i, (py, px) in enumerate(phases):
        pos = torch.outer(fac[py][0], fac[px][0]).clamp(min=0).sum().item()
        a.recycle(6, f"U{i}", f"WN{i}", EPS * pos)
        a.write(7, f"WN{i}", dict(mn, **{f"Z{i}": -1.0}), c0 - k["dt"])
    for i, (py, px) in enumerate(phases):
        for c in range(3):
            r = c * SCALE * SCALE + py * SCALE + px
            a.tail_add(r, f"WP{i}", -k["dg"])
            a.tail_add(r, f"WN{i}", k["dg"])
    return a.state_dict(), k, a.names

# ======================================================================
# Model C: both in one net (denoiser + lanczos3 undershoot removal)
# ======================================================================
# clean v8 = noisyedge v5 + clean-edge fixes: dr (relative de-ringing
# deadband), dgB (luma-derived undershoot correction at half weight on blue),
# hcap (capped luma HH evidence), chroma evidence stored at EI/8
MERGED_KNOBS = dict(KNOBS, dg=0.85, cw=0.85, dt=0.0, rho=0.75, rhoy=0.75, dgT=50.0, wC=0.65, kY=5.0,
                    dr=0.001, dgB=0.5, hcap=0.5)
SM = 1.0 / 64        # storage scale of channels whose slot is reused later


def construct_merged(knobs=None):
    """delta = denoiser(construct) + undershoot removal, in one net.

    The de-ringing half is cut to what carries its gain (measured on the
    training shards): only the UNDERSHOOT (dark halo) is removed, and only
    the 4 edge phases are computed -- a corner phase gets cw * (the two
    adjacent edge phases). The denoiser is construct() unchanged in its
    arithmetic; what changes is where things live:

      * every channel that dies early is stored scaled (SM or smaller), so
        its slot can be recycled with a small offset;
      * the 6 P/N accumulator slots and the T slot host the de-ringing
        front end during blocks 0-4, the 8 gate-ingredient slots host its
        back end during blocks 2-8 and the sub-pixel differences after;
      * the gate T is folded into the N accumulators in block 7, so Y, Cb,
        Cr and T are dead by block 8 and host the 4 undershoot channels.

    Block by block (D = denoiser, R = de-ringing):
      0  D gate            R Lc, Lc2 = F(Y); min pass 1; frame-ring mask
      1  D gate blur       R U0, U1 (lanczos factor 1); min pass 2 + ring
      2  D gate blur       R U2, U3; Z0, Z1 = relu(lanczos); min pass 3
      3  D gate blur       R Z2, Z3; min pass 4
      4  D T, TH           R N0, N1 = EOTF knots of min Y
      5  D diffusion       R N2
      6  D diffusion
      7  D gamma knots Q; N += n*k*T
      8  D d2/dx2          R W-_p = relu(F(min Y) - Z_p)
      9  D d2/dy2
      10 D d/dx
      11 D d/dy
    Uses 32/32 channels and 12/12 blocks.
    """
    k = dict(MERGED_KNOBS)
    k.update(knobs or {})
    m, n = int(k["m"]), int(k["n"])
    if (m, n) != (5, 2):
        raise ValueError("the merged layout is scheduled for m = 5, n = 2")
    gblur = (1 - k["gb"]) * BLUR + k["gb"] * torch.full((3, 3), 1 / 9, dtype=torch.float64)
    box = torch.full((3, 3), 1 / 9, dtype=torch.float64)
    a = Asm()
    comps = ("Y", "CB", "CR")
    kc = {"Y": k["kY"], "CB": k["kC"], "CR": k["kC"]}
    c0, c1, c2 = eotf_segments(k["k1"], k["k2"])
    EI = 1.0 / 2048      # gate-ingredient storage scale

    # ---- stem ---------------------------------------------------------
    # Stored: Y, B, R (not Cb + 0.5, Cr + 0.5): all three are >= 0 without
    # an offset, so fp16 keeps full relative precision. Cb, Cr and the
    # per-colour s_c are linear read-outs (coefficient rows below).
    a.stem("Y", ID, A[0], 0.5, sc=SM)
    a.stem("CB", ID, [0, 0, 1], 0.5, sc=SM)              # holds B
    a.stem("CR", ID, [1, 0, 0], 0.5, sc=SM)              # holds R
    Minv = torch.linalg.inv(torch.stack([A[0], torch.tensor([0, 0, 1.0], dtype=torch.float64),
                                         torch.tensor([1.0, 0, 0], dtype=torch.float64)]))
    stored = ("Y", "CB", "CR")
    rd = lambda row: {nm: float(v) for nm, v in zip(stored, torch.as_tensor(row, dtype=torch.float64) @ Minv)}
    comp_rd = {cp: rd(A[j]) for j, cp in enumerate(comps)}
    # never-used slots: zero offset. The SK knots relu(s_c - k_i) are NOT
    # taken from the stem here but written in block 7 from the same stored
    # Y/Cb/Cr the Q knots read, so Q - SK cancels exactly (also in fp16)
    # wherever the denoiser does nothing.
    # noisyedge: the knot-1 pairs (Q_c1, SK_c1) are MERGED into the Q_c1
    # slot (SK_c1 written early, negated around a constant XB, Q_c1 added on
    # top): the three SK_c1 slots host CONST (= XB, cancels XB at the
    # zero-padded border under the difference kernels) and two per-colour
    # second-difference outputs.
    for nm in ("G", "TH") + tuple(f"Q{c}{i}" for c in range(3) for i in (1, 2)) \
            + tuple(f"SK{c}2" for c in range(3)) + ("CONST",):
        a.ch(nm)
    SE_ = 16.0
    a.sc["T"] = SM
    a.ch("T")
    ing, bounds = {}, {}
    for nm, kern, row, gain, bound, es in (
            ("HH", HH, A[0], k["wHH"] / 4.79, 8.0, EI / 4),
            ("GX", GX, A[0], -k["rho"], 0.5, EI),
            # noisyedge: chroma noise evidence through the Immerkaer mask
            # too (stock: HF = I - blur, which fires on every colour edge and
            # so OPENS the gate along chroma edges, where the near-unclipped
            # chroma diffusion (kC) then smears the edge). HH is blind to
            # straight edges and ramps; 0.639/4.79 matches the stock
            # white-noise response per unit wC.
            # clean: stored at EI/8 (not EI/4) so the recycled Z0..Z3 slots
            # keep a 2^-11 offset; with 2^-10 the fp16 residue of Z became a
            # positive de-ringing bias on 2/3 of a dark frame
            ("CBH", HH, A[1], k["wC"] * 0.639 / 4.79, 8.0, EI / 8),
            ("CRH", HH, A[2], k["wC"] * 0.639 / 4.79, 8.0, EI / 8),
            # noisyedge: |Gy| too -- with |Gx| alone every horizontal edge
            # (and the y-part of any edge) leaves the gate open on clean
            # sharp structure; rho is split so white noise keeps the stock
            # balance (E|Gx| = E|Gy|)
            ("GY", GY, A[0], -k["rhoy"], 0.5, EI)):
        hc = k.get("hcap", 0.0) if nm == "HH" else 0.0
        if hc > 0:
            # clean: capped one-sided luma evidence 2*(relu(HH) - relu(HH - hcap))
            # = 2*min(relu(HH), hcap): same mean as |HH| on noise well below
            # the cap; isolated structure spikes (corners, text, fine
            # patterns) count at most hcap each
            for tag, sg, bias in (("p", 2.0, 0.0), ("n", -2.0, -hc)):
                a.stem(nm + tag, es * kern, row, es * bias)
                ing[nm + tag] = sg * gain / es
                bounds[nm + tag] = es * bound
            continue
        for sgn, tag in ((1.0, "p"), (-1.0, "n")):
            a.stem(nm + tag, es * sgn * kern, row)
            ing[nm + tag] = gain / es
            bounds[nm + tag] = es * bound
    # de-ringing front end, in the slots the P/N accumulators take at block 5
    SY = 1.0 / 4096      # tiny: U0, U1 inherit these slots in block 1
    a.stem("YK1", ID, A[0], 0.5 - k["k1"], sc=SY)
    a.stem("YK2", ID, A[0], 0.5 - k["k2"], sc=SY)
    SU, SP, SW, SE = 1.0 / 72, 4.0, 16.0, 16.0  # U0/U1; P/N; W-; differences
    SN2 = 0.25           # N2 sits on a 1/32 offset: store it larger
    a.sc["Lc"] = a.sc["Lc2"] = SM
    a.ch("Lc")
    a.ch("Lc2")
    XB = 2.0             # >= 2 * max relu(s - k1) (s in [0, 1]); exact in fp16
    a.stem("CONST", torch.zeros(3, 3, dtype=torch.float64), [0, 0, 0], XB)
    a.off["CONST"] = XB  # value 0, stored XB everywhere
    a.stem("Mn", ID, -A[0], 0.5, sc=1.0 / 1024)          # 1 - Y
    a.stem("ONE", torch.zeros(3, 3, dtype=torch.float64), [0, 0, 0], 1.0, sc=1.0 / 4096)
    assert len(a.names) == C

    # ---- D: noise gate, blocks 0..4 (as construct) ---------------------
    a.kern(0, ing, gblur)
    a.write(0, "G", ing, k["gf"], add_off=k["gf"])
    for b in range(1, m - 1):
        a.dw[b, a.names["G"]] = gblur - ID
        a.pw[b, a.names["G"], a.names["G"]] += 1.0
        a.pb[b, a.names["G"]] += BI
        a.off["G"] += BI
    a.kern(m - 1, ["G"], gblur)
    for t in ("T", "TH"):
        a.write(m - 1, t, {"G": 1.0}, -k["t0"])

    # ---- R: lanczos3 of L = F(Y) at the 4 edge phases -------------------
    phases = [(0, 1), (2, 1), (1, 0), (1, 2)]            # (py, px)
    fac = {1: (torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),) * 2}
    for j in (0, 2):
        fac[j] = _factor5(_lanczos_taps((j - 1) / 3.0))
    K1 = [torch.outer(fac[py][0], fac[px][0]) for py, px in phases]
    K2 = [torch.outer(fac[py][1], fac[px][1]) for py, px in phases]
    lum = {"Y": c0, "YK1": c1, "YK2": c2}
    a.write(0, "Lc", lum)
    a.write(0, "Lc2", lum)
    a.recycle(0, "YK1", "U0", SY, sc=SU)
    a.recycle(0, "YK2", "U1", SY, sc=SU)
    # frame-ring mask (zero padding here vs clamp-to-edge in the player):
    # block 0 turns ONE into 1 + g*[outermost ring]; block 1 adds
    # h*(box(ONE') - 1) > 2 to Mn on the 2-px ring -> min Y < 0 -> no
    # correction there.
    g_ring, h_ring = 8.0, 4.0
    j = a.names["ONE"]
    a.dw[0, j] = box
    a.pw[0, j, j] += -g_ring
    a.pb[0, j] += g_ring * a.sc["ONE"]
    j = a.names["Mn"]                                     # in-place 3x3 max of 1 - Y
    for b, (dy, dx) in zip(range(4), ((1, 0), (1, 2), (0, 1), (2, 1))):
        sh = torch.zeros(3, 3, dtype=torch.float64)
        sh[dy, dx] = 1.0
        a.dw[b, j] = sh - ID
        a.pw[b, j, j] += 1.0
    a.kern(1, ["ONE"], box)
    a.write(1, "Mn", {"ONE": h_ring}, -h_ring)
    # lanczos factor 1 (blocks 1, 2) and factor 2 (blocks 2, 3)
    a.recycle(1, "GXp", "U2", bounds["GXp"], sc=SM)
    a.recycle(1, "GXn", "U3", bounds["GXn"], sc=SM)
    for old, new in (("CBHp", "Z0"), ("CBHn", "Z1"), ("CRHp", "Z2"), ("CRHn", "Z3"),
                     ("HHp", "N0"), ("HHn", "N1")):
        a.recycle(1, old, new, bounds[old], sc=SM)
    a.recycle(1, "GYp", "EXX0", bounds["GYp"], sc=SE_)
    a.recycle(1, "GYn", "EXX2", bounds["GYn"], sc=SE_)
    for i, (src, blk) in enumerate((("Lc", 1), ("Lc2", 1), ("Lc", 2), ("Lc2", 2))):
        a.kern(blk, [src], K1[i])
        a.write(blk, f"U{i}", {src: 1.0})
    for i, blk in enumerate((2, 2, 3, 3)):
        a.kern(blk, [f"U{i}"], K2[i])
        a.write(blk, f"Z{i}", {f"U{i}": 1.0})
    # EOTF knots of min Y = 1 - Mn
    a.write(4, "N0", {"Mn": -1.0}, 1.0)
    a.write(4, "N1", {"Mn": -1.0}, 1.0 - k["k1"])
    upos = max(float(K.clamp(min=0).sum()) for K in K1)
    a.recycle(4, "U2", "N2", a.off["U2"] + SM * upos, sc=SN2)
    a.recycle(4, "U3", "EXX1", a.off["U3"] + SM * upos, sc=SE)
    a.write(5, "N2", {"N1": 1.0}, -(k["k2"] - k["k1"]))
    # hand the front-end slots to the P/N accumulators (equal offsets per
    # pair: the pair is always read as P - N, so the constant cancels even
    # under a difference kernel at the zero-padded border)
    for (o1, o2, cp, bnd) in (("U0", "U1", "Y", SY + SU * upos), ("Lc", "Lc2", "CB", SM),
                              ("Mn", "ONE", "CR", 16.0 / 1024)):
        a.recycle(4, o1, "P" + cp, bnd, sc=SP)
        a.recycle(4, o2, "N" + cp, bnd, sc=SP)

    # ---- D: merged knot 1: Q_c1 slot = XB - SK_c1 (blocks 2, 3) -------------
    for c in range(3):
        s0_ = rd([1.0 if j == c else 0.0 for j in range(3)])
        a.write(2, f"Q{c}1", s0_, -k["k1"])                       # SK_c1
        a.write(3, f"Q{c}1", {f"Q{c}1": -2.0}, XB, add_off=XB)   # XB - SK_c1
    # the gate accumulator G is dead after block 4: second-difference slot
    gmax = (k["wHH"] / 4.79 * 8.0 + 2 * k["wC"] * 0.639 / 4.79 * 8.0) * 1.25 + k["gf"] + (m - 2) * BI
    a.recycle(m, "G", "EYY0", gmax + a.off["G"], sc=SE_)

    # ---- D: clipped-flux diffusion, blocks 5, 6 (as construct) ----------
    for b in range(m, m + n):
        a.kern(b, ["TH"], HF)
        for c in comps:
            a.kern(b, [c, "P" + c, "N" + c], HF)
            h = _add({"P" + c: -k["a"], "N" + c: k["a"],
                      "TH": k["a"] * (b - m) * kc[c]}, comp_rd[c])
            a.write(b, "P" + c, dict(h, T=kc[c]))
            a.write(b, "N" + c, dict(h, T=-kc[c]))

    def d_rgb(c, folded):
        t = {}
        for j, cp in enumerate(comps):
            w = float(AI[c, j]) * -k["a"]
            _add(t, {"P" + cp: w, "N" + cp: -w})
            if not folded:
                _add(t, {"T": -w * n * kc[cp]})
        return t

    # ---- D: gamma knots, block 7; then fold T into N --------------------
    bg = m + n
    e_terms = []
    for c in range(3):
        s0_ = rd([1.0 if j == c else 0.0 for j in range(3)])
        sc_ = _add(d_rgb(c, False), s0_)
        sb = 0.0
        for i, kn in ((1, k["k1"]), (2, k["k2"])):
            a.write(bg, f"Q{c}{i}", sc_, sb - kn)
            if i == 2:
                a.write(bg, f"SK{c}{i}", s0_, sb - kn)
        e = _add({}, d_rgb(c, True), c0)
        _add(e, {f"Q{c}1": c1, "CONST": -c1, f"Q{c}2": c2, f"SK{c}2": -c2})
        e_terms.append(e)
    for cp in comps:                                      # N' = N + n*k*T
        a.write(bg, "N" + cp, {"T": n * kc[cp]})
    for i, old in enumerate(("Y", "CB", "CR", "T")):
        a.recycle(bg, old, f"WN{i}", SM, sc=SW)


    # ---- R: undershoot, block 8 -----------------------------------------
    # clean: relative deadband dr -- W- = relu((1 - dr) F(min Y) - Z_p - ...).
    # The fp16 storage of Z_p / F(min Y) leaves a level-proportional residue
    # (~2^-11 of the value) that the relu turns into a constant positive bias
    # on bright flat areas (~0.8 code on a saturated colour bar); a deadband
    # proportional to the level removes it and keeps the dark-halo gain.
    rl = 1.0 - k.get("dr", 0.0)
    lo = {"N0": c0 * rl, "N1": c1 * rl, "N2": c2 * rl}
    # noisyedge: no undershoot removal where the noise gate is open (TH = T):
    # the luma-based correction lands on all three colours alike, and under
    # noise the 3x3 luma minimum is noise, not structure
    for i in range(4):
        a.write(bg + 1, f"WN{i}", dict(lo, **{f"Z{i}": -1.0, "TH": -k["dgT"]}), -k["dt"])
    a.recycle(bg + 1, "TH", "EYY2", gmax, sc=SE_)
    zpos = max(float(K.clamp(min=0).sum()) for K in K2) * upos
    for old, new in (("Z0", "EYY1"), ("Z1", "EX0"), ("Z2", "EX1"), ("Z3", "EX2")):
        a.recycle(bg + 1, old, new, a.off[old] + SM * zpos, sc=SE)
    for old, new in (("N0", "EY0"), ("N1", "EY1"), ("N2", "EY2")):
        a.recycle(bg + 1, old, new, a.off[old] + a.sc[old], sc=SE)

    # ---- D: differences of E, blocks 8..11 ------------------------------
    allsrc = sorted({s for e in e_terms for s in e})
    DYY = DXX.t().contiguous()
    # per-colour second differences (stock: green's, shared by R and B --
    # on saturated colour it writes green's structure into R and B)
    a.kern(bg + 1, allsrc, DXX)
    a.kern(bg + 2, allsrc, DYY)
    for c in range(3):
        a.write(bg + 1, f"EXX{c}", e_terms[c], BE, add_off=BE)
        a.write(bg + 2, f"EYY{c}", e_terms[c], BE, add_off=BE)
    for blk, kern, tag in ((bg + 3, GX, "EX"), (bg + 4, GY, "EY")):
        a.kern(blk, allsrc, kern)
        for c in range(3):
            a.write(blk, f"{tag}{c}", e_terms[c], BE, add_off=BE)

    # ---- tail ----------------------------------------------------------
    off = (-1.0, 0.0, 1.0)
    pidx = {p: i for i, p in enumerate(phases)}
    for c in range(3):
        for py in range(SCALE):
            for px in range(SCALE):
                r = c * SCALE * SCALE + py * SCALE + px
                for s, v in e_terms[c].items():
                    a.tail_add(r, s, k["gain"] * v)
                for nm, v in ((f"EX{c}", k["beta"] * off[px]), (f"EY{c}", k["beta"] * off[py]),
                              (f"EXX{c}", k["w"] * off[px] ** 2), (f"EYY{c}", k["w"] * off[py] ** 2)):
                    a.tail_add(r, nm, k["gain"] * v)
                if (py, px) in pidx:
                    a.tail_add(r, f"WN{pidx[(py, px)]}", k["dg"] * k.get("dg" + "RGB"[c], 1.0))
                elif (py, px) != (1, 1):
                    for q in ((py, 1), (1, px)):
                        a.tail_add(r, f"WN{pidx[q]}", k["cw"] * k["dg"] * k.get("dg" + "RGB"[c], 1.0))
    # the temporal path (ForgeStage) captures the noise gate T: written by
    # block m-1 into the slot that later holds WN3, stored scaled by SM
    names = dict(a.names)
    names["__gate__"] = (m - 1, a.names["WN3"], 1.0 / SM)
    return a.state_dict(), k, names


CONSTRUCTORS = {"denoise": construct, "dering": construct_dering, "merged": construct_merged}


def build(knobs=None, device="cpu", mode="denoise"):
    sd, k, _ = CONSTRUCTORS[mode](knobs)
    net = ForgeDeltaNet()
    net.load_state_dict(sd)
    return net.to(device).eval(), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--knobs", default="{}")
    ap.add_argument("--mode", choices=tuple(CONSTRUCTORS), default="denoise")
    a = ap.parse_args()
    sd, k, names = CONSTRUCTORS[a.mode](json.loads(a.knobs))
    net = ForgeDeltaNet()
    net.load_state_dict(sd)      # shape / key check
    n_par = sum(v.numel() for v in sd.values())
    nz = sum(int((v != 0).sum()) for v in sd.values())
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))
    from _repo_io import repo_open_w
    with repo_open_w(a.out, "wb") as f:
        torch.save({"net": sd, "step": 0, "knobs": k, "constructed": a.mode}, f)
    gate = names.pop("__gate__", None)
    if gate:      # sidecar the player reads next to the exported weights
        gp = os.path.splitext(a.out)[0] + ".forge.gate"
        with repo_open_w(gp, "w") as f:
            f.write(f"{gate[0]} {gate[1]} {gate[2]:g}\n")
        print(f"temporal gate sidecar {gp}: block {gate[0]} slot {gate[1]} scale {gate[2]:g}")
    print(f"constructed {a.out}: params {n_par:,} ({nz:,} non-zero), "
          f"channels used {len(set(names.values()))}/{C}")
    print("knobs:", json.dumps(k))


if __name__ == "__main__":
    main()
