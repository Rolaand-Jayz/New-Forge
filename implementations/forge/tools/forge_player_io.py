#!/usr/bin/env python3
"""FORGE player I/O — torch replica of the player's colour front end and
capture encode, so the training base sees EXACTLY what the deployed base sees.

The player does NOT upscale sRGB values. Its chain (verified from source):
  decode yuv420p (sw) -> NV12 repack (sws, chroma untouched)
  -> shaders/yuv_to_rgb.comp: chroma BILINEAR-sampled at luma texel centres
     (clamp-to-edge, i.e. centre-sited chroma), limited->full range, matrix
     (BT.709 if width>=720 or height>576 when untagged, else BT.601),
     clamp [0,1], decode transfer (untagged -> BT.709 inverse OETF) -> LINEAR
  -> stored rgba16f
  -> shaders/scaler.comp lanczos3 (6x6, wsum-normalized, clamp-to-edge),
     c = max(c, 0)  (no upper clamp), stored rgba16f
  -> capture: clamp [0,1] -> IEC sRGB encode -> round(v*255)
     (CaptureController.cpp fp16ToRgba8).

So the FORGE base must be computed in linear light on the fp16-rounded
front-end output. tools/forge_consistency.py checks this replica against
player captures (F-CONSISTENCY, FORGE-NEW-MODEL.md §3).
"""
import torch
import torch.nn.functional as F

# Column-major in GLSL: rgb = Y*col0 + Cb*col1 + Cr*col2.
_MATS = {
    0: ((1.0, 0.0, 1.5748), (1.0, -0.1873, -0.4681), (1.0, 1.8556, 0.0)),   # bt709
    1: ((1.0, 0.0, 1.402), (1.0, -0.3441, -0.7141), (1.0, 1.772, 0.0)),     # bt601
    2: ((1.0, 0.0, 1.4746), (1.0, -0.1646, -0.5714), (1.0, 1.8814, 0.0)),   # bt2020
}


def default_matrix(width, height):
    """VideoDecoder.cpp:95 — untagged streams."""
    return 0 if (width >= 720 or height > 576) else 1


def _bilinear_center(plane, out_h, out_w):
    """Sample a [h,w] plane at the luma-texel-centre positions of an
    out_h x out_w grid, bilinear + clamp-to-edge — what texture() does with
    yUv = (pix + 0.5) / srcSize on the half-res chroma texture."""
    return F.interpolate(plane[None, None], size=(out_h, out_w),
                         mode="bilinear", align_corners=False)[0, 0]


GUIDED_TAU, GUIDED_EPS = 0.016, 1e-5


def _box3(x):
    """3x3 mean, replicate padding (x [..., h, w])."""
    return F.avg_pool2d(F.pad(x[None], (1, 1, 1, 1), mode="replicate"), 3, 1)[0]


def _bil2(x, h, w):
    """chroma-grid [C,h0,w0] -> luma grid [C,h,w], centre-sited bilinear, clamped."""
    return F.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False)[0]


def guided_chroma(yn, cn, tau=GUIDED_TAU, eps=GUIDED_EPS):
    """Reprojection-checked guided chroma (shaders/chroma_guided.comp).
    yn [H,W] normalized luma, cn [2,H/2,W/2] normalized chroma -> [2,H,W]."""
    from forge_lanczos import lanczos3
    H, W = yn.shape
    h0, w0 = cn.shape[1:]
    I = F.avg_pool2d(yn[None, None], 2)[0, 0][:h0, :w0]
    mI, mII = _box3(I[None])[0], _box3((I * I)[None])[0]
    mP, mIP = _box3(cn), _box3(I[None] * cn)
    a = (mIP - mI * mP) / ((mII - mI * mI) + eps)
    b = mP - a * mI
    a, b = _box3(a), _box3(b)
    q = _bil2(a, H, W) * yn[None] + _bil2(b, H, W)
    d = cn - F.avg_pool2d(q[None], 2)[0]
    wgt = (1.0 - _box3(d).abs() / tau).clamp(0.0, 1.0)
    lz = lanczos3(cn.permute(1, 2, 0).contiguous(), 2).permute(2, 0, 1)
    wu = _bil2(wgt, H, W)
    return wu * q + (1.0 - wu) * lz


def yuv420_to_linear(y8, u8, v8, matrix=0, full_range=False, transfer=3, chroma="bilinear"):
    """uint8 planes (torch, [H,W],[H/2,W/2]x2) -> linear RGB [H,W,3] float32,
    rounded through fp16 like the rgba16f working image.
    chroma: "bilinear" (yuv_to_rgb.comp alone; the F-QUALITY ruler decode) or
    "guided" (chroma_guided.comp prepass — the FORGE front end)."""
    h, w = y8.shape
    y = y8.float() / 255.0
    if chroma == "guided":
        k = 1.0 if full_range else 255.0 / 224.0
        yn = y if full_range else (y - 16.0 / 255.0) * (255.0 / 219.0)
        cn = torch.stack([(u8.float() / 255.0 - 0.5) * k, (v8.float() / 255.0 - 0.5) * k])
        g = guided_chroma(yn, cn)
        # through the sampled-value domain (rg32f) and back, as the player does
        cb, cr = g[0] / k, g[1] / k
    else:
        cb = _bilinear_center(u8.float() / 255.0, h, w) - 0.5
        cr = _bilinear_center(v8.float() / 255.0, h, w) - 0.5
    if not full_range:
        y = (y - 16.0 / 255.0) * (255.0 / 219.0)
        cb = cb * (255.0 / 224.0)
        cr = cr * (255.0 / 224.0)
    m = _MATS[matrix]
    rgb = torch.stack([m[i][0] * y + m[i][1] * cb + m[i][2] * cr
                       for i in range(3)], dim=-1).clamp(0.0, 1.0)
    if transfer == 3:      # BT.709 inverse OETF
        rgb = torch.where(rgb >= 0.081,
                          ((rgb + 0.099) / 1.099).pow(1.0 / 0.45), rgb / 4.5)
    elif transfer == 0:    # sRGB EOTF
        rgb = torch.where(rgb > 0.04045,
                          ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    else:
        raise NotImplementedError("PQ/HLG not needed for the SDR corpus")
    return rgb.half().float()


def linear_to_srgb8(lin):
    """CaptureController.cpp fp16ToRgba8: fp16 -> clamp[0,1] -> sRGB -> u8."""
    v = lin.half().float().nan_to_num(nan=1.0).clamp(0.0, 1.0)
    v = torch.where(v <= 0.0031308, 12.92 * v,
                    1.055 * v.clamp(min=1e-12).pow(1.0 / 2.4) - 0.055)
    return (v * 255.0 + 0.5).floor().clamp(0, 255).to(torch.uint8)


def srgb8_to_linear(s8):
    """Inverse used for GT / training targets stored as sRGB8."""
    v = s8.float() / 255.0
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055).pow(2.4))


def bt709_inv_oetf(code):
    """BT.709 code value [0,1] -> linear (what yuv_to_rgb.comp transferSel 3
    does after the matrix)."""
    code = code.clamp(0.0, 1.0)
    return torch.where(code >= 0.081, ((code + 0.099) / 1.099).pow(1.0 / 0.45),
                       code / 4.5)


# PLAYER chroma convention. yuv_to_rgb.comp centres chroma at UNORM 0.5
# (8-bit: 127.5, not the standard 128) — a half-code chroma bias that lifts
# R/B (~2 sRGB codes in dark scenes). Filed as a player finding (fix = user
# decision: it re-freezes every baseline). FORGE's GT uses the SAME
# convention, so a model can never "win" by learning to undo it. When the
# player is fixed, flip this AND the shader together.
PLAYER_CHROMA_CENTER_UNORM = 0.5


def gt_linear_from_yuv(y, u, v, bits, matrix=0):
    """THE GT definition (training targets AND F-QUALITY scoring) = the
    player's own decode of the pristine master: native YUV planes (4:2:0 /
    4:2:2 / 4:4:4, 8/10-bit) normalized exactly like yuv_to_rgb.comp
    (raw/(255*2^(bits-8)); luma (y-16/255)*255/219; chroma
    (c-PLAYER_CHROMA_CENTER_UNORM)*255/224), centre-sited bilinear chroma,
    matrix, clamp, BT.709 inverse OETF. No swscale colour math: its
    untagged default is BT.601 (§6) and even with the matrix pinned its
    RGB48 output is ~0.5 code off an exact BT.709 decode (measured)."""
    h, w = y.shape
    # 8-bit-consistent scale: 10-bit code 4k == 8-bit code k exactly. (The
    # player's own p010 path uses raw/1023 — 0.3% off its 8-bit raw/255;
    # LR inputs are 8-bit, so GT must share the 8-bit scale or the net
    # learns a level fix. Identical to the player for bits == 8.)
    mx = 255.0 * float(1 << (bits - 8))
    yf = (y.float() / mx - 16.0 / 255.0) * (255.0 / 219.0)

    def up(c):
        c = c.float() / mx
        if c.shape != (h, w):
            c = F.interpolate(c[None, None], size=(h, w), mode="bilinear",
                              align_corners=False)[0, 0]
        return (c - PLAYER_CHROMA_CENTER_UNORM) * (255.0 / 224.0)
    cb, cr = up(u), up(v)
    m = _MATS[matrix]
    rgb = torch.stack([m[i][0] * yf + m[i][1] * cb + m[i][2] * cr
                       for i in range(3)], dim=-1).clamp(0.0, 1.0)
    return bt709_inv_oetf(rgb)


def srgb_encode(lin):
    """Differentiable IEC sRGB encode (training loss domain = metric domain)."""
    v = lin.clamp(0.0, 1.0)
    return torch.where(v <= 0.0031308, 12.92 * v,
                       1.055 * v.clamp(min=1e-8).pow(1.0 / 2.4) - 0.055)


def player_base(lin, scale):
    """The deployed base: lanczos3 in linear light, max(0), fp16 storage."""
    from forge_lanczos import lanczos3
    return lanczos3(lin, scale).clamp(min=0.0).half().float()


def bt709_oetf(lin):
    lin = lin.clamp(0.0, 1.0)
    return torch.where(lin >= 0.018, 1.099 * lin.clamp(min=1e-12).pow(0.45) - 0.099, 4.5 * lin)


def player_base_code(lin, scale, transfer=3):
    """FORGE's base since 2026-11-16 (shaders/forge_tail.comp flags bit 1/2):
    lanczos3 on the CODE values the content was downscaled in, decoded back.
    transfer 3 = BT.709 OETF, 0 = sRGB; anything else = the linear base.
    (player_base stays the linear-light RULER base of forge_tdev/F-QUALITY.)"""
    from forge_lanczos import lanczos3
    if transfer == 3:
        enc, dec = bt709_oetf, bt709_inv_oetf
    elif transfer == 0:
        enc = srgb_encode

        def dec(p):
            p = p.clamp(0.0, 1.0)
            return torch.where(p > 0.04045, ((p + 0.055) / 1.055).pow(2.4), p / 12.92)
    else:
        return player_base(lin, scale)
    p = lanczos3(enc(lin), scale).clamp(min=0.0).half().float()
    return dec(p).half().float()


def forge_base(lin, scale, transfer=3):
    """The base the FORGE stage composites onto: code-domain unless
    FORGE_BASE=linear (= the player's ANVIL_FORGE_BASE=linear)."""
    import os
    if os.environ.get("FORGE_BASE", "code") == "linear":
        return player_base(lin, scale)
    return player_base_code(lin, scale, transfer)


BP_TREF = 0.004  # back-projection noise reference (= the temporal path's tref)


def _l3w(x):
    import math
    x = abs(x)
    if x < 1e-8:
        return 1.0
    if x >= 3:
        return 0.0
    return 3 * math.sin(math.pi * x) * math.sin(math.pi * x / 3) / (math.pi * x) ** 2


def down3_code(H):
    """lanczos3 x1/3 downscale [3h,3w,C] -> [h,w,C] (17 taps, LR pixel i
    centred on HR pixel 3i+1, replicate edges, horizontal then vertical) —
    the generic model of how 720p content is made from its master. Mirrors
    shaders/forge_bp.comp pass 0."""
    k = torch.tensor([_l3w(d / 3) for d in range(-8, 9)], dtype=torch.float32)
    k = (k / k.sum()).to(H.device)
    x = H.permute(2, 0, 1)[:, None]
    x = F.conv2d(F.pad(x, (7, 9, 0, 0), mode="replicate"), k.view(1, 1, 1, 17), stride=(1, 3))
    x = F.conv2d(F.pad(x, (0, 0, 7, 9), mode="replicate"), k.view(1, 1, 17, 1), stride=(3, 1))
    return x[:, 0].permute(1, 2, 0)


def back_project(out, fused, T, beta=1.0, tref=BP_TREF, transfer=3):
    """Noise-gated single-frame back-projection (temporal study PLAN §6,
    pt_sbp; shaders/forge_bp.comp): in the code domain, push the 4K output
    toward consistency with the LR frame the net saw,
        Q = enc(out);  r = enc(fused) - down3(Q);  g = clamp(1 - T/tref)
        out' = dec(Q + beta * bilinear_up(g) * lanczos3_up(r))
    T = the stored noise gate [h,w] (forge_temporal.Temporal.T). Where the
    gate says noise, the step fades out (it would otherwise copy the noise)."""
    from forge_lanczos import lanczos3
    if transfer == 0:
        enc = srgb_encode

        def dec(p):
            p = p.clamp(0.0, 1.0)
            return torch.where(p > 0.04045, ((p + 0.055) / 1.055).pow(2.4), p / 12.92)
    else:
        enc, dec = bt709_oetf, bt709_inv_oetf
    Q = enc(out)
    r = enc(fused) - down3_code(Q)
    g = (1.0 - T / tref).clamp(0.0, 1.0)[None, None]
    gh = F.interpolate(g, size=Q.shape[:2], mode="bilinear", align_corners=False)[0, 0][..., None]
    return dec(Q + beta * gh * lanczos3(r.contiguous(), 3))


def player_decode(y8, u8, v8, matrix=0, full_range=False, transfer=3):
    """What the player's front end feeds FORGE: guided chroma unless
    FORGE_CHROMA=bilinear (= the player's ANVIL_CHROMA=bilinear)."""
    import os
    return yuv420_to_linear(y8, u8, v8, matrix, full_range, transfer,
                            chroma=os.environ.get("FORGE_CHROMA", "guided"))
