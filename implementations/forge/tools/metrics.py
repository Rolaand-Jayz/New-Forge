#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ANVIL metrics — offline image/sequence quality metrics CLI for the experiment lab.

OVERVIEW
    Compare two images (A = test/processed, B = reference), two directories of
    sequentially-named images (paired by sorted filename order), or analyse one
    directory for temporal flicker.  Outputs a human table on stdout and an
    optional machine-readable JSON report.

    usage examples
        metrics.py a.png b.png
        metrics.py out_dir/ gt_dir/ --gt --json report.json
        metrics.py out_dir/ ref.png                    # ref broadcast to all frames
        metrics.py out_dir/ --temporal --verbose
        metrics.py out.png ref.png --diff heat/ --gain 8 --periodicity
        metrics.py --selftest

INPUTS / PAIRING
    A and B are each either a single image file or a directory.  Directories are
    scanned for *.png / *.jpg / *.jpeg (case-insensitive, hidden files skipped)
    and paired by plain ``sorted()`` order — zero-pad your sequence numbers.
    * file vs file        : one pair.
    * dir vs dir          : counts must match (else error); in non-GT mode, if
                            the reference directory contains exactly one frame
                            it is broadcast to every A frame (with a notice).
    * dir vs file         : the reference image is broadcast to every A frame.
    * --gt                : strict mode.  A and B must both be directories with
                            identical frame counts (no broadcast, no leniency).
    * --temporal          : takes A only (a directory); B must be absent.

DECODING
    PNG (8/ and 16-bit) and JPG are supported.  Decoding backend is detected at
    runtime: PIL/Pillow if importable, otherwise each file is piped through
    ``ffmpeg -f rawvideo -pix_fmt rgb24 -`` (rgb48le when the source pixel
    format is 16-bit), with geometry from ``ffprobe``.  Grayscale is upconverted
    to triplicated RGB on both backends so results are backend-independent
    (16-bit grayscale via PIL stays single-channel; values are identical across
    channels so every metric below is unchanged).  Alpha is dropped.
    Code values are normalised to [0, 1]: 8-bit /255, 16-bit /65535.
    Environment overrides (for testing): ANVIL_METRICS_FORCE_PURE=1 disables
    numpy, ANVIL_METRICS_FORCE_FFMPEG=1 disables PIL decoding.

NUMPY POLICY
    numpy is OPTIONAL: every metric has a pure-Python fallback so the tool runs
    on a bare CPython.  When numpy is importable it accelerates every metric
    (vectorised paths, same formulas — --selftest asserts the two paths agree
    to <1e-9).  Without numpy, SSIM/periodicity on megapixel frames is slow
    (single-threaded pure Python); it is correct, just not fast.  numpy is
    never *required* — the sane requirement would only ever be speed.

REFERENCE TRANSFER CONTRACT (post-port vs legacy-raw)
    Since commit 9938e07 the player decodes BT.709/PQ/HLG to linear at entry
    and sRGB-encodes ONCE at present/capture, so captured PNGs (A) are
    sRGB-encoded linear light, while ground-truth references (B) are ffmpeg
    extracts of the raw video carrying BT.709 OETF code values.  By default B
    is therefore mapped bt709-decode -> linear -> sRGB-encode before any
    metric (same transform as tools/lab/ref_transfer.py; both selftests
    cross-check parity).  --legacy-raw (or LEGACY_RAW=1) skips the transform:
    the report is then labelled a PRE-PORT-CONTRACT comparison and must never
    be mixed with post-port verdicts.  The contract mode is printed in the
    summary ("ref contract") and recorded in JSON ("ref_contract").

METRIC DEFINITIONS (all on [0,1]-normalised code values, no gamma applied —
"linear of code values": we compute on code values, NOT linear light)
    PSNR
        MSE  = mean over all pixels AND channels of (A-B)^2, in [0,1] units.
        PSNR = 10*log10(1/MSE) dB, i.e. peak = 1.0.  Identical inputs give
        MSE = 0 -> PSNR = +inf (printed "inf", JSON null, "all_identical":true).
    SSIM
        Standard Wang et al. implementation: 11x11 Gaussian window, sigma 1.5
        (separable, weights renormalised to sum 1), C1 = 0.01^2, C2 = 0.03^2
        with dynamic range L = 1.0.  Local means/variances/covariance over the
        valid (border-cropped by 5px) region; the index is the mean of the
        local SSIM map.  Default operates on Rec.601 luma
        Y = 0.299 R + 0.587 G + 0.114 B (--ssim-mode luma);
        --ssim-mode channels computes SSIM per channel then averages.
        Frames must be >= 11 px in both dimensions.
    PERIODICITY (--periodicity [N], default N=2)
        Lattice-artifact detector for the known 2x2/subpixel repeating-pattern
        failure mode.  For each lag (dx,dy):
            MAD(dx,dy) = mean over the valid overlap of
                         |Y(x,y) - Y(x+dx, y+dy)|          (Rec.601 luma)
        "Aligned" (half-pixel-consistent) lags satisfy dx % N == 0 and
        dy % N == 0 — a period-N lattice repeats onto itself under such shifts.
        All other lags in the set are "misaligned" ("odd").  Then
            lattice_score = mean(MAD over misaligned lags)
                            / (mean(MAD over aligned lags) + 1e-6)
        A strong lattice artifact has ~zero residual at aligned lags and full
        amplitude at odd lags, so the score is HUGE; natural/aperiodic content
        sits near 1; flat content scores 0 (0/eps).  HIGHER = MORE PERIODIC
        ARTIFACT.  Default lag set for period N:
            aligned     = {(+N,0),(-N,0),(0,+N),(0,-N)}
            misaligned  = {(+/-k,0),(0,+/-k): 1<=k<N} U {(+/-d,+/-d): 1<=d<N}
        (--lags "dx,dy;dx,dy;..." overrides the set; classification still uses
        N).  For N=2 this is exactly (+-2,0),(0,+-2) aligned vs (+-1,0),(0,+-1),
        (+-1,+-1) misaligned.  Computed on A frames only (the artifact lives in
        the processed output).
    TEMPORAL (--temporal)
        Flicker proxy on one directory: for each consecutive frame pair,
        delta_i = mean |Y_i - Y_{i+1}| (Rec.601 luma, [0,1] units); the report
        carries the mean over pairs and each per-pair delta.
    --diff [DIR]
        Per frame, |luma(A) - luma(B)| * --gain (default 4.0), clamped to [0,1],
        coloured with an inline viridis-like colormap (piecewise-linear over
        the 11 canonical viridis anchor colours) and written as an 8-bit RGB
        PNG by the tool's own stdlib PNG writer (zlib + struct — no deps).

JSON SCHEMA (--json PATH)
    {
      "tool": "anvil-metrics", "version": "1.0.0", "generated_utc": "...",
      "decode": {"backend": "PIL"|"ffmpeg", "numpy_used": true|false},
      "config": {"mode": "pair|sequence|gt|temporal", "metrics": [...],
                 "ssim_mode": "luma|channels", "periodicity_n": 2|null,
                 "lags": [[dx,dy],...], "lag_classes": {"dx,dy": "aligned|misaligned"},
                 "ref_contract": "...", "ref_transfer_applied": true|false,
                 "gain": 4.0, "diff_dir": "..."|null},
      "inputs": {"a": "path", "b": "path|null",
                 "frames_a": ["file",...], "frames_b": ["file",...]|null},
      "comparison": {                       # null in --temporal mode
        "num_frames": n,
        "psnr":  {"mean": dB|null, "min": dB|null, "per_frame": [dB|null,...],
                  "all_identical": bool},
        "ssim":   {"mean": f, "min": f, "per_frame": [f,...]},
        "periodicity": {                    # only with --periodicity
          "lattice_score": {"mean": f, "min": f, "per_frame": [f,...]},
          "lags": [{"dx": n, "dy": n, "class": "...",
                    "mad_mean": f, "mad_min": f, "mad_per_frame": [f,...]}]}},
      "temporal": {"num_frames": n, "num_pairs": n,
                   "mean_delta": f, "per_pair": [f,...]}   # only --temporal
    }
    +/-inf (only possible for PSNR of identical frames) is encoded as null.

SELFTEST (--selftest)
    Generates synthetic in-memory references (flat frames, gradients, known
    additive-noise pairs with analytically known PSNR, a strong period-2
    checkerboard for periodicity, a 3-frame known ramp for temporal) and
    asserts every metric within tolerance, on the pure-Python path and, when
    available, on the numpy path plus pure-vs-numpy parity.  Also round-trips
    the internal PNG writer.  Exit code 0 on success, 1 on any failure.

EXIT CODES: 0 ok; 1 selftest failure; 2 usage/input error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import zlib
from datetime import datetime, timezone

TOOL = "anvil-metrics"
VERSION = "1.0.0"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

# --------------------------------------------------------------------------
# optional dependencies (detected at import time; overridable for testing)
# --------------------------------------------------------------------------
_NUMPY = None
if os.environ.get("ANVIL_METRICS_FORCE_PURE", "") != "1":
    try:
        import numpy as _NUMPY  # type: ignore
    except Exception:  # pragma: no cover - depends on host
        _NUMPY = None
_NUMPY_IMPORTABLE = _NUMPY is not None
# mutable switch so --selftest can exercise both paths in one process
_USE_NUMPY = _NUMPY_IMPORTABLE

_PIL_IMAGE = None
if os.environ.get("ANVIL_METRICS_FORCE_FFMPEG", "") != "1":
    try:
        from PIL import Image as _PIL_IMAGE  # type: ignore
    except Exception:  # pragma: no cover
        _PIL_IMAGE = None


class MetricsError(Exception):
    """User-facing error (bad input, unsupported file, ...)."""


# --------------------------------------------------------------------------
# image container: w, h, chs = list of flat (row-major) float channels in [0,1]
# --------------------------------------------------------------------------
class Frame:
    __slots__ = ("w", "h", "chs", "name")

    def __init__(self, w: int, h: int, chs, name: str = ""):
        self.w, self.h, self.chs, self.name = w, h, chs, name

    @property
    def nch(self) -> int:
        return len(self.chs)

    def plane_np(self, c: int):
        return _NUMPY.asarray(self.chs[c], dtype=_NUMPY.float64).reshape(self.h, self.w)


def _match_channels(fa: Frame, fb: Frame):
    """Index pairs for channel-wise comparison; a 1-channel gray frame is
    broadcast onto a multi-channel one (its channels are identical)."""
    if fa.nch == fb.nch:
        return list(range(fa.nch)), list(range(fb.nch))
    if fa.nch == 1:
        return [0] * fb.nch, list(range(fb.nch))
    if fb.nch == 1:
        return list(range(fa.nch)), [0] * fa.nch
    raise MetricsError(
        f"channel-count mismatch: A has {fa.nch}, B has {fb.nch}")


def _check_same_dims(fa: Frame, fb: Frame):
    if fa.w != fb.w or fa.h != fb.h:
        raise MetricsError(
            f"dimension mismatch: A is {fa.w}x{fa.h}, B is {fb.w}x{fb.h} "
            f"(A frame '{fa.name}' vs B frame '{fb.name}')")


def luma_list(f: Frame):
    if f.nch == 1:
        return list(f.chs[0])
    r, g, b = f.chs[0], f.chs[1], f.chs[2]
    return [0.299 * ri + 0.587 * gi + 0.114 * bi
            for ri, gi, bi in zip(r, g, b)]


def luma_np(f: Frame):
    if f.nch == 1:
        return f.plane_np(0)
    r, g, b = f.plane_np(0), f.plane_np(1), f.plane_np(2)
    return 0.299 * r + 0.587 * g + 0.114 * b


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
def _pixel_values(im):
    """Flat pixel sequence; Pillow >=13 deprecates getdata in its favour."""
    if hasattr(im, "get_flattened_data"):
        return list(im.get_flattened_data())
    return list(im.getdata())


def _pil_decode(path: str) -> Frame:
    im = _PIL_IMAGE.open(path)
    im.load()
    w, h = im.size
    mode = im.mode
    if mode in ("I;16", "I;16L", "I;16B", "I;16LE", "I;16BE", "I"):
        # 16-bit grayscale PNG (PIL exposes it as I;16* or 32-bit 'I').
        vals = _pixel_values(im)
        scale = 65535.0
        ch = [v / scale for v in vals]
        return Frame(w, h, [ch], os.path.basename(path))
    if mode != "RGB":
        im = im.convert("RGB")
    vals = _pixel_values(im)
    n = w * h
    r = [0.0] * n
    g = [0.0] * n
    b = [0.0] * n
    for i, px in enumerate(vals):
        r[i] = px[0] / 255.0
        g[i] = px[1] / 255.0
        b[i] = px[2] / 255.0
    return Frame(w, h, [r, g, b], os.path.basename(path))


def _ffprobe_geometry(path: str, ffprobe_bin: str):
    cmd = [ffprobe_bin, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,pix_fmt", "-of", "json", path]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        raise MetricsError(f"ffprobe not found at '{ffprobe_bin}'")
    if proc.returncode != 0:
        raise MetricsError(
            f"ffprobe failed on '{path}': {proc.stderr.decode('utf-8', 'replace').strip()}")
    info = json.loads(proc.stdout.decode("utf-8", "replace"))
    streams = info.get("streams") or []
    if not streams:
        raise MetricsError(f"no video stream in '{path}'")
    st = streams[0]
    w, h = int(st["width"]), int(st["height"])
    pix = st.get("pix_fmt", "")
    if w <= 0 or h <= 0:
        raise MetricsError(f"bad geometry reported for '{path}'")
    return w, h, pix


def _ffmpeg_decode(path: str, ffmpeg_bin: str, ffprobe_bin: str) -> Frame:
    w, h, pix = _ffprobe_geometry(path, ffprobe_bin)
    sixteen = "16" in pix  # e.g. rgb48le, gray16le, yuv420p16le
    fmt = "rgb48le" if sixteen else "rgb24"
    cmd = [ffmpeg_bin, "-v", "error", "-i", path, "-map", "0:v:0",
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", fmt, "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        raise MetricsError(f"ffmpeg not found at '{ffmpeg_bin}'")
    if proc.returncode != 0:
        raise MetricsError(
            f"ffmpeg failed on '{path}': {proc.stderr.decode('utf-8', 'replace').strip()}")
    itemsize = 2 if sixteen else 1
    expected = w * h * 3 * itemsize
    data = proc.stdout
    if len(data) < expected:
        raise MetricsError(
            f"ffmpeg returned {len(data)} bytes, expected {expected} "
            f"for '{path}' ({w}x{h} {fmt})")
    if sixteen:
        import array
        arr = array.array("H")
        arr.frombytes(data[:expected])
        if sys.byteorder != "little":
            arr.byteswap()  # we asked for little-endian rgb48le
        scale = 65535.0
        r = [arr[i] / scale for i in range(0, expected // 2, 3)]
        g = [arr[i] / scale for i in range(1, expected // 2, 3)]
        b = [arr[i] / scale for i in range(2, expected // 2, 3)]
        return Frame(w, h, [r, g, b], os.path.basename(path))
    scale = 255.0
    r = [data[i] / scale for i in range(0, expected, 3)]
    g = [data[i] / scale for i in range(1, expected, 3)]
    b = [data[i] / scale for i in range(2, expected, 3)]
    return Frame(w, h, [r, g, b], os.path.basename(path))


def decode(path: str, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> Frame:
    if _PIL_IMAGE is not None:
        try:
            return _pil_decode(path)
        except MetricsError:
            raise
        except Exception as e:
            # fall through to ffmpeg before giving up
            sys.stderr.write(f"note: PIL failed on '{path}' ({e}); trying ffmpeg\n")
    return _ffmpeg_decode(path, ffmpeg_bin, ffprobe_bin)


def list_frames(path: str):
    """Return the list of frame file paths for a file-or-directory input."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise MetricsError(f"input not found: '{path}'")
    names = sorted(n for n in os.listdir(path)
                   if not n.startswith(".")
                   and os.path.splitext(n)[1].lower() in IMG_EXTS)
    if not names:
        raise MetricsError(f"no .png/.jpg/.jpeg frames found in '{path}'")
    return [os.path.join(path, n) for n in names]


# --------------------------------------------------------------------------
# PSNR
# --------------------------------------------------------------------------
def psnr(fa: Frame, fb: Frame) -> float:
    _check_same_dims(fa, fb)
    ca, cb = _match_channels(fa, fb)
    se = 0.0
    n = 0
    if _USE_NUMPY:
        for a, b in zip(ca, cb):
            d = fa.plane_np(a) - fb.plane_np(b)
            se += float((d * d).sum())
            n += d.size
    else:
        for a, b in zip(ca, cb):
            for x, y in zip(fa.chs[a], fb.chs[b]):
                d = x - y
                se += d * d
                n += 1
    mse = se / n
    if mse <= 0.0:
        return math.inf
    return -10.0 * math.log10(mse)


# --------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------
_SSIM_SIZE = 11
_SSIM_SIGMA = 1.5
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def gaussian_kernel(size: int = _SSIM_SIZE, sigma: float = _SSIM_SIGMA):
    half = size // 2
    g = [math.exp(-((i - half) ** 2) / (2.0 * sigma * sigma)) for i in range(size)]
    s = sum(g)
    return [v / s for v in g]


def _filt_np(a, kern):
    """Separable 2-D 'valid' correlation with a 1-D kernel (rows then cols)."""
    k = len(kern)
    h, w = a.shape
    tw = w - k + 1
    tmp = _NUMPY.zeros((h, tw), dtype=_NUMPY.float64)
    for i in range(k):
        tmp += kern[i] * a[:, i:i + tw]
    th = h - k + 1
    out = _NUMPY.zeros((th, tw), dtype=_NUMPY.float64)
    for j in range(k):
        out += kern[j] * tmp[j:j + th, :]
    return out


def _ssim_plane_np(x, y, kern) -> float:
    mx = _filt_np(x, kern)
    my = _filt_np(y, kern)
    mxx = _filt_np(x * x, kern)
    myy = _filt_np(y * y, kern)
    mxy = _filt_np(x * y, kern)
    vx = mxx - mx * mx
    vy = myy - my * my
    cxy = mxy - mx * my
    num = (2.0 * mx * my + _C1) * (2.0 * cxy + _C2)
    den = (mx * mx + my * my + _C1) * (vx + vy + _C2)
    return float((num / den).mean())


def _filt_pure(flat, w, h, kern):
    k = len(kern)
    ow = w - k + 1
    oh = h - k + 1
    tmp = [0.0] * (ow * h)
    for y in range(h):
        rb = y * w
        tb = y * ow
        for x in range(ow):
            s = 0.0
            for i in range(k):
                s += kern[i] * flat[rb + x + i]
            tmp[tb + x] = s
    out = [0.0] * (ow * oh)
    for y in range(oh):
        ob = y * ow
        for x in range(ow):
            s = 0.0
            for j in range(k):
                s += kern[j] * tmp[(y + j) * ow + x]
            out[ob + x] = s
    return out, ow, oh


def _ssim_plane_pure(x, y, w, h, kern) -> float:
    mx, ow, oh = _filt_pure(x, w, h, kern)
    my, _, _ = _filt_pure(y, w, h, kern)
    xx = [v * v for v in x]
    yy = [v * v for v in y]
    xy = [a * b for a, b in zip(x, y)]
    mxx, _, _ = _filt_pure(xx, w, h, kern)
    myy, _, _ = _filt_pure(yy, w, h, kern)
    mxy, _, _ = _filt_pure(xy, w, h, kern)
    n = ow * oh
    total = 0.0
    for i in range(n):
        mx_i, my_i = mx[i], my[i]
        vx = mxx[i] - mx_i * mx_i
        vy = myy[i] - my_i * my_i
        cxy = mxy[i] - mx_i * my_i
        num = (2.0 * mx_i * my_i + _C1) * (2.0 * cxy + _C2)
        den = (mx_i * mx_i + my_i * my_i + _C1) * (vx + vy + _C2)
        total += num / den
    return total / n


def _ssim_plane(xa, ya, w, h, kern) -> float:
    if w < _SSIM_SIZE or h < _SSIM_SIZE:
        raise MetricsError(
            f"SSIM needs >= {_SSIM_SIZE}x{_SSIM_SIZE} px, got {w}x{h}")
    if _USE_NUMPY:
        return _ssim_plane_np(xa, ya, kern)
    return _ssim_plane_pure(xa, ya, w, h, kern)


def ssim(fa: Frame, fb: Frame, mode: str = "luma") -> float:
    _check_same_dims(fa, fb)
    w, h = fa.w, fa.h
    if w < _SSIM_SIZE or h < _SSIM_SIZE:
        raise MetricsError(
            f"SSIM needs >= {_SSIM_SIZE}x{_SSIM_SIZE} px, got {w}x{h}")
    kern = gaussian_kernel()
    if mode == "luma":
        if _USE_NUMPY:
            return _ssim_plane_np(luma_np(fa), luma_np(fb), kern)
        return _ssim_plane_pure(luma_list(fa), luma_list(fb), w, h, kern)
    ca, cb = _match_channels(fa, fb)
    vals = []
    if _USE_NUMPY:
        for a, b in zip(ca, cb):
            vals.append(_ssim_plane_np(fa.plane_np(a), fb.plane_np(b), kern))
    else:
        for a, b in zip(ca, cb):
            vals.append(_ssim_plane_pure(fa.chs[a], fb.chs[b], w, h, kern))
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------
# periodicity (lattice-artifact detector)
# --------------------------------------------------------------------------
_LATTICE_EPS = 1e-6


def default_lags(n: int):
    """(aligned, misaligned) lag sets for lattice period n (n >= 2)."""
    if n < 2:
        raise MetricsError("--periodicity N must be >= 2")
    aligned = [(n, 0), (-n, 0), (0, n), (0, -n)]
    misaligned = []
    for k in range(1, n):
        misaligned += [(k, 0), (-k, 0), (0, k), (0, -k)]
    for d in range(1, n):
        misaligned += [(d, d), (d, -d), (-d, d), (-d, -d)]
    return aligned, misaligned


def parse_lags(csv: str):
    lags = []
    for part in csv.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(",")
        if len(bits) != 2:
            raise MetricsError(f"bad lag '{part}' (expected 'dx,dy')")
        dx, dy = int(bits[0]), int(bits[1])
        if abs(dx) < 1 and abs(dy) < 1:
            raise MetricsError(f"lag ({dx},{dy}) is a zero shift")
        lags.append((dx, dy))
    if not lags:
        raise MetricsError("empty --lags set")
    # de-duplicate, keep stable order, sort for reporting
    seen = set()
    uniq = [lg for lg in lags if not (lg in seen or seen.add(lg))]
    return sorted(uniq)


def lag_mad(plane, w, h, dx, dy) -> float:
    """Mean |P(x,y) - P(x+dx,y+dy)| over the valid overlap.

    ``plane`` is a flat list (pure path) or 2-D numpy array."""
    x0, x1 = max(0, -dx), min(w, w - dx)
    y0, y1 = max(0, -dy), min(h, h - dy)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise MetricsError(
            f"image {w}x{h} too small for lag ({dx},{dy})")
    if _USE_NUMPY:
        a = plane[y0:y1, x0:x1]
        b = plane[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        return float(_NUMPY.abs(a - b).mean())
    s = 0.0
    n = 0
    for y in range(y0, y1):
        rb = y * w
        rb2 = (y + dy) * w
        for x in range(x0, x1):
            d = plane[rb + x] - plane[rb2 + x + dx]
            s += d if d >= 0.0 else -d
            n += 1
    return s / n


def periodicity(frame: Frame, lags, n: int):
    """Return ({(dx,dy): MAD}, lattice_score).  Luma domain, [0,1] units."""
    if _USE_NUMPY:
        plane = luma_np(frame)
    else:
        plane = luma_list(frame)
    mads = {}
    for (dx, dy) in lags:
        mads[(dx, dy)] = lag_mad(plane, frame.w, frame.h, dx, dy)
    aligned = [mads[lg] for lg in lags if lg[0] % n == 0 and lg[1] % n == 0]
    misaligned = [mads[lg] for lg in lags
                  if not (lg[0] % n == 0 and lg[1] % n == 0)]
    if not aligned or not misaligned:
        raise MetricsError(
            "lag set must contain at least one aligned lag "
            f"(dx,dy both multiples of {n}) and one misaligned lag")
    score = (sum(misaligned) / len(misaligned)) / (
        sum(aligned) / len(aligned) + _LATTICE_EPS)
    return mads, score


# --------------------------------------------------------------------------
# temporal
# --------------------------------------------------------------------------
def temporal_delta(fa: Frame, fb: Frame) -> float:
    """Mean |luma(A) - luma(B)| in [0,1] units (flicker proxy per pair)."""
    _check_same_dims(fa, fb)
    if _USE_NUMPY:
        d = luma_np(fa) - luma_np(fb)
        return float(_NUMPY.abs(d).mean())
    s = 0.0
    for a, b in zip(luma_list(fa), luma_list(fb)):
        d = a - b
        s += d if d >= 0.0 else -d
    return s / (fa.w * fa.h)


# --------------------------------------------------------------------------
# reference transfer (post-port color contract)
# --------------------------------------------------------------------------
# Same transform as tools/lab/ref_transfer.py; both --selftests cross-check
# parity so the two implementations cannot silently diverge.

def _bt709_oetf_inv(x):
    """BT.709 inverse OETF on one [0,1] code value -> linear (clamped)."""
    lin = x / 4.5 if x < 0.081 else ((x + 0.099) / 1.099) ** (1.0 / 0.45)
    return min(1.0, max(0.0, lin))


def _srgb_eotf_inv(lin):
    """sRGB encode on one linear value -> [0,1] code value (clamped)."""
    v = 12.92 * lin if lin <= 0.0031308 else 1.055 * lin ** (1.0 / 2.4) - 0.055
    return min(1.0, max(0.0, v))


def ref_transfer_scalar(x):
    """Reference transfer on one code value: bt709-decode -> sRGB-encode."""
    return _srgb_eotf_inv(_bt709_oetf_inv(min(1.0, max(0.0, x))))


def ref_transfer_np(c):
    """Reference transfer on a numpy array of code values."""
    c = _NUMPY.clip(c, 0.0, 1.0)
    lin = _NUMPY.where(c < 0.081, c / 4.5, ((c + 0.099) / 1.099) ** (1.0 / 0.45))
    lin = _NUMPY.clip(lin, 0.0, 1.0)
    v = _NUMPY.where(lin <= 0.0031308, 12.92 * lin,
                     1.055 * lin ** (1.0 / 2.4) - 0.055)
    return _NUMPY.clip(v, 0.0, 1.0)


def transfer_reference(f: Frame) -> Frame:
    """Copy of ``f`` with every channel mapped through the reference transfer.

    Every channel here is a color channel: decode() drops alpha (RGB
    conversion at _pil_decode / rgb24/rgb48 rawvideo, "Alpha is dropped" in
    the DECODING docstring), so unlike tools/lab/ref_transfer.py there is no
    alpha plane to preserve and mapping all channels IS the same contract."""
    if _USE_NUMPY:
        chs = [list(ref_transfer_np(f.plane_np(c)).ravel()) for c in range(f.nch)]
    else:
        chs = [[ref_transfer_scalar(v) for v in ch] for ch in f.chs]
    return Frame(f.w, f.h, chs, f.name)


# --------------------------------------------------------------------------
# viridis-like colormap + stdlib PNG writer (no external deps)
# --------------------------------------------------------------------------
# the 11 canonical viridis anchor colours (matplotlib), linearly interpolated
_VIRIDIS_ANCHORS = (
    (0.267004, 0.004874, 0.329415), (0.282623, 0.140926, 0.457517),
    (0.253935, 0.265254, 0.529983), (0.206756, 0.371758, 0.553112),
    (0.163625, 0.471133, 0.558148), (0.127568, 0.566949, 0.550556),
    (0.134692, 0.658636, 0.517649), (0.266941, 0.748751, 0.440573),
    (0.477504, 0.821444, 0.318195), (0.741388, 0.873449, 0.149586),
    (0.993248, 0.906157, 0.143936),
)


def colormap_viridis_like(t: float):
    """Piecewise-linear viridis-like colour for t in [0,1] -> (r,g,b) floats."""
    if t <= 0.0:
        return _VIRIDIS_ANCHORS[0]
    if t >= 1.0:
        return _VIRIDIS_ANCHORS[-1]
    pos = t * (len(_VIRIDIS_ANCHORS) - 1)
    i = int(pos)
    f = pos - i
    a, b = _VIRIDIS_ANCHORS[i], _VIRIDIS_ANCHORS[i + 1]
    return (a[0] + f * (b[0] - a[0]),
            a[1] + f * (b[1] - a[1]),
            a[2] + f * (b[2] - a[2]))


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_png_rgb(path: str, w: int, h: int, rows) -> None:
    """Write an 8-bit RGB PNG.  ``rows``: iterable of per-row bytes, len 3*w."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = bytearray()
    for r in rows:
        if len(r) != 3 * w:
            raise MetricsError("png writer: row length mismatch")
        raw.append(0)  # filter type 0 (None)
        raw.extend(r)
    idat = zlib.compress(bytes(raw), 6)
    with _repo_open_w(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", idat))
        f.write(_png_chunk(b"IEND", b""))


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_open_w(path, mode):
    """Repo-internal write contract: every output this harness produces lives
    inside the project tree; refuse writes resolving outside it. Containment
    for buggy invocations, not a policy change — no documented flow writes
    elsewhere."""
    ap = os.path.realpath(os.path.abspath(path))
    if not ap.startswith(_REPO_ROOT + os.sep):
        raise MetricsError(f"refusing write outside repo root: {ap}")
    return open(ap, mode)


def write_diff_heatmap(path: str, fa: Frame, fb: Frame, gain: float) -> None:
    _check_same_dims(fa, fb)
    if _USE_NUMPY:
        d = _NUMPY.abs(luma_np(fa) - luma_np(fb)) * gain
        flat = [min(1.0, float(v)) for v in d.ravel()]
    else:
        flat = [min(1.0, abs(a - b) * gain)
                for a, b in zip(luma_list(fa), luma_list(fb))]
    w, h = fa.w, fa.h
    rows = []
    cm = colormap_viridis_like
    for y in range(h):
        row = bytearray(3 * w)
        base = y * w
        for x in range(w):
            r, g, b = cm(flat[base + x])
            i = 3 * x
            row[i] = int(round(r * 255.0)) & 0xFF
            row[i + 1] = int(round(g * 255.0)) & 0xFF
            row[i + 2] = int(round(b * 255.0)) & 0xFF
        rows.append(bytes(row))
    write_png_rgb(path, w, h, rows)


# --------------------------------------------------------------------------
# aggregation + reporting
# --------------------------------------------------------------------------
def _agg(vals):
    """mean/min aggregate; inf-aware mean (any inf -> inf)."""
    mean = math.inf if any(math.isinf(v) for v in vals) else sum(vals) / len(vals)
    return {"mean": mean, "min": min(vals), "per_frame": vals}


def _jsonable(x):
    if isinstance(x, float) and math.isinf(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _fmt(x, digits=6):
    if x is None:
        return "null"
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        if math.isnan(x):
            return "nan"
        return f"{x:.{digits}f}"
    return str(x)


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def _frame_flat(w, h, value, nch=3):
    ch = [value] * (w * h)
    return Frame(w, h, [list(ch) for _ in range(nch)], "synthetic")


def _frame_fn(w, h, fn, nch=3):
    chs = [[fn(x, y) for y in range(h) for x in range(w)] for _ in range(nch)]
    return Frame(w, h, chs, "synthetic")


def _frame_from_single(w, h, values):
    return Frame(w, h, [list(values)], "synthetic")


def _noisy(base_vals, amp, rng):
    return [min(1.0, max(0.0, v + rng.uniform(-amp, amp))) for v in base_vals]


def _selftest() -> int:
    global _USE_NUMPY
    print(f"{TOOL} v{VERSION} selftest (numpy importable: "
          f"{_NUMPY_IMPORTABLE})")
    failures = []
    n_checks = 0

    def check(name, ok, detail=""):
        nonlocal n_checks
        n_checks += 1
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f"  -> {detail}"
        print(line)
        if not ok:
            failures.append(name)

    def run_core(tag):
        """Core analytic checks; runs once pure and (if possible) once numpy."""
        rng = random.Random(20261020)

        # --- PSNR: analytic, known-noise pair -----------------------------
        w, h = 64, 48
        a_code = [128] * (w * h)
        b_code = [128 + (10 if (x + y) % 2 == 0 else -10)
                  for y in range(h) for x in range(w)]
        fa = _frame_from_single(w, h, [v / 255.0 for v in a_code])
        fb = _frame_from_single(w, h, [v / 255.0 for v in b_code])
        got = psnr(fa, fb)
        want = 20.0 * math.log10(255.0 / 10.0)
        check(f"[{tag}] psnr known-noise pair == 20*log10(255/10)",
              abs(got - want) < 1e-9, f"got={got!r} want={want!r}")

        # --- PSNR: constants 0.2 vs 0.7 -> 10*log10(1/0.25) ---------------
        fa = _frame_flat(16, 16, 0.2)
        fb = _frame_flat(16, 16, 0.7)
        want = 10.0 * math.log10(4.0)
        check(f"[{tag}] psnr constant offset == 10*log10(4)",
              abs(psnr(fa, fb) - want) < 1e-9, f"got={psnr(fa, fb)!r} want={want!r}")

        # --- PSNR: identical -> inf ---------------------------------------
        fa = _frame_fn(16, 16, lambda x, y: 0.3 + 0.4 * ((x * 7 + y * 3) % 16) / 16.0)
        fb = _frame_fn(16, 16, lambda x, y: 0.3 + 0.4 * ((x * 7 + y * 3) % 16) / 16.0)
        check(f"[{tag}] psnr identical -> inf", psnr(fa, fb) == math.inf,
              f"got={psnr(fa, fb)!r}")

        # --- SSIM: identical -> 1.0 ---------------------------------------
        base = [0.5 + 0.25 * math.sin(2.0 * math.pi * x / 16.0)
                * math.cos(2.0 * math.pi * y / 12.0)
                for y in range(64) for x in range(64)]
        fb0 = _frame_from_single(64, 64, base)
        s_id = ssim(fb0, fb0)
        check(f"[{tag}] ssim identical == 1.0", abs(s_id - 1.0) < 1e-9,
              f"got={s_id!r}")

        # --- SSIM: monotone decreasing with noise amplitude ---------------
        s_prev = s_id
        mono = True
        for amp in (0.02, 0.06, 0.15):
            fn_noisy = _frame_from_single(64, 64, _noisy(base, amp, rng))
            s = ssim(fb0, fn_noisy)
            if not (0.0 < s < s_prev < 1.0 + 1e-12):
                mono = False
            s_prev = s
        check(f"[{tag}] ssim decreases monotonically with noise", mono,
              f"last={s_prev!r}")

        # --- SSIM: brute-force 2-D window cross-check (validates the -----
        #     separable gaussian implementation) on a 24x18 random plane
        kern = gaussian_kernel()
        k2d = [[kern[i] * kern[j] for i in range(11)] for j in range(11)]
        rng2 = random.Random(7)
        vals = [rng2.random() for _ in range(24 * 18)]
        valsy = [rng2.random() for _ in range(24 * 18)]
        brute = _ssim_bruteforce(vals, valsy, 24, 18, k2d)
        sep = _ssim_plane_pure(vals, valsy, 24, 18, kern)
        check(f"[{tag}] ssim separable == brute-force 2-D windows",
              abs(brute - sep) < 1e-9, f"separable={sep!r} brute={brute!r}")

        # --- periodicity: strong period-2 (2x2-cell) checkerboard ---------
        W = H = 64
        checker = [0.25 if ((x + y) % 2 == 0) else 0.75
                   for y in range(H) for x in range(W)]
        fchk = _frame_from_single(W, H, checker)
        lags = sorted(set(default_lags(2)[0] + default_lags(2)[1]))
        mads, score = periodicity(fchk, lags, 2)
        aligned_zero = all(mads[lg] == 0.0 for lg in lags
                           if lg[0] % 2 == 0 and lg[1] % 2 == 0)
        check(f"[{tag}] periodicity checkerboard aligned-lag MADs == 0",
              aligned_zero, str({lg: mads[lg] for lg in lags if
                                 lg[0] % 2 == 0 and lg[1] % 2 == 0}))
        check(f"[{tag}] periodicity checkerboard lattice_score > 100",
              score > 100.0, f"score={score!r}")

        # --- periodicity: flat -> 0 ----------------------------------------
        fflat = _frame_flat(W, H, 0.5, nch=1)
        _, s_flat = periodicity(fflat, lags, 2)
        check(f"[{tag}] periodicity flat lattice_score < 1", s_flat < 1.0,
              f"score={s_flat!r}")

        # --- periodicity: horizontal gradient, analytic-ish ~0.75 ---------
        grad_vals = [x / 31.0 for y in range(32) for x in range(32)]
        fgrad = _frame_from_single(32, 32, grad_vals)
        _, s_grad = periodicity(fgrad, lags, 2)
        check(f"[{tag}] periodicity gradient lattice_score < 2",
              0.0 < s_grad < 2.0, f"score={s_grad!r}")

        # --- periodicity: white noise ~ 1 ---------------------------------
        rng3 = random.Random(99)
        noise_vals = [rng3.random() for _ in range(32 * 32)]
        fnoise = _frame_from_single(32, 32, noise_vals)
        _, s_noise = periodicity(fnoise, lags, 2)
        check(f"[{tag}] periodicity white-noise lattice_score in (0.3, 3)",
              0.3 < s_noise < 3.0, f"score={s_noise!r}")

        # --- temporal: 3-frame ramp, deltas exactly 0.2 -------------------
        frames = [_frame_flat(8, 8, c / 255.0) for c in (51, 102, 153)]
        deltas = [temporal_delta(frames[i], frames[i + 1]) for i in range(2)]
        mean_d = sum(deltas) / len(deltas)
        check(f"[{tag}] temporal ramp deltas == 0.2 exactly",
              all(abs(d - 51.0 / 255.0) < 1e-12 for d in deltas)
              and abs(mean_d - 0.2) < 1e-12,
              f"deltas={deltas!r} mean={mean_d!r}")

        # --- reference transfer: published anchors + frame mapping --------
        d5 = _bt709_oetf_inv(0.5)
        check(f"[{tag}] ref transfer bt709 inverse-OETF mid anchor",
              abs(d5 - ((0.5 + 0.099) / 1.099) ** (1.0 / 0.45)) < 1e-15,
              f"got={d5!r}")
        check(f"[{tag}] ref transfer srgb published mid anchor "
              f"(0.21404114 lin <-> 0.5)",
              abs(_srgb_eotf_inv(0.21404114) - 0.5) < 1e-6,
              f"got={_srgb_eotf_inv(0.21404114)!r}")
        fgrad = _frame_fn(16, 16, lambda x, y: (x + y) / 45.0)
        ftrans = transfer_reference(fgrad)
        want_row = [ref_transfer_scalar(v) for v in fgrad.chs[0]]
        check(f"[{tag}] transfer_reference == per-scalar map",
              max(abs(a - b) for a, b in zip(want_row, ftrans.chs[0])) < 1e-12,
              f"n={len(ftrans.chs[0])}")
        grid8 = [k / 255.0 for k in range(256)]
        mapped = [ref_transfer_scalar(v) for v in grid8]
        check(f"[{tag}] transfer strictly monotonic on 8-bit grid",
              all(b > a for a, b in zip(mapped, mapped[1:])),
              f"code20={mapped[20]:.6f} code21={mapped[21]:.6f}")
        check(f"[{tag}] transfer endpoints 0/1 preserved",
              abs(mapped[0]) < 1e-12 and abs(mapped[-1] - 1.0) < 1e-12,
              f"0->{mapped[0]!r} 1->{mapped[-1]!r}")

    def _ssim_bruteforce(x, y, w, h, k2d):
        """Direct per-window 11x11 SSIM (reference for cross-check)."""
        ow, oh = w - 10, h - 10
        total = 0.0
        for oy in range(oh):
            for ox in range(ow):
                mx = my = mxx = myy = mxy = 0.0
                for j in range(11):
                    rb = (oy + j) * w + ox
                    for i in range(11):
                        wgt = k2d[j][i]
                        xv = x[rb + i]
                        yv = y[rb + i]
                        mx += wgt * xv
                        my += wgt * yv
                        mxx += wgt * xv * xv
                        myy += wgt * yv * yv
                        mxy += wgt * xv * yv
                vx = mxx - mx * mx
                vy = myy - my * my
                cxy = mxy - mx * my
                num = (2.0 * mx * my + _C1) * (2.0 * cxy + _C2)
                den = (mx * mx + my * my + _C1) * (vx + vy + _C2)
                total += num / den
        return total / (ow * oh)

    # ---- run core checks pure, and (if importable) under numpy ----------
    saved = _USE_NUMPY
    try:
        _USE_NUMPY = False
        run_core("pure")
        if _NUMPY_IMPORTABLE:
            _USE_NUMPY = True
            run_core("numpy")

            # ---- pure vs numpy parity on a random pair ------------------
            rng = random.Random(4242)
            w, h = 40, 30
            vals_a = [rng.random() for _ in range(w * h * 3)]
            vals_b = [min(1.0, max(0.0, v + rng.uniform(-0.1, 0.1)))
                      for v in vals_a]
            fa = Frame(w, h, [vals_a[0::3], vals_a[1::3], vals_a[2::3]], "rand")
            fb = Frame(w, h, [vals_b[0::3], vals_b[1::3], vals_b[2::3]], "rand")
            lags = sorted(set(default_lags(2)[0] + default_lags(2)[1]))

            _USE_NUMPY = False
            p_pure = psnr(fa, fb)
            s_pure = ssim(fa, fb)
            m_pure, sc_pure = periodicity(fa, lags, 2)
            _USE_NUMPY = True
            p_np = psnr(fa, fb)
            s_np = ssim(fa, fb)
            m_np, sc_np = periodicity(fa, lags, 2)
            check("parity psnr pure == numpy", abs(p_pure - p_np) < 1e-9,
                  f"pure={p_pure!r} numpy={p_np!r}")
            check("parity ssim pure == numpy", abs(s_pure - s_np) < 1e-9,
                  f"pure={s_pure!r} numpy={s_np!r}")
            check("parity periodicity pure == numpy",
                  all(abs(m_pure[lg] - m_np[lg]) < 1e-9 for lg in lags)
                  and abs(sc_pure - sc_np) < 1e-9,
                  f"pure={sc_pure!r} numpy={sc_np!r}")
    finally:
        _USE_NUMPY = saved

    # ---- parity with tools/lab/ref_transfer.py ----------------------------
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "lab"))
        import ref_transfer as lab_ref_transfer
        grid = [k / 255.0 for k in range(256)]
        mine = [ref_transfer_scalar(v) for v in grid]
        theirs = lab_ref_transfer.bt709_to_srgb(grid)
        dmax = max(abs(a - float(b)) for a, b in zip(mine, theirs))
        check("parity with tools/lab/ref_transfer.py", dmax < 1e-12,
              f"max err {dmax:.2e}")
    except Exception as e:  # pragma: no cover - sibling tool may be mid-edit
        print(f"  [skip] ref_transfer.py parity: import failed ({e})")

    # ---- colormap endpoints ----------------------------------------------
    c0 = colormap_viridis_like(0.0)
    c1 = colormap_viridis_like(1.0)
    cm0 = colormap_viridis_like(0.5)
    check("colormap endpoints match viridis anchors",
          all(abs(a - b) < 1e-9 for a, b in zip(c0, _VIRIDIS_ANCHORS[0]))
          and all(abs(a - b) < 1e-9 for a, b in zip(c1, _VIRIDIS_ANCHORS[-1])),
          f"c0={c0!r} c1={c1!r}")
    check("colormap midpoint in range", all(0.0 <= v <= 1.0 for v in cm0),
          f"cm0={cm0!r}")

    # ---- lag parser -------------------------------------------------------
    parsed = parse_lags("2,0;0,-2;1,1")
    check("lag parser '2,0;0,-2;1,1'",
          parsed == sorted([(2, 0), (0, -2), (1, 1)]), f"parsed={parsed!r}")

    # ---- PNG writer round-trip (stdlib parse-back) ------------------------
    with tempfile.TemporaryDirectory(prefix="anvil_metrics_selftest_") as td:
        path = os.path.join(td, "hm.png")
        w, h = 4, 3
        def _row(y):
            out = bytearray()
            for x in range(w):
                v = (x + y) % 256
                out += bytes((v, (v + 64) % 256, (v + 128) % 256))
            return bytes(out)
        rows = [_row(y) for y in range(h)]
        write_png_rgb(path, w, h, rows)
        with open(path, "rb") as f:
            blob = f.read()
        ok_sig = blob[:8] == b"\x89PNG\r\n\x1a\n"
        wv, hv = struct.unpack(">II", blob[16:24])
        # walk chunks, decompress IDAT, verify raw scanline stream
        pos = 8
        idat = b""
        chunks_ok = True
        while pos < len(blob):
            (clen,) = struct.unpack(">I", blob[pos:pos + 4])
            tag = blob[pos + 4:pos + 8]
            data = blob[pos + 8:pos + 8 + clen]
            (crc,) = struct.unpack(">I", blob[pos + 8 + clen:pos + 12 + clen])
            if crc != (zlib.crc32(tag + data) & 0xFFFFFFFF):
                chunks_ok = False
            if tag == b"IDAT":
                idat += data
            pos += 12 + clen
        raw = zlib.decompress(idat)
        ok_raw = (len(raw) == h * (1 + 3 * w) and raw[0] == 0
                  and list(raw[1:1 + 3 * w]) == list(rows[0]))
        check("png writer signature+IHDR", ok_sig and wv == w and hv == h,
              f"sig={ok_sig} w={wv} h={hv}")
        check("png writer chunk CRCs + IDAT scanlines", chunks_ok and ok_raw,
              f"crc_ok={chunks_ok} raw_len={len(raw)}")

        # ---- heatmap writer smoke test ----------------------------------
        fa = _frame_flat(12, 10, 0.5, nch=1)
        fb = _frame_flat(12, 10, 0.75, nch=1)
        hm = os.path.join(td, "heat.png")
        write_diff_heatmap(hm, fa, fb, 4.0)
        with open(hm, "rb") as f:
            ok_hm = f.read(8) == b"\x89PNG\r\n\x1a\n"
        check("diff heatmap writes valid PNG", ok_hm)

    print(f"selftest: {n_checks} checks, {len(failures)} failure(s)")
    if failures:
        for name in failures:
            print(f"  FAILED: {name}")
        return 1
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(
        prog="metrics.py",
        description="ANVIL offline metrics: PSNR / SSIM / lattice-periodicity "
                    "/ temporal-flicker on images and image sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "lattice score (precise definition):\n"
            "  MAD(dx,dy) = mean over the valid overlap of |Y(x,y) - Y(x+dx,y+dy)|,\n"
            "  Y = Rec.601 luma on [0,1]-normalised code values.\n"
            "  aligned lags: dx %% N == 0 and dy %% N == 0 (half-pixel-consistent\n"
            "    for the N=2 subpixel lattice; a period-N lattice maps onto\n"
            "    itself under these shifts).\n"
            "  misaligned ('odd') lags: every other lag in the set.\n"
            "  lattice_score = mean(MAD_misaligned) / (mean(MAD_aligned) + 1e-6).\n"
            "  flat content -> 0; natural content -> ~1; strong 2x2/subpixel\n"
            "  repeating-pattern artifact -> very large.  HIGHER = MORE PERIODIC\n"
            "  ARTIFACT.  Computed on A (the test output) only.\n"
            "\n"
            "exit codes: 0 ok, 1 selftest failure, 2 usage/input error."))
    p.add_argument("a", nargs="?", help="A: test image or directory of frames")
    p.add_argument("b", nargs="?", help="B: reference image or directory "
                                        "(omit with --temporal)")
    p.add_argument("--gt", action="store_true",
                   help="strict ground-truth mode: A and B must both be "
                        "directories with identical frame counts "
                        "(no broadcast)")
    p.add_argument("--legacy-raw", action="store_true",
                   help="compare the raw BT.709 reference UNTRANSFORMED "
                        "(LEGACY_RAW=1 also sets this): the report is then a "
                        "PRE-PORT-CONTRACT comparison and must never be mixed "
                        "with post-port verdicts")
    p.add_argument("--temporal", action="store_true",
                   help="single-directory mode: mean inter-frame luma delta "
                        "over consecutive pairs (flicker proxy); B must be "
                        "absent")
    p.add_argument("--periodicity", nargs="?", type=int, const=2, default=None,
                   metavar="N",
                   help="enable lattice-artifact detector with lattice period "
                        "N (default 2 when flag given without a value)")
    p.add_argument("--lags", metavar="DX,DY;DX,DY;...",
                   help="override lag set, e.g. '2,0;0,-2;1,1' "
                        "(classification still uses N)")
    p.add_argument("--metrics", default="psnr,ssim",
                   help="comma-separated pair metrics to compute "
                        "(default 'psnr,ssim')")
    p.add_argument("--ssim-mode", choices=("luma", "channels"), default="luma",
                   help="SSIM domain: Rec.601 luma (default) or "
                        "per-channel-then-mean")
    p.add_argument("--diff", nargs="?", const="./diff_heatmaps", default=None,
                   metavar="DIR",
                   help="write per-frame |luma(A)-luma(B)|*gain heatmap PNGs "
                        "to DIR (default './diff_heatmaps' when flag given "
                        "without a value)")
    p.add_argument("--gain", type=float, default=4.0, metavar="G",
                   help="heatmap amplification for --diff (default 4.0)")
    p.add_argument("--json", metavar="PATH", help="machine-readable report "
                                                  "(schema in module docstring)")
    p.add_argument("--verbose", action="store_true",
                   help="print per-frame rows")
    p.add_argument("--ffmpeg", default="ffmpeg", metavar="PATH",
                   help="ffmpeg binary (fallback decode backend)")
    p.add_argument("--ffprobe", default="ffprobe", metavar="PATH",
                   help="ffprobe binary (fallback decode backend)")
    p.add_argument("--selftest", action="store_true",
                   help="run built-in synthetic self-tests and exit")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.selftest:
        return _selftest()

    # ---- input validation ------------------------------------------------
    if args.a is None:
        sys.stderr.write("error: input A is required unless --selftest\n")
        return 2
    if args.temporal:
        if args.b is not None:
            sys.stderr.write("error: --temporal takes a single directory (A); "
                             "B must be omitted\n")
            return 2
        if not os.path.isdir(args.a):
            sys.stderr.write("error: --temporal requires A to be a directory\n")
            return 2
    else:
        if args.b is None:
            sys.stderr.write("error: input B is required "
                             "(or use --temporal / --selftest)\n")
            return 2

    want = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    for m in want:
        if m not in ("psnr", "ssim"):
            sys.stderr.write(f"error: unknown metric '{m}' in --metrics\n")
            return 2

    do_periodicity = args.periodicity is not None or args.lags is not None
    n_period = args.periodicity if args.periodicity is not None else 2
    if args.lags is not None:
        try:
            lags = parse_lags(args.lags)
        except MetricsError as e:
            sys.stderr.write(f"error: {e}\n")
            return 2
    else:
        try:
            al, mis = default_lags(n_period)
            lags = sorted(set(al + mis))
        except MetricsError as e:
            sys.stderr.write(f"error: {e}\n")
            return 2

    # ---- resolve frame lists ---------------------------------------------
    try:
        frames_a = list_frames(args.a)
        if args.temporal:
            frames_b = None
            mode = "temporal"
        else:
            frames_b = list_frames(args.b)
            a_is_dir = os.path.isdir(args.a)
            if args.gt:
                mode = "gt"
                if not (os.path.isdir(args.a) and os.path.isdir(args.b)):
                    raise MetricsError("--gt requires A and B to be directories")
                if len(frames_a) != len(frames_b):
                    raise MetricsError(
                        f"--gt frame-count mismatch: A has {len(frames_a)}, "
                        f"B has {len(frames_b)}")
            elif a_is_dir:
                # sequence comparison; broadcast a single reference frame
                # (a lone file, or a reference directory holding one image)
                # across all A frames when counts differ
                if len(frames_b) == 1 and len(frames_a) > 1:
                    sys.stderr.write(
                        f"note: reference has a single frame; broadcasting it "
                        f"to all {len(frames_a)} A frames\n")
                    frames_b = frames_b * len(frames_a)
                elif len(frames_a) != len(frames_b):
                    raise MetricsError(
                        f"frame-count mismatch: A has {len(frames_a)}, "
                        f"B has {len(frames_b)} (use --gt for strict GT "
                        f"semantics, or a single reference image to broadcast)")
                mode = "sequence"
            else:
                mode = "pair"
    except MetricsError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    backend = "PIL" if _PIL_IMAGE is not None else "ffmpeg"

    # ---- reference transfer contract --------------------------------------
    legacy_raw = args.legacy_raw or os.environ.get("LEGACY_RAW", "") == "1"
    if mode == "temporal":
        ref_contract = "n/a (temporal mode has no reference)"
        apply_ref_transfer = False
    elif legacy_raw:
        ref_contract = ("LEGACY-RAW pre-port-contract comparison "
                        "(ref not transferred)")
        apply_ref_transfer = False
    else:
        ref_contract = "post-port: ref bt709-decode -> sRGB-encode"
        apply_ref_transfer = True

    # ---- compute ----------------------------------------------------------
    per_psnr = [] if "psnr" in want else None
    per_ssim = [] if "ssim" in want else None
    per_lattice = [] if do_periodicity else None
    per_lag_mads = {lg: [] for lg in lags} if do_periodicity else None
    per_pair_delta = [] if args.temporal else None
    written_heatmaps = []

    try:
        if args.temporal:
            prev = None
            for p in frames_a:
                cur = decode(p, args.ffmpeg, args.ffprobe)
                if prev is not None:
                    per_pair_delta.append(temporal_delta(prev, cur))
                prev = cur
        else:
            if args.diff is not None:
                os.makedirs(args.diff, exist_ok=True)
            for i, (pa, pb) in enumerate(zip(frames_a, frames_b)):
                fa = decode(pa, args.ffmpeg, args.ffprobe)
                fb = decode(pb, args.ffmpeg, args.ffprobe)
                if apply_ref_transfer:
                    fb = transfer_reference(fb)
                if per_psnr is not None:
                    per_psnr.append(psnr(fa, fb))
                if per_ssim is not None:
                    per_ssim.append(ssim(fa, fb, args.ssim_mode))
                if do_periodicity:
                    mads, score = periodicity(fa, lags, n_period)
                    per_lattice.append(score)
                    for lg in lags:
                        per_lag_mads[lg].append(mads[lg])
                if args.diff is not None:
                    hm = os.path.join(args.diff, f"diff_{i:04d}.png")
                    write_diff_heatmap(hm, fa, fb, args.gain)
                    written_heatmaps.append(hm)
    except MetricsError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2

    # ---- human table ------------------------------------------------------
    out = sys.stdout
    print("=" * 72, file=out)
    print(f" {TOOL} v{VERSION}", file=out)
    print(f" decode backend : {backend}   numpy: "
          f"{'yes' if _USE_NUMPY else 'no (pure-python fallback)'}", file=out)
    print(f" mode           : {mode}   A={args.a}"
          + (f"   B={args.b}" if args.b else ""), file=out)
    print(f" ref contract   : {ref_contract}", file=out)
    print("-" * 72, file=out)

    if not args.temporal:
        rows = []
        if per_psnr is not None:
            agg = _agg(per_psnr)
            rows.append(("psnr_db", _fmt(agg["mean"]), _fmt(agg["min"])))
        if per_ssim is not None:
            agg = _agg(per_ssim)
            rows.append(("ssim", _fmt(agg["mean"]), _fmt(agg["min"])))
        if per_lattice is not None:
            agg = _agg(per_lattice)
            rows.append(("lattice_score", _fmt(agg["mean"]), _fmt(agg["min"])))
        w1 = max(len(r[0]) for r in rows) if rows else 10
        print(f" {'metric'.ljust(w1)}   {'mean':>14}   {'min':>14}", file=out)
        for name, mean, mn in rows:
            print(f" {name.ljust(w1)}   {mean:>14}   {mn:>14}", file=out)
        if per_lattice is not None:
            print(f"   per-lag mean MAD (class):", file=out)
            for lg in lags:
                cls = "aligned" if (lg[0] % n_period == 0
                                    and lg[1] % n_period == 0) else "misaligned"
                m = sum(per_lag_mads[lg]) / len(per_lag_mads[lg])
                print(f"     lag({lg[0]:+d},{lg[1]:+d}) {cls:<10} "
                      f"mad_mean {m:.6f}", file=out)
        if args.verbose:
            print("-" * 72, file=out)
            print(" per-frame:", file=out)
            hdr = "  frame"
            if per_psnr is not None:
                hdr += "       psnr_db"
            if per_ssim is not None:
                hdr += "         ssim"
            if per_lattice is not None:
                hdr += "   lattice"
            print(hdr, file=out)
            for i in range(len(frames_a)):
                line = f"  {i:04d}"
                if per_psnr is not None:
                    line += f"  {_fmt(per_psnr[i]):>12}"
                if per_ssim is not None:
                    line += f"  {_fmt(per_ssim[i]):>12}"
                if per_lattice is not None:
                    line += f"  {_fmt(per_lattice[i]):>10}"
                line += f"   {os.path.basename(frames_a[i])}"
                print(line, file=out)
    else:
        mean_d = sum(per_pair_delta) / len(per_pair_delta)
        print(f" temporal frames          : {len(frames_a)}", file=out)
        print(f" temporal pairs           : {len(per_pair_delta)}", file=out)
        print(f" temporal mean_delta      : {mean_d:.6f}", file=out)
        if args.verbose:
            print("-" * 72, file=out)
            for i, d in enumerate(per_pair_delta):
                print(f"  pair {i:04d}: {os.path.basename(frames_a[i])} -> "
                      f"{os.path.basename(frames_a[i + 1])}  delta {d:.6f}",
                      file=out)

    if written_heatmaps:
        print("-" * 72, file=out)
        print(f" wrote {len(written_heatmaps)} heatmap(s) to "
              f"{args.diff} (gain {args.gain})", file=out)
    print("=" * 72, file=out)

    # ---- JSON -------------------------------------------------------------
    if args.json:
        report = {
            "tool": TOOL,
            "version": VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "decode": {"backend": backend, "numpy_used": _USE_NUMPY},
            "config": {
                "mode": mode,
                "metrics": want,
                "ssim_mode": args.ssim_mode,
                "periodicity_n": n_period if do_periodicity else None,
                "lags": [list(lg) for lg in lags] if do_periodicity else [],
                "lag_classes": {f"{dx},{dy}":
                                ("aligned" if dx % n_period == 0
                                 and dy % n_period == 0 else "misaligned")
                                for (dx, dy) in lags} if do_periodicity else {},
                "ref_contract": ref_contract,
                "ref_transfer_applied": apply_ref_transfer,
                "gain": args.gain,
                "diff_dir": args.diff,
            },
            "inputs": {
                "a": args.a,
                "b": args.b,
                "frames_a": frames_a,
                "frames_b": frames_b,
            },
            "comparison": None,
            "temporal": None,
        }
        if not args.temporal:
            comparison = {"num_frames": len(frames_a)}
            if per_psnr is not None:
                agg = _agg(per_psnr)
                agg["all_identical"] = any(math.isinf(v) for v in per_psnr)
                comparison["psnr"] = agg
            if per_ssim is not None:
                comparison["ssim"] = _agg(per_ssim)
            if do_periodicity:
                comparison["periodicity"] = {
                    "lattice_score": _agg(per_lattice),
                    "lags": [
                        {"dx": lg[0], "dy": lg[1],
                         "class": ("aligned" if lg[0] % n_period == 0
                                   and lg[1] % n_period == 0 else "misaligned"),
                         "mad_mean": sum(per_lag_mads[lg]) / len(per_lag_mads[lg]),
                         "mad_min": min(per_lag_mads[lg]),
                         "mad_per_frame": per_lag_mads[lg]}
                        for lg in lags],
                }
            report["comparison"] = comparison
        else:
            report["temporal"] = {
                "num_frames": len(frames_a),
                "num_pairs": len(per_pair_delta),
                "mean_delta": (sum(per_pair_delta) / len(per_pair_delta)
                               if per_pair_delta else None),
                "per_pair": per_pair_delta,
            }
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _repo_open_w(args.json, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
            f.write("\n")
        print(f"json report written to {args.json}", file=out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
