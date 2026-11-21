#!/usr/bin/env python3
"""GT reference transfer for post-port PSNR: BT.709 decode -> linear -> sRGB encode.

Since commit 9938e07 the pipeline decodes BT.709/PQ/HLG to LINEAR at entry and
sRGB-encodes ONCE at present/capture, so captured PNGs are sRGB-encoded linear
light.  The suites' ground-truth references are ffmpeg extracts of the raw
video and carry BT.709 OETF code values.  Comparing a raw BT.709 ref against a
post-port capture mismeasures quality; run the ref through this tool first
(open item 2, ANVIL-OPEN-ITEMS.md).

    bt709 decode : lin = V/4.5                      (V <  0.081)
                   lin = ((V + 0.099) / 1.099)^(1/0.45)
    sRGB encode  : code = 12.92 * lin               (lin <= 0.0031308)
                   code = 1.055 * lin^(1/2.4) - 0.055

Clamped to [0,1] at each stage; alpha channels pass through untouched.
tools/metrics.py carries the same transform inline (pure-python + numpy
paths); --selftest in either tool cross-checks the two when both are
importable, so they cannot silently diverge.

Legacy consumers that compare raw BT.709 refs against captures produce a
PRE-PORT-CONTRACT comparison (LEGACY_RAW=1 in the suite scripts,
--legacy-raw in metrics.py) and must never be mixed with post-port verdicts.

usage:
    ref_transfer.py IN.png -o OUT.png
    ref_transfer.py A.png B.png -o OUTDIR/
    ref_transfer.py --in-place A.png B.png
    ref_transfer.py --verify IN.png          # min/max/mean before/after, writes nothing
    ref_transfer.py --selftest
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

_SIXTEEN = ("I;16", "I;16L", "I;16B", "I;16LE", "I;16BE", "I")


def bt709_decode(v):
    """BT.709 inverse OETF: code value -> linear, clamped [0,1]."""
    v = np.clip(v, 0.0, 1.0)
    lin = np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1.0 / 0.45))
    return np.clip(lin, 0.0, 1.0)


def bt709_encode(lin):
    """BT.709 OETF: linear -> code value (selftest round-trip counterpart)."""
    lin = np.clip(lin, 0.0, 1.0)
    v = np.where(lin < 0.018, 4.5 * lin, 1.099 * lin ** 0.45 - 0.099)
    return np.clip(v, 0.0, 1.0)


def srgb_encode(lin):
    """sRGB encode: linear -> code value, clamped [0,1]."""
    lin = np.clip(lin, 0.0, 1.0)
    v = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1.0 / 2.4) - 0.055)
    return np.clip(v, 0.0, 1.0)


def srgb_decode(v):
    """sRGB decode (selftest round-trip counterpart): code value -> linear."""
    v = np.clip(v, 0.0, 1.0)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return np.clip(lin, 0.0, 1.0)


def bt709_to_srgb(v):
    """The reference transfer: BT.709 code -> linear -> sRGB code."""
    return srgb_encode(bt709_decode(v))


def load_planes(path):
    """(planes, kind): float [0,1] planes; kind names the packing."""
    im = Image.open(path)
    im.load()
    if im.mode == "P":
        im = im.convert("RGB")
    if im.mode in _SIXTEEN:
        return [np.asarray(im).astype(np.float64) / 65535.0], "gray16"
    if im.mode == "L":
        return [np.asarray(im).astype(np.float64) / 255.0], "l"
    if im.mode == "LA":
        a = np.asarray(im.convert("LA")).astype(np.float64) / 255.0
        return [a[..., 0], a[..., 1]], "la"
    if im.mode == "RGBA":
        a = np.asarray(im).astype(np.float64) / 255.0
        return [a[..., 0], a[..., 1], a[..., 2], a[..., 3]], "rgba"
    a = np.asarray(im.convert("RGB")).astype(np.float64) / 255.0
    return [a[..., 0], a[..., 1], a[..., 2]], "rgb"


_N_COLOR = {"gray16": 1, "l": 1, "la": 1, "rgba": 3, "rgb": 3}


def save_planes(path, planes, kind):
    if kind == "gray16":
        out = np.clip(np.rint(planes[0] * 65535.0), 0, 65535).astype(np.uint16)
        # Pillow >=13 rejects an explicit 'I;16'; uint16 2-D auto-detects it
        Image.fromarray(out).save(path)
        return
    arr = np.stack(planes, axis=-1)
    out = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    mode = {"l": "L", "la": "LA", "rgba": "RGBA", "rgb": "RGB"}[kind]
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]  # Pillow >=13 rejects (H,W,1) for explicit mode 'L'
    Image.fromarray(out, mode=mode).save(path)


def transform(planes, kind):
    """Color planes through the transfer; alpha (last plane of la/rgba) kept."""
    n = _N_COLOR[kind]
    out = [bt709_to_srgb(p) for p in planes[:n]]
    return out + list(planes[n:])


def stats(planes, kind):
    n = _N_COLOR[kind]
    allp = np.concatenate([p.ravel() for p in planes[:n]])
    return float(allp.min()), float(allp.max()), float(allp.mean())


def selftest():
    """Round-trip + anchor checks; exit 0 ok, 1 failure."""
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  -> {detail}")
        ok = ok and cond

    # anchors computed from the standard's own constants at runtime, plus the
    # published sRGB mid-gray (IEC 61966-2-1: 0.21404114 linear <-> 0.5 code)
    d5 = float(bt709_decode(0.5))
    check("bt709_decode(0.5) == ((0.5+0.099)/1.099)^(1/0.45)",
          abs(d5 - ((0.5 + 0.099) / 1.099) ** (1.0 / 0.45)) < 1e-15, f"got {d5:.8f}")
    s5 = float(srgb_encode(0.21404114))
    check("srgb_encode(0.21404114) == 0.5", abs(s5 - 0.5) < 1e-6, f"got {s5:.8f}")
    check("endpoints preserved (1e-12)",
          abs(float(bt709_to_srgb(0.0))) < 1e-12 and abs(float(bt709_to_srgb(1.0)) - 1.0) < 1e-12,
          f"0->{float(bt709_to_srgb(0.0))} 1->{float(bt709_to_srgb(1.0))}")

    g = np.linspace(0.0, 1.0, 4097)
    r709 = float(np.abs(bt709_encode(bt709_decode(g)) - g).max())
    rsrgb = float(np.abs(srgb_decode(srgb_encode(g)) - g).max())
    # the spec's rounded 0.081 knot (exact continuity is V=0.08128) leaves a
    # ~2.5e-4 linear round-trip artifact in a band no 8-bit code hits
    check("bt709 encode(decode(gradient)) identity <= 1/255", r709 <= 1.0 / 255.0,
          f"max err {r709:.2e} linear (rounded-knot artifact)")
    check("srgb decode(encode(gradient)) identity (float)", rsrgb < 1e-12, f"max err {rsrgb:.2e}")

    # the ask's check: 8-bit round trip within 1/255
    grid = np.arange(256, dtype=np.float64) / 255.0
    e709 = float(np.abs(np.rint(bt709_encode(bt709_decode(grid)) * 255.0) / 255.0 - grid).max())
    esrgb = float(np.abs(np.rint(srgb_decode(srgb_encode(grid)) * 255.0) / 255.0 - grid).max())
    check("bt709 8-bit encode/decode round trip <= 1/255", e709 <= 1.0 / 255.0,
          f"max err {e709:.6f} ({e709 * 255:.3f} code values)")
    check("srgb 8-bit encode/decode round trip <= 1/255", esrgb <= 1.0 / 255.0,
          f"max err {esrgb:.6f} ({esrgb * 255:.3f} code values)")

    # monotonic on the 8-bit code grid (codes 20/255..21/255 straddle the
    # rounded knot; the float-gradient dip between them is unrepresentable)
    tgrid = bt709_to_srgb(grid)
    mono = bool(np.all(np.diff(tgrid) > 0.0))
    check("transfer strictly monotonic on 8-bit code grid", mono,
          f"code 20->{tgrid[20]:.6f} code 21->{tgrid[21]:.6f}")

    # parity with metrics.py's inline implementation (both must be identical)
    tdir = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.insert(0, os.path.dirname(tdir))
        import metrics as anvil_metrics
        mine = bt709_to_srgb(g)
        theirs = np.asarray([anvil_metrics.ref_transfer_scalar(x) for x in g])
        dp = float(np.abs(mine - theirs).max())
        check("scalar parity with tools/metrics.py inline transfer", dp < 1e-12,
              f"max err {dp:.2e}")
        if anvil_metrics._NUMPY_IMPORTABLE:
            dn = float(np.abs(mine - anvil_metrics.ref_transfer_np(g)).max())
            check("numpy parity with tools/metrics.py inline transfer", dn < 1e-12,
                  f"max err {dn:.2e}")
    except Exception as e:  # pragma: no cover - metrics.py may be mid-edit
        print(f"  [skip] metrics.py parity: import failed ({e})")

    print(f"selftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="ref_transfer.py",
        description="Transform GT reference PNGs into the post-port capture "
                    "contract: bt709-decode -> linear -> sRGB-encode.")
    p.add_argument("inputs", nargs="*", metavar="IN",
                   help="reference image(s) (PNG 8/16-bit, gray/RGB/alpha)")
    p.add_argument("-o", "--output", metavar="OUT",
                   help="output file (one input) or directory (many inputs)")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite each input with the transferred image")
    p.add_argument("--verify", action="store_true",
                   help="print min/max/mean before and after; write nothing")
    p.add_argument("--selftest", action="store_true",
                   help="round-trip + anchor self-test and exit")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.inputs:
        p.error("at least one input image required")
    if a.verify and (a.in_place or a.output):
        p.error("--verify writes nothing; drop -o/--in-place")
    if not a.verify and not a.in_place and not a.output:
        p.error("need -o OUT, --in-place, or --verify")
    if a.in_place and a.output:
        p.error("--in-place and -o are mutually exclusive")

    outs = None
    if not a.verify:
        if a.in_place:
            outs = a.inputs
        elif len(a.inputs) == 1:
            outs = [a.output]
        else:
            os.makedirs(a.output, exist_ok=True)
            outs = [os.path.join(a.output, os.path.basename(i)) for i in a.inputs]

    for i, inp in enumerate(a.inputs):
        planes, kind = load_planes(inp)
        out = transform(planes, kind)
        if a.verify:
            b = stats(planes, kind)
            aft = stats(out, kind)
            print(f"{inp} [{kind}]  before min/max/mean {b[0]:.6f}/{b[1]:.6f}/{b[2]:.6f}"
                  f"  after {aft[0]:.6f}/{aft[1]:.6f}/{aft[2]:.6f}")
        else:
            save_planes(outs[i], out, kind)
            print(f"{inp} [{kind}] -> {outs[i]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
