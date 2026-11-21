#!/usr/bin/env python3
"""FORGE data builder — degraded-720p / pristine-2160p packed tile shards.

Spec: docs/design/FORGE-NEW-MODEL.md §4 (+ §6 amendments).

  sources.json  : [{"id", "path", "split": "train"|"val", "license",
                    "url", "attribution", "max_seconds"?, "start"?}, ...]
  encode        : per (source, chain) one ffmpeg pass -> 1280x720 yuv420p
                  encode in <out>/enc/ (the degradation is REAL codec damage)
  tiles         : stream LR encode + 4K master in lockstep, every
                  --stride-th frame cut --tiles-per-frame random tiles:
                    lr  float16 [N,3,160,160]  linear light, the player's
                        front end (forge_player_io.yuv420_to_linear) on the
                        decoded LR; 128 core + 16 px margin each side,
                        replicate-padded at frame borders (== the scaler's
                        clamp-to-edge)
                    gt  uint8   [N,384,384,3]  sRGB8 of gt_linear(master)
                        for the 128 core (3x) — exactly the metric's GT
                  shard dir <out>/shards/<source>__<chain>/ written to a
                  .tmp dir then os.replace'd: an interrupted build never
                  leaves a half shard, and reruns skip finished shards.

Usage:
  forge_data.py build  sources.json --out .scratch/forge/data [--jobs 4]
  forge_data.py verify sources.json --out ...   (alignment check only)
  forge_data.py synth  --out .scratch/forge/smoke-data   (tiny synthetic
                  shard set for trainer smoke tests)
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_player_io import (gt_linear_from_yuv, linear_to_srgb8,  # noqa: E402
                             player_base, yuv420_to_linear)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR_W, LR_H, SCALE = 1280, 720, 3
GT_W, GT_H = LR_W * SCALE, LR_H * SCALE
CORE, MARGIN = 128, 16
LR_T = CORE + 2 * MARGIN

# GT normalization: any master -> exactly 3840x2160 (cover + centre crop).
PREFIX = (f"scale={GT_W}:{GT_H}:force_original_aspect_ratio=increase:"
          f"flags=lanczos+accurate_rnd,crop={GT_W}:{GT_H},setsar=1")


def _down(flt, w=LR_W, h=LR_H):
    # +accurate_rnd: without it swscale's 10->8-bit path lifts every plane
    # ~+0.3 codes (measured; 2-3 sRGB codes in dark scenes) — a pipeline DC
    # bias the net would learn as a free "gain". With it: <=0.007 codes.
    return f"scale={w}:{h}:flags={flt}+accurate_rnd"


X264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
# The degradation catalog. Each: (720p filter chain, encoder args) or a
# two-stage chain (degrade -> poor upscale -> re-encode). Tag names are the
# per-chain breakdown keys in eval.
CHAINS = {
    "c10_lanczos":  (_down("lanczos"), X264 + ["-crf", "10"]),
    "c18_bicubic":  (_down("bicubic"), X264 + ["-crf", "18"]),
    "c23_lanczos":  (_down("lanczos"), X264 + ["-crf", "23"]),
    "c23_area":     (_down("area"), X264 + ["-crf", "23"]),
    "c28_bilinear": (_down("bilinear"), X264 + ["-crf", "28"]),
    "x265_28":      (_down("lanczos"), ["-c:v", "libx265", "-preset", "medium",
                                         "-crf", "28", "-pix_fmt", "yuv420p",
                                         "-x265-params", "log-level=error"]),
    "av1_40":       (_down("lanczos"), ["-c:v", "libsvtav1", "-preset", "8",
                                         "-crf", "40", "-pix_fmt", "yuv420p"]),
    "noisy10":      (_down("lanczos") + ",noise=alls=20:allf=t+u",
                     X264 + ["-crf", "10"]),
    "noisy23":      (_down("lanczos") + ",noise=alls=20:allf=t+u",
                     X264 + ["-crf", "23"]),
    # degrade -> poorly upscale -> re-encode (user-commissioned chain)
    "reup":         ("TWO-STAGE", None),
}


def _guard(p):
    ap = os.path.realpath(os.path.abspath(p))
    if not ap.startswith(REPO + os.sep):
        raise ValueError(f"refusing write outside repo root: {ap}")
    return ap


def _seek_args(src):
    a = []
    if src.get("start"):
        a += ["-ss", str(src["start"])]
    return a


def _dur_args(src):
    return ["-t", str(src["max_seconds"])] if src.get("max_seconds") else []


def encode(src, chain, out):
    """One (source, chain) 720p encode; atomic, skip if done."""
    dst = _guard(os.path.join(out, "enc", f"{src['id']}__{chain}.mkv"))
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp.mkv"
    base = ["ffmpeg", "-v", "error", "-y"] + _seek_args(src) + ["-i", src["path"]] \
        + _dur_args(src) + ["-an", "-sn", "-fps_mode", "passthrough"]
    if chain == "reup":
        mid = tmp + ".mid.mkv"
        subprocess.run(base + ["-vf", PREFIX + "," + _down("bicubic", 640, 360)]
                       + X264 + ["-crf", "26", mid], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mid, "-fps_mode",
                        "passthrough", "-vf", _down("bilinear")]
                       + X264 + ["-crf", "23", tmp], check=True)
        os.remove(mid)
    else:
        vf, enc = CHAINS[chain]
        subprocess.run(base + ["-vf", PREFIX + "," + vf] + enc + [tmp], check=True)
    os.replace(tmp, dst)
    return dst


def _sel(stride, offset):
    """ffmpeg-side frame selection (decode-order n): only the sampled frames
    pay for colour conversion / piping."""
    return [] if stride <= 1 else ["-vf", f"select=eq(mod(n\\,{stride})\\,{offset})"]


def _yuv_stream(path, w=LR_W, h=LR_H, stride=1, offset=0):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path] + _sel(stride, offset)
                         + ["-fps_mode", "passthrough", "-f", "rawvideo",
                            "-pix_fmt", "yuv420p", "-"],
                         stdout=subprocess.PIPE)
    ys, cs = w * h, (w // 2) * (h // 2)
    try:
        while True:
            b = p.stdout.read(ys + 2 * cs)
            if len(b) < ys + 2 * cs:
                return
            a = np.frombuffer(b, np.uint8)
            yield (a[:ys].reshape(h, w), a[ys:ys + cs].reshape(h // 2, w // 2),
                   a[ys + cs:].reshape(h // 2, w // 2))
    finally:
        p.kill()
        p.wait()


_FMT = {  # native mezz/master pix_fmt -> (bits, chroma w div, chroma h div)
    "yuv420p": (8, 2, 2), "yuv422p": (8, 2, 1), "yuv444p": (8, 1, 1),
    "yuv420p10le": (10, 2, 2), "yuv422p10le": (10, 2, 1), "yuv444p10le": (10, 1, 1),
}


def _gt_stream(src, stride=1, offset=0):
    """Native YUV planes of the 3840x2160 master/mezz (no swscale colour
    math — see forge_player_io.gt_linear_from_yuv). Yields (y,u,v,bits)."""
    w, h, pf = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt,width,height", "-of", "csv=p=0", src["path"]],
        check=True, capture_output=True, text=True).stdout.strip().split(",")[:3]
    w, h = int(w), int(h)
    if (w, h) != (GT_W, GT_H) or pf not in _FMT:
        raise RuntimeError(f"GT source must be {GT_W}x{GT_H} in {list(_FMT)}: {pf} {w}x{h}")
    bits, dx, dy = _FMT[pf]
    bpe = 1 if bits == 8 else 2
    dt = np.uint8 if bits == 8 else np.uint16
    ys, cs = w * h, (w // dx) * (h // dy)
    n = (ys + 2 * cs) * bpe
    p = subprocess.Popen(["ffmpeg", "-v", "error"] + _seek_args(src) + ["-i", src["path"]]
                         + _dur_args(src) + _sel(stride, offset)
                         + ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pf, "-"],
                         stdout=subprocess.PIPE)
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                return
            a = np.frombuffer(b, dt)
            yield (a[:ys].reshape(h, w), a[ys:ys + cs].reshape(h // dy, w // dx),
                   a[ys + cs:].reshape(h // dy, w // dx), bits)
    finally:
        p.kill()
        p.wait()


def _gt_lin(planes):
    y, u, v, bits = planes
    t = lambda x: torch.from_numpy(x.astype(np.int32))
    return gt_linear_from_yuv(t(y), t(u), t(v), bits)


def _psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def verify_alignment(src, enc_path, probe=6):
    """LR frame k must match GT frame k better than k+-1 (lanczos base vs
    GT, PSNR). Catches encoder frame drops / offset starts before a single
    tile is cut. Returns (ok, table)."""
    lrs = []
    for i, (y, u, v) in enumerate(_yuv_stream(enc_path)):
        if i >= probe:
            break
        lin = yuv420_to_linear(torch.from_numpy(y.copy()), torch.from_numpy(u.copy()),
                               torch.from_numpy(v.copy()))
        lrs.append(linear_to_srgb8(player_base(lin, SCALE)).numpy())
    gts = []
    for i, g in enumerate(_gt_stream(src)):
        if i >= probe:
            break
        gts.append(linear_to_srgb8(_gt_lin(g)).numpy())
    rows, losses = [], 0
    for k in range(1, min(len(lrs), len(gts)) - 1):
        s = [float(_psnr(lrs[k], gts[k + d])) for d in (-1, 0, 1)]
        rows.append((k, [round(x, 2) for x in s]))
        # near-static content ties within noise (sintel_cave intro: 0.06 dB);
        # a real offset on moving content loses by dB (sol_levante: 8-12 dB)
        if max(s[0], s[2]) > s[1] + 0.25:
            losses += 1
    return losses * 2 < max(1, len(rows)), rows


def _pad_crop(lin, y0, x0):
    """LR tile with margin; replicate pad == clamp-to-edge."""
    h, w = lin.shape[:2]
    ys = np.clip(np.arange(y0 - MARGIN, y0 + CORE + MARGIN), 0, h - 1)
    xs = np.clip(np.arange(x0 - MARGIN, x0 + CORE + MARGIN), 0, w - 1)
    return lin[ys][:, xs]


def build_shard(src, chain, out, stride, tpf, seed):
    name = f"{src['id']}__{chain}"
    final = _guard(os.path.join(out, "shards", name))
    if os.path.exists(os.path.join(final, "done.json")):
        return name, "skip"
    enc = encode(src, chain, out)
    ok, rows = verify_alignment(src, enc)
    if not ok:
        return name, f"ALIGN-FAIL {rows}"
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    rng = random.Random(f"{seed}:{name}")
    # each chain samples a DIFFERENT frame phase: more distinct content per
    # tile budget than every chain re-cutting the same frames
    offset = list(CHAINS).index(chain) % stride
    lr_list, gt_list, meta = [], [], []
    for k, ((y, u, v), g) in enumerate(zip(_yuv_stream(enc, stride=stride, offset=offset),
                                           _gt_stream(src, stride, offset))):
        i = offset + k * stride
        lin = yuv420_to_linear(torch.from_numpy(y.copy()), torch.from_numpy(u.copy()),
                               torch.from_numpy(v.copy())).numpy()
        gtf = None
        for _ in range(tpf):
            ty = rng.randrange(0, LR_H - CORE + 1)
            tx = rng.randrange(0, LR_W - CORE + 1)
            lr_list.append(_pad_crop(lin, ty, tx).transpose(2, 0, 1).astype(np.float16))
            if gtf is None:
                gtf = linear_to_srgb8(_gt_lin(g)).numpy()   # once per frame
            gt_list.append(gtf[ty * SCALE:(ty + CORE) * SCALE, tx * SCALE:(tx + CORE) * SCALE].copy())
            meta.append({"frame": i, "y": ty, "x": tx})
    if not lr_list:
        shutil.rmtree(tmp)
        return name, "EMPTY"
    np.save(os.path.join(tmp, "lr.npy"), np.stack(lr_list))
    np.save(os.path.join(tmp, "gt.npy"), np.stack(gt_list))
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"source": src["id"], "chain": chain, "split": src.get("split", "train"),
                   "license": src.get("license"), "url": src.get("url"),
                   "attribution": src.get("attribution"), "stride": stride,
                   "tiles_per_frame": tpf, "lr_tile": LR_T, "margin": MARGIN,
                   "core": CORE, "scale": SCALE, "align": rows, "tiles": meta}, f)
    with open(os.path.join(tmp, "done.json"), "w") as f:
        json.dump({"n": len(lr_list)}, f)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return name, f"ok n={len(lr_list)}"


def _job(args):
    torch.set_num_threads(2)
    return build_shard(*args)


def cmd_build(a):
    srcs = json.load(open(a.sources))
    chains = a.chains.split(",") if a.chains else list(CHAINS)
    jobs = [(s, c, a.out, a.stride, a.tiles_per_frame, a.seed)
            for s in srcs for c in chains]
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, status in ex.map(_job, jobs):
            print(f"{name}: {status}", flush=True)


def cmd_verify(a):
    for s in json.load(open(a.sources)):
        enc = encode(s, "c10_lanczos", a.out)
        print(s["id"], verify_alignment(s, enc))


def cmd_synth(a):
    """Tiny synthetic shard set (smooth random fields; LR = box-down of GT
    + noise) — trainer plumbing smoke only, not a quality signal."""
    rng = np.random.default_rng(0)
    for k, split in enumerate(["train", "train", "val"]):
        d = _guard(os.path.join(a.out, "shards", f"synth{k}__smoke"))
        os.makedirs(d, exist_ok=True)
        n = 64
        g = torch.from_numpy(rng.random((n, 3, 48, 48)).astype(np.float32))
        g = torch.nn.functional.interpolate(g, size=(LR_T * SCALE,) * 2, mode="bicubic",
                                            align_corners=False).clamp(0, 1)
        lr = torch.nn.functional.avg_pool2d(g, SCALE) + 0.02 * torch.randn(n, 3, LR_T, LR_T)
        gt = g[:, :, MARGIN * SCALE:(MARGIN + CORE) * SCALE, MARGIN * SCALE:(MARGIN + CORE) * SCALE]
        np.save(os.path.join(d, "lr.npy"), lr.clamp(0, 1).numpy().astype(np.float16))
        np.save(os.path.join(d, "gt.npy"),
                linear_to_srgb8(gt.permute(0, 2, 3, 1)).numpy())
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"source": f"synth{k}", "chain": "smoke", "split": split,
                       "lr_tile": LR_T, "margin": MARGIN, "core": CORE, "scale": SCALE}, f)
        with open(os.path.join(d, "done.json"), "w") as f:
            json.dump({"n": n}, f)
    print("synth shards written:", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("sources")
    b.add_argument("--out", required=True)
    b.add_argument("--chains")
    b.add_argument("--stride", type=int, default=12)
    b.add_argument("--tiles-per-frame", type=int, default=8)
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("--seed", type=int, default=1949)
    v = sub.add_parser("verify")
    v.add_argument("sources")
    v.add_argument("--out", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "verify": cmd_verify, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    main()
