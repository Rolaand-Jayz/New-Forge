#!/usr/bin/env python3
"""FORGE source fetcher: online 4K -> lossless 3840x2160 mezzanine.

Per manifest entry (tools/forge/sources-v1.json):
  download (curl, resumable, size-verified) -> [unzip] -> ffmpeg ONE pass to
  <out>/mezz/<id>.mkv : cover-scale/centre-crop to 3840x2160, 10-bit,
  source chroma format kept (4:2:0/4:2:2/4:4:4), x264 LOSSLESS (qp 0,
  fast decode), tagged BT.709 limited -> provenance json -> raw deleted.
Everything atomic (.tmp + os.replace); reruns skip finished entries.

Then writes <out>/sources.json for tools/forge_data.py (path = mezz).

Colour: all v1 sources are BT.709 SDR limited-range YUV (verified on the
J2K excerpts: luma 109..917 of 64..940). in_color_matrix/out are pinned to
bt709 so the mezz conversion never applies swscale's untagged-601 guess.

Usage: forge_fetch.py tools/forge/sources-v1.json --out .scratch/forge/sources
       [--jobs 3] [--only id1,id2] [--keep-raw]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = "forge-dataset-fetch/1.0"
W, H = 3840, 2160
PREFIX = (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos+accurate_rnd:"
          "in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=tv,"
          f"crop={W}:{H},setsar=1")


def _guard(p):
    ap = os.path.realpath(os.path.abspath(p))
    if not ap.startswith(REPO + os.sep):
        raise ValueError(f"refusing write outside repo root: {ap}")
    return ap


def _len(url):
    out = subprocess.run(["curl", "-sIL", "-A", UA, url], capture_output=True,
                         text=True).stdout
    n = [int(l.split(":")[1]) for l in out.splitlines()
         if l.lower().startswith("content-length:")]
    return n[-1] if n else None


def download(url, dst, canonical=None):
    """Resumable, retried, size-verified. When fetched through a mirror/CDN
    (Netflix: http://download.opencontent.netflix.com, ~40x faster than the
    S3 endpoint here), the size must equal the canonical S3 object's."""
    want = _len(url)
    if canonical:
        cw = _len(canonical)
        if cw is None or want != cw:
            raise RuntimeError(f"mirror size {want} != canonical {cw}: {url}")
    if os.path.exists(dst) and want and os.path.getsize(dst) == want:
        return dst
    for attempt in range(6):
        if os.path.exists(dst) and want and os.path.getsize(dst) > want:
            os.remove(dst)          # overshoot = corrupt partial: start over
        r = subprocess.run(["curl", "-sSL", "-A", UA, "--retry", "5", "--retry-delay", "10",
                            "-C", "-", "-o", dst, url])
        if r.returncode == 33:      # server has no byte ranges: full restart
            if os.path.exists(dst):
                os.remove(dst)
            subprocess.run(["curl", "-sSL", "-A", UA, "--retry", "5", "--retry-delay", "10",
                            "-o", dst, url])
        if os.path.exists(dst) and (want is None or os.path.getsize(dst) == want):
            return dst
    raise RuntimeError(f"size mismatch after retries: {dst} {os.path.getsize(dst)} != {want}")


def _pixfmt(path_args):
    out = subprocess.run(["ffprobe", "-v", "error"] + path_args +
                         ["-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
                          "-of", "csv=p=0"], capture_output=True, text=True).stdout.strip()
    out = out.splitlines()[0] if out else ""
    if "444" in out or out.startswith(("rgb", "gbr")):
        return "yuv444p10le"
    if "422" in out:
        return "yuv422p10le"
    return "yuv420p10le"


USERS = {}   # url -> number of manifest entries using it


def mezz(e, out, keep_raw):
    dst = _guard(os.path.join(out, "mezz", f"{e['id']}.mkv"))
    prov = dst.replace(".mkv", ".json")
    if os.path.exists(prov):
        return e["id"], "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    raw_dir = _guard(os.path.join(out, "raw", e["id"]))
    os.makedirs(raw_dir, exist_ok=True)
    k = e["kind"]
    raw = None
    if k == "local":
        inp = ["-i", os.path.join(REPO, e["path"])]
    else:
        # one download shared by entries that point at the same URL (bbb segs)
        name = e["url"].split("/")[-2] + "__" + os.path.basename(e["url"])
        shared = _guard(os.path.join(out, "raw", "_shared", name))
        os.makedirs(os.path.dirname(shared), exist_ok=True)
        raw = download(e["url"], shared, e.get("canonical_url"))
        if k == "zip-j2k":
            xd = os.path.join(raw_dir, "x")
            if not glob.glob(os.path.join(xd, "**", "*.j2k"), recursive=True):
                subprocess.run(["unzip", "-q", "-o", raw, "-d", xd], check=True)
            frames = sorted(glob.glob(os.path.join(xd, "**", "*.j2k"), recursive=True))
            lst = os.path.join(raw_dir, "frames.txt")
            with open(lst, "w") as f:
                for p in frames:
                    f.write(f"file '{p}'\nduration {1.0 / eval(e['fps']):.9f}\n")
            inp = ["-f", "concat", "-safe", "0", "-i", lst]
        else:
            inp = ["-i", raw]
    seek = ["-ss", str(e["start"])] if e.get("start") else []
    inp = seek + inp
    pf = _pixfmt(inp[-2:] if k != "zip-j2k" else ["-i", frames[0]])
    rate = ["-r", e["fps"]] if k == "zip-j2k" else []
    lim = ["-frames:v", str(e["max_frames"])] if e.get("max_frames") else []
    tmp = dst + ".tmp.mkv"
    cmd = (["ffmpeg", "-v", "error", "-y", "-threads", "8"] + inp + rate + lim +
           ["-an", "-sn", "-dn", "-vf", PREFIX, "-c:v", "libx264", "-preset", "ultrafast",
            "-qp", "0", "-pix_fmt", pf, "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv", tmp])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return e["id"], f"FFMPEG-FAIL {r.stderr[-400:]}"
    n = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_packets,width,height,pix_fmt",
                        "-of", "json", tmp], capture_output=True, text=True).stdout
    info = json.loads(n)["streams"][0]
    os.replace(tmp, dst)
    with open(prov, "w") as f:
        json.dump({**e, "mezz": os.path.relpath(dst, REPO), "frames": int(info["nb_read_packets"]),
                   "mezz_pix_fmt": info["pix_fmt"], "raw_bytes": os.path.getsize(raw) if raw else None,
                   "ffmpeg_cmd": cmd}, f, indent=1)
    if not keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        if raw and USERS.get(e["url"], 1) == 1:
            os.remove(raw)          # single-user download: free it now
    return e["id"], f"ok frames={info['nb_read_packets']} {info['pix_fmt']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    ents = man["sources"]
    if a.only:
        keep = set(a.only.split(","))
        ents = [e for e in ents if e["id"] in keep]
    out = _guard(a.out)
    for e in man["sources"]:
        if e["kind"] != "local":
            USERS[e["url"]] = USERS.get(e["url"], 0) + 1
    # licence texts, archived once per URL
    lic_dir = os.path.join(out, "licenses")
    os.makedirs(lic_dir, exist_ok=True)
    for u in sorted({e["license_url"] for e in ents if e["license_url"].endswith(".txt")}):
        subprocess.run(["curl", "-sSL", "-A", UA, "-o",
                        os.path.join(lic_dir, u.split("/")[-3] + "__" + os.path.basename(u)), u])
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, s in ex.map(lambda e: mezz(e, out, a.keep_raw), ents):
            print(f"{i}: {s}", flush=True)
    # shared raw downloads are only removed once every user is done
    if not a.keep_raw:
        users = {}
        for e in man["sources"]:
            if e["kind"] != "local":
                users.setdefault(e["url"].split("/")[-2] + "__" + os.path.basename(e["url"]), []).append(e["id"])
        for name, ids in users.items():
            if all(os.path.exists(os.path.join(out, "mezz", f"{i}.json")) for i in ids):
                p = os.path.join(out, "raw", "_shared", name)
                if os.path.exists(p):
                    os.remove(p)
    # forge_data sources.json from the finished mezzanines
    srcs = []
    for p in sorted(glob.glob(os.path.join(out, "mezz", "*.json"))):
        j = json.load(open(p))
        srcs.append({"id": j["id"], "path": os.path.join(REPO, j["mezz"]), "split": j["split"],
                     "license": j["license"], "url": j.get("url", j.get("path")),
                     "attribution": j["attribution"], "content": j["content"]})
    with open(os.path.join(out, "sources.json"), "w") as f:
        json.dump(srcs, f, indent=1)
    print(f"sources.json: {len(srcs)} mezzanines")


if __name__ == "__main__":
    sys.exit(main())
