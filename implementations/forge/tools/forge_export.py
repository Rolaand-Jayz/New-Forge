#!/usr/bin/env python3
"""FORGE weight exporter: torch checkpoint -> .forge binary for the player's
ForgeStage (src/pipeline/ForgeStage.cpp, shaders/forge_*.comp).

Usage:
  forge_export.py CKPT OUT.forge          export (CKPT = train_forge.py
                                          checkpoint {"net": state_dict, ...}
                                          or a plain ForgeDeltaNet state_dict)
  forge_export.py --make random OUT.pt    parity-test checkpoint: torch default
                                          init everywhere, tail ~ N(0, 0.02)
                                          (weights AND bias) -> delta != 0
  forge_export.py --make zero OUT.pt      ForgeDeltaNet() as constructed:
                                          random body, ZERO tail (delta == 0)
  --seed N                                RNG seed for --make (default 1234)

.forge FORMAT v1 (all little-endian):
  off  size  field
  0    8     magic  b"FORGEW1\\0"
  8    4     u32 version = 1
  12   4     u32 header_bytes (offset of the payload; = 68 + 48*tensor_count)
  16   4     u32 channels  (C = 32)
  20   4     u32 blocks    (12)
  24   4     u32 scale     (3)
  28   4     u32 tensor_count (= 4 + 4*blocks = 52)
  32   4     u32 payload_floats (= 18299 for C=32/12 blocks/3x)
  36   32    sha256 of the payload bytes (integrity; the loader verifies it)
  68   48*n  tensor table, one entry per tensor in payload order:
               char name[24] (NUL-padded state_dict key)
               u32 ndim, u32 dims[4] (unused dims = 1), u32 offset_floats
  hdr  4*p   payload: fp32 arrays, row-major (torch .contiguous() order),
             concatenated in this FIXED order:
               stem.weight [C,3,3,3]   stem.bias [C]
               for i in 0..blocks-1:
                 body.i.dw.weight [C,1,3,3]  body.i.dw.bias [C]
                 body.i.pw.weight [C,C,1,1]  body.i.pw.bias [C]
               tail.weight [27,C,1,1]  tail.bias [27]
The player's tell-line prints sha256 of the WHOLE file (== `sha256sum`);
this tool prints both the file and the payload digest.
"""
import argparse
import hashlib
import os
import struct
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_model import BLOCKS, C, SCALE, ForgeDeltaNet  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab"))
from _repo_io import repo_open_w  # noqa: E402

MAGIC = b"FORGEW1\0"
VERSION = 1
FIXED_HDR = 68
ENTRY = 48


def tensor_order(blocks=BLOCKS):
    names = ["stem.weight", "stem.bias"]
    for i in range(blocks):
        names += [f"body.{i}.dw.weight", f"body.{i}.dw.bias",
                  f"body.{i}.pw.weight", f"body.{i}.pw.bias"]
    return names + ["tail.weight", "tail.bias"]


def load_state(path):
    st = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(st, dict) and "net" in st and isinstance(st["net"], dict):
        st = st["net"]
    net = ForgeDeltaNet()
    net.load_state_dict(st)       # strict: shapes/keys must match the model
    return net.state_dict()


def pack(sd):
    names = tensor_order()
    table, payload, off = [], [], 0
    for n in names:
        t = sd[n].detach().to(torch.float32).contiguous().cpu()
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {n}")
        dims = list(t.shape) + [1] * (4 - t.dim())
        nm = n.encode()
        if len(nm) >= 24:
            raise ValueError(f"name too long: {n}")
        table.append(struct.pack("<24sIIIIII", nm, t.dim(), *dims, off))
        payload.append(t.numpy().astype("<f4").tobytes())
        off += t.numel()
    body = b"".join(payload)
    hdr_bytes = FIXED_HDR + ENTRY * len(names)
    hdr = struct.pack("<8sIIIIIII", MAGIC, VERSION, hdr_bytes, C, BLOCKS, SCALE,
                      len(names), off)
    hdr += hashlib.sha256(body).digest()
    out = hdr + b"".join(table) + body
    assert len(hdr) == FIXED_HDR and len(out) == hdr_bytes + 4 * off
    return out, body, off


def read_forge(path):
    """Inverse of pack(): .forge -> state_dict (used by the parity gate so
    torch runs the EXACT bytes the player loaded)."""
    with open(os.path.realpath(path), "rb") as f:
        blob = f.read()
    magic, ver, hdr_bytes, c, blocks, scale, n, nfl = struct.unpack_from(
        "<8sIIIIIII", blob, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not a FORGE v1 weight file")
    body = blob[hdr_bytes:]
    if len(body) != 4 * nfl or hashlib.sha256(body).digest() != blob[36:68]:
        raise ValueError("payload size/sha256 mismatch")
    sd = {}
    for i in range(n):
        nm, nd, d0, d1, d2, d3, off = struct.unpack_from("<24sIIIIII", blob,
                                                         FIXED_HDR + ENTRY * i)
        shape = [d0, d1, d2, d3][:nd]
        cnt = 1
        for d in shape:
            cnt *= d
        t = torch.frombuffer(bytearray(body[4 * off:4 * (off + cnt)]),
                             dtype=torch.float32).reshape(shape)
        sd[nm.rstrip(b"\0").decode()] = t.clone()
    return sd, hashlib.sha256(blob).hexdigest()


def make(kind, seed):
    torch.manual_seed(seed)
    net = ForgeDeltaNet()            # torch default init; tail zeroed
    if kind == "random":
        nn.init.normal_(net.tail.weight, std=0.02)
        nn.init.normal_(net.tail.bias, std=0.02)
    return net.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--make", choices=["random", "zero"])
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    if a.make:
        sd = make(a.make, a.seed)
        with repo_open_w(a.out, "wb") as f:
            torch.save({"net": sd, "note": f"forge_export --make {a.make} "
                                           f"seed={a.seed}"}, f)
        tail = sd["tail.weight"].abs().max().item()
        print(f"wrote {a.out}: --make {a.make} seed={a.seed} "
              f"max|tail.weight|={tail:.4g}")
        return 0
    if not a.src:
        ap.error("CKPT required unless --make")
    sd = load_state(a.src)
    blob, body, nfl = pack(sd)
    with repo_open_w(a.out, "wb") as f:
        f.write(blob)
    print(f"exported {a.src} -> {a.out}")
    print(f"  params={nfl} bytes={len(blob)}")
    print(f"  payload sha256={hashlib.sha256(body).hexdigest()}")
    print(f"  file    sha256={hashlib.sha256(blob).hexdigest()} "
          f"(player tell-line prints the first 16 hex)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
