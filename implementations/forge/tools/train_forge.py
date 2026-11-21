#!/usr/bin/env python3
"""FORGE Stage-1 trainer — lanczos3 base + learned delta, pause/reboot safe.

    out_lin = lanczos3(lr_lin) + delta(srgb(lr_lin))         (linear light)
    loss    = L1( srgb(clamp(out_lin, 0, 1)), gt_srgb )     (metric domain)

Contract: docs/design/FORGE-NEW-MODEL.md (§1 design, §3 falsifiers, §6
domain rulings). Data: tools/forge_data.py shards.

Survival (explicit user requirement):
  * resume.pt holds net + opt + sched + step + best + ALL RNG states
    (python/numpy/torch/cuda) + data-pool RNG; written to resume.pt.tmp
    then os.replace() — a power cut mid-write never corrupts state.
  * saved every --save-every steps and on SIGTERM/SIGINT (graceful:
    finish the current step, save, exit 0) — `systemctl --user stop
    forge-train` IS the pause button; start resumes from the saved state (pool contents are refilled, so not bit-for-bit).
  * DONE marker when --steps reached: restarts (reboot) exit immediately.
  * crash / GPU reset -> nonzero exit -> systemd Restart=on-failure.

Data I/O: the work drive is a spinning disk. A background thread streams
CONTIGUOUS chunks of random shards into a RAM pool; batches are sampled
from the pool (no per-tile random seeks — the C4 I/O lesson).

Log lines (train_dashboard.py-compatible):
  step N: train L holdout H      saved PATH (step N, holdout H)
  resumed from step N            eval ... per-chain dPSNR table
"""
import argparse
import glob
import json
import math
import os
import random
import signal
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_lanczos import lanczos3_nchw  # noqa: E402
from forge_model import ForgeDeltaNet, forward_fast  # noqa: E402
from forge_player_io import srgb_encode  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = 3


def _guard(p):
    ap = os.path.realpath(os.path.abspath(p))
    if not ap.startswith(REPO + os.sep):
        raise ValueError(f"refusing write outside repo root: {ap}")
    return ap


# ---------------------------------------------------------------- data
class Shard:
    def __init__(self, d):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.lr = np.load(os.path.join(d, "lr.npy"), mmap_mode="r")
        self.gt = np.load(os.path.join(d, "gt.npy"), mmap_mode="r")
        self.n = len(self.lr)
        self.chain = self.meta["chain"]
        self.split = self.meta.get("split", "train")
        self.margin = self.meta["margin"]


def load_shards(roots):
    out = []
    for r in roots:
        for d in sorted(glob.glob(os.path.join(r, "shards", "*"))):
            if os.path.exists(os.path.join(d, "done.json")):
                out.append(Shard(d))
    return out


class Pool:
    """RAM shuffle pool fed by sequential chunk reads of random shards."""

    def __init__(self, shards, cap, chunk, seed, roots=None, rescan_s=300):
        self.shards, self.cap, self.chunk = shards, cap, chunk
        self.roots, self.rescan_s, self.last_scan = roots, rescan_s, time.time()
        self.rng = random.Random(seed)
        self.lr, self.gt = [], []
        self.lock = threading.Lock()
        self.stop = False
        self._fill_once(min(cap, 4 * chunk))
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _read(self):
        s = self.rng.choice(self.shards)
        i0 = self.rng.randrange(0, max(1, s.n - self.chunk + 1))
        i1 = min(s.n, i0 + self.chunk)
        return (torch.from_numpy(np.ascontiguousarray(s.lr[i0:i1])),
                torch.from_numpy(np.ascontiguousarray(s.gt[i0:i1])))

    def _fill_once(self, n):
        while sum(len(x) for x in self.lr) < n:
            lr, gt = self._read()
            with self.lock:
                self.lr.append(lr)
                self.gt.append(gt)

    def _rescan(self):
        """Data streams in while training (build_loop): pick up newly
        finished TRAIN shards. The val set stays fixed for comparability."""
        self.last_scan = time.time()
        have = {s.dir for s in self.shards}
        new = [s for s in load_shards(self.roots) if s.split == "train" and s.dir not in have]
        if new:
            self.shards = self.shards + new
            print(f"data: +{len(new)} train shards (now {len(self.shards)}, "
                  f"{sum(s.n for s in self.shards)} tiles)", flush=True)

    def _run(self):
        while not self.stop:
            if self.roots and time.time() - self.last_scan > self.rescan_s:
                try:
                    self._rescan()
                except Exception as e:  # a shard mid-replace: try next scan
                    print(f"rescan skipped: {e}", flush=True)
            lr, gt = self._read()
            with self.lock:
                self.lr.append(lr)
                self.gt.append(gt)
                while sum(len(x) for x in self.lr) > self.cap:
                    self.lr.pop(0)
                    self.gt.pop(0)
            time.sleep(0.05)

    def batch(self, bs, rng):
        with self.lock:
            sizes = [len(x) for x in self.lr]
            tot = sum(sizes)
            idx = sorted(rng.randrange(tot) for _ in range(bs))
            lrs, gts, k, acc = [], [], 0, 0
            for i in idx:
                while i >= acc + sizes[k]:
                    acc += sizes[k]
                    k += 1
                lrs.append(self.lr[k][i - acc])
                gts.append(self.gt[k][i - acc])
        return torch.stack(lrs), torch.stack(gts)


def augment(lr, gt, rng):
    """Dihedral aug, identical on LR (B,3,H,W) and GT (B,H,W,3). Lanczos
    with the player's centre convention commutes with flips/transposes."""
    gt = gt.permute(0, 3, 1, 2)
    if rng.random() < 0.5:
        lr, gt = lr.flip(-1), gt.flip(-1)
    if rng.random() < 0.5:
        lr, gt = lr.flip(-2), gt.flip(-2)
    if rng.random() < 0.5:
        lr, gt = lr.transpose(-1, -2), gt.transpose(-1, -2)
    return lr, gt


# ---------------------------------------------------------------- model step
def forward(net, lr_lin, margin, delta_fn=None):
    """lr_lin [B,3,T,T] linear (with margin) -> (out_srgb, base_srgb) for
    the core region, at 3x. delta_fn: the compiled training fast path
    (forge_model.forward_fast); eval uses the reference module."""
    base = lanczos3_nchw(lr_lin, SCALE).clamp(min=0.0)
    inp = srgb_encode(lr_lin) - 0.5
    delta = delta_fn(net, inp) if delta_fn else net(inp)
    m = margin * SCALE
    base, delta = base[..., m:-m, m:-m], delta[..., m:-m, m:-m]
    out = srgb_encode((base + delta).clamp(0.0, 1.0))
    return out, srgb_encode(base)


def psnr(a, b):
    mse = F.mse_loss(a, b, reduction="none").flatten(1).mean(1).clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def evaluate(net, val, dev, bs=16, max_per_shard=128):
    """Fixed holdout tiles (first max_per_shard of every val shard), tiled
    — never a full frame (the C4 OOM lesson). PSNR is POOLED per chain
    (sum of squared error over all its tiles, then one PSNR) — like the
    campaign's whole-frame PSNR. A mean of per-tile PSNRs is dominated by
    near-black tiles where the base scores 60+ dB (measured: -6.1 dB
    per-tile-mean vs -1.4 dB pooled on the same checkpoint).
    Returns mean L1 + per-chain (base PSNR, forge PSNR, dPSNR)."""
    net.eval()
    acc, l1s = {}, []
    for s in val:
        n = min(s.n, max_per_shard)
        for i0 in range(0, n, bs):
            lr = torch.from_numpy(np.ascontiguousarray(s.lr[i0:i0 + bs])).to(dev).float()
            gt = torch.from_numpy(np.ascontiguousarray(s.gt[i0:i0 + bs])).to(dev)
            gt = gt.permute(0, 3, 1, 2).float() / 255.0
            out, base = forward(net, lr, s.margin)
            out8 = (out * 255).round() / 255     # score like an 8-bit capture
            base8 = (base * 255).round() / 255
            l1s.append(F.l1_loss(out, gt).item())
            a = acc.setdefault(s.chain, [0.0, 0.0, 0])
            a[0] += ((base8 - gt) ** 2).sum().item()
            a[1] += ((out8 - gt) ** 2).sum().item()
            a[2] += gt.numel()
    net.train()
    ps = lambda se, n: 10 * math.log10(n / max(se, 1e-12))
    table = {c: (ps(b, n), ps(o, n), ps(o, n) - ps(b, n)) for c, (b, o, n) in acc.items()}
    return float(np.mean(l1s)), table


# ---------------------------------------------------------------- survival
def rng_state():
    st = {"py": random.getstate(), "np": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--pool-tiles", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1949)
    ap.add_argument("--max-steps-this-run", type=int, default=0,
                    help="smoke: stop (with save) after N steps of this process")
    a = ap.parse_args()

    out = _guard(a.out)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "DONE")):
        print(f"DONE marker present in {out} — nothing to do")
        return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        # measured 2026-11-12, ROCm 7.2 / torch 2.15: 5/10 fresh processes
        # returned corrupted first-call GPU results with this on, 0/10 off
        print("refusing: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
              "corrupts first-call results on this ROCm stack")
        return 4

    shards = load_shards(a.data)
    train = [s for s in shards if s.split == "train"]
    val = [s for s in shards if s.split == "val"]
    if not train or not val:
        print(f"need train AND val shards (train {len(train)}, val {len(val)})")
        return 2
    print(f"data: {len(train)} train shards ({sum(s.n for s in train)} tiles), "
          f"{len(val)} val shards ({sum(s.n for s in val)} tiles); device {dev}")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    net = ForgeDeltaNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)

    def lr_at(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        p = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    resume = os.path.join(out, "resume.pt")
    step, best = 0, float("inf")
    batch_rng = random.Random(a.seed + 1)
    if os.path.exists(resume):
        st = torch.load(resume, weights_only=False, map_location="cpu")
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
        set_rng_state(st["rng"])
        batch_rng.setstate(st["batch_rng"])
        print(f"resumed from step {step} (best holdout {best:.5f})", flush=True)

    pool = Pool(train, a.pool_tiles, a.chunk, seed=f"{a.seed}:{step}", roots=a.data)
    fast = torch.compile(forward_fast, dynamic=False) if dev == "cuda" else forward_fast

    def state():
        return {"net": net.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "best": best,
                "rng": rng_state(), "batch_rng": batch_rng.getstate(),
                "args": vars(a)}

    stop = {"sig": None}

    def _sig(signum, frame):
        stop["sig"] = signum   # finish the current step, then save + exit
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    if step == 0:
        # zero-init sanity: forge == base EXACTLY, so every dPSNR must be
        # +0.000 here — a nonzero value means corrupted GPU results
        hold, table = evaluate(net, val, dev)
        print(f"step 0: train {hold:.5f} holdout {hold:.5f}  (init; expect dPSNR 0)", flush=True)
        for c, (pb, po, dp) in sorted(table.items()):
            print(f"  eval {c:14s} base {pb:6.2f}  forge {po:6.2f}  dPSNR {dp:+.3f}", flush=True)
        if any(abs(dp) > 1e-9 for _, _, dp in table.values()):
            print("INIT CHECK FAILED — forge != base at zero init; aborting")
            return 5
        # the zero-init model (== lanczos3) is the first best candidate:
        # selection can never return something worse than the floor
        best = hold
        atomic_save({"net": net.state_dict(), "step": 0, "holdout": hold,
                     "table": table, "args": vars(a)}, os.path.join(out, "best.pt"))
        print(f"saved {os.path.join(out, 'best.pt')} (step 0, holdout {hold:.5f})", flush=True)
    run_steps, losses, t0 = 0, [], time.time()
    while step < a.steps:
        lr_t, gt_t = pool.batch(a.batch, batch_rng)
        lr_t, gt_t = augment(lr_t, gt_t, batch_rng)
        lr_t = lr_t.to(dev, non_blocking=True).float()
        gt_t = gt_t.to(dev, non_blocking=True).float() / 255.0
        o, _ = forward(net, lr_t, train[0].margin, fast)
        loss = F.l1_loss(o, gt_t)
        if not torch.isfinite(loss):
            print(f"non-finite loss at step {step} — exiting for restart from last save")
            return 3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        run_steps += 1
        losses.append(loss.item())

        if step % a.eval_every == 0 or step == a.steps:
            hold, table = evaluate(net, val, dev)
            tr = float(np.mean(losses[-a.eval_every:]))
            rate = run_steps / (time.time() - t0)
            print(f"step {step}: train {tr:.5f} holdout {hold:.5f}  "
                  f"({rate:.1f} it/s, lr {sched.get_last_lr()[0]:.2e})", flush=True)
            for c, (pb, po, dp) in sorted(table.items()):
                print(f"  eval {c:14s} base {pb:6.2f}  forge {po:6.2f}  dPSNR {dp:+.3f}",
                      flush=True)
            if hold < best:
                best = hold
                atomic_save({"net": net.state_dict(), "step": step, "holdout": hold,
                             "table": table, "args": vars(a)},
                            os.path.join(out, "best.pt"))
                print(f"saved {os.path.join(out, 'best.pt')} (step {step}, holdout {hold:.5f})",
                      flush=True)
        if step % a.save_every == 0 or stop["sig"] or step == a.steps or \
                (a.max_steps_this_run and run_steps >= a.max_steps_this_run):
            atomic_save(state(), resume)
        if stop["sig"]:
            print(f"signal {stop['sig']} — state saved at step {step}, exiting", flush=True)
            pool.stop = True
            return 0
        if a.max_steps_this_run and run_steps >= a.max_steps_this_run:
            print(f"max-steps-this-run reached — state saved at step {step}", flush=True)
            pool.stop = True
            return 0
    with open(os.path.join(out, "DONE"), "w") as f:
        f.write(f"completed {step} steps\n")
    print(f"training complete at step {step}; DONE written", flush=True)
    pool.stop = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
