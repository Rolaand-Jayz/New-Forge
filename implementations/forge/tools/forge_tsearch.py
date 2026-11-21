#!/usr/bin/env python3
"""FORGE filler search — keeps the single test lane busy with coordinate-search
moves over the constructed weights (forge_construct merged knobs) and the
temporal knobs, at the LOWEST queue priority (9), so worker candidates (prio
0-5) always run first and the lane never idles.

It submits a move only when the queue has nothing pending, so at most one
filler job ever waits. Objective (non-gate dev set, forge_tdev cells):

  J = mean_cells(temporal - base) - PEN * mean_clean(max(0, base - temporal))

clean = c10_lanczos and c23_lanczos cells: FORGE doing worse than plain
lanczos3 on clean or compressed footage is penalised on top of the mean.
Gate scenes are never touched (forge_tdev only knows non-gate sources).

  forge_tsearch.py run        loop until plateau (state in .scratch/forge/tq/search/)
  forge_tsearch.py best       print the incumbent knobs and score
  forge_tsearch.py score N..  J + per-cell temporal-vs-base of finished queue jobs
"""
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forge_construct as FC  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(REPO, ".scratch/forge/tq")
SD = os.path.join(Q, "search")
STATE = os.path.join(SD, "state.json")
LOG = os.path.join(SD, "search.log")
TQ = os.path.join(REPO, "tools/forge_tq.sh")
PEN = 2.0
CLEAN = ("c10_lanczos", "c23_lanczos")

# temporal knobs = forge_tdev eval args (player push constants)
TEMPORAL = {"kappa": 32.0, "amin": 0.25, "tref": 0.004, "hold": 0.9, "nclamp": 0.0, "tdead": 0.0}

# kind: "mul" (positive scale, step = factor), "add" (step = delta), "int"
SPACE = {
    "rho": ("mul", 1.25, 0.1, 10), "t0": ("mul", 1.5, 1e-5, 0.01), "wHH": ("mul", 1.2, 0.2, 5),
    # m (gate blur passes) and n (diffusion iterations) are pinned at 5 / 2 by
    # the merged block schedule, so they are not searched
    "wC": ("mul", 1.3, 0.02, 3), "gf": ("mul", 1.5, 1e-4, 0.2),
    "gb": ("int", 1, 0, 1), "kY": ("mul", 1.2, 0.5, 20), "kC": ("mul", 1.2, 1, 40),
    "a": ("mul", 1.1, 0.3, 1.5), "k1": ("add", 0.03, 0.02, 0.4),
    "k2": ("add", 0.05, 0.2, 0.8), "beta": ("add", 0.08, 0.0, 1.0), "w": ("add", 0.03, 0.0, 0.4),
    "gain": ("mul", 1.1, 0.2, 1.5), "dg": ("mul", 1.15, 0.1, 1.5), "cw": ("mul", 1.15, 0.1, 1.5),
    "dt": ("add", 0.002, 0.0, 0.05),
    "kappa": ("mul", 1.5, 1, 2000), "amin": ("add", 0.05, 0.05, 0.9), "tref": ("mul", 1.5, 5e-4, 0.05),
    "hold": ("add", 0.03, 0.5, 0.99), "nclamp": ("add", 0.01, 0.0, 0.2), "tdead": ("add", 0.0005, 0.0, 0.01),
}
ORDER = list(SPACE)


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def score(res):
    d = [r["temporal"] - r["base"] for r in res]
    loss = [max(0.0, r["base"] - r["temporal"]) for r in res if r["chain"] in CLEAN]
    return sum(d) / len(d) - PEN * (sum(loss) / len(loss) if loss else 0.0)


def valid(k):
    return k["k1"] < k["k2"] - 0.05


def move(k, name, sign, steps):
    kind, _, lo, hi = SPACE[name]
    st = steps[name]
    v = k[name]
    if kind == "mul":
        nv = v * (st if sign > 0 else 1 / st) if v > 0 else (lo if sign > 0 else 0.0)
    else:
        nv = v + sign * st
    if kind == "int":
        nv = int(round(nv))
    nv = min(max(nv, lo), hi)
    if nv == v:
        return None
    out = dict(k, **{name: nv})
    return out if valid(out) else None


def queue_has_pending():
    p = os.path.join(Q, "pending")
    return os.path.isdir(p) and any(f.endswith(".job") for f in os.listdir(p))


def submit(name, knobs):
    ck = os.path.join(SD, f"{name}.pt")
    wk = {k: knobs[k] for k in FC.MERGED_KNOBS}
    subprocess.run([sys.executable, os.path.join(REPO, "tools/forge_construct.py"), ck,
                    "--mode", "merged", "--knobs", json.dumps(wk)],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    ev = []
    for t in TEMPORAL:
        ev += [f"--{t}", repr(float(knobs[t]))]
    subprocess.run([TQ, "submit", name, "--prio", "9", "--ckpt", ck, "--"] + ev,
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)


def result(name):
    """(rc, res) once the lane finished NAME, else None."""
    led = os.path.join(Q, "results/LEDGER.tsv")
    if not os.path.exists(led):
        return None
    for line in open(led):
        f = line.rstrip("\n").split("\t")
        if f[0] == name:
            js = os.path.join(Q, "results", name + ".json")
            res = json.load(open(js)) if f[7] == "0" and os.path.exists(js) else None
            return int(f[7]), res
    return None


def init_state():
    inc = dict(FC.MERGED_KNOBS, **TEMPORAL)
    steps = {n: SPACE[n][1] for n in ORDER}
    return {"inc": inc, "inc_J": None, "inc_res": None, "steps": steps, "idx": 0, "sign": 1,
            "tried_minus": False, "pending": None, "cand": None, "n": 0, "sweeps_without_gain": 0,
            "gain_this_sweep": False}


def save(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, STATE)


def next_candidate(s):
    """Advance the coordinate pointer to the next valid move; None = sweep wrapped."""
    for _ in range(2 * len(ORDER) + 2):
        name = ORDER[s["idx"]]
        c = move(s["inc"], name, s["sign"], s["steps"])
        if c is not None:
            return name, c
        advance(s, improved=False)
    return None, None


def advance(s, improved):
    name = ORDER[s["idx"]]
    if improved:
        return  # keep going the same direction on the same knob
    if s["sign"] > 0 and not s["tried_minus"]:
        s["sign"], s["tried_minus"] = -1, True
        return
    # both directions failed: shrink this knob's step, move to the next knob
    kind = SPACE[name][0]
    if kind == "mul":
        s["steps"][name] = 1 + (s["steps"][name] - 1) / 2
    elif kind == "add":
        s["steps"][name] = s["steps"][name] / 2
    s["idx"], s["sign"], s["tried_minus"] = (s["idx"] + 1) % len(ORDER), 1, False
    if s["idx"] == 0:
        s["sweeps_without_gain"] = 0 if s["gain_this_sweep"] else s["sweeps_without_gain"] + 1
        s["gain_this_sweep"] = False
        log(f"sweep done; sweeps without gain: {s['sweeps_without_gain']}")


def run():
    os.makedirs(SD, exist_ok=True)
    s = json.load(open(STATE)) if os.path.exists(STATE) else init_state()
    log(f"search up; incumbent J={s['inc_J']}")
    while True:
        if s["pending"]:
            r = result(s["pending"])
            if r is None:
                time.sleep(5)
                continue
            rc, res = r
            name = s["pending"]
            if res is None:
                log(f"{name}: eval failed rc={rc}; treating as worse")
                J = -math.inf
            else:
                J = score(res)
            if s["inc_J"] is None:
                s["inc_J"], s["inc_res"] = J, res
                log(f"{name}: baseline J={J:+.4f}")
            else:
                knob = ORDER[s["idx"]]
                better = J > s["inc_J"] + 1e-4
                log(f"{name}: {knob}={s['cand'][knob]!r} J={J:+.4f} vs inc {s['inc_J']:+.4f} "
                    f"-> {'ACCEPT' if better else 'reject'}")
                if better:
                    s["inc"], s["inc_J"], s["inc_res"] = s["cand"], J, res
                    s["gain_this_sweep"] = True
                    with open(os.path.join(SD, "best.json"), "w") as f:
                        json.dump({"J": J, "knobs": s["inc"], "res": res, "from": name}, f, indent=1)
                advance(s, better)
            s["pending"], s["cand"] = None, None
            save(s)
            continue
        if s["sweeps_without_gain"] >= 2:
            log("plateau: two full sweeps without a gain; search stops")
            return
        if queue_has_pending():
            time.sleep(5)
            continue
        s["n"] += 1
        name = f"srch{s['n']:04d}"
        if s["inc_J"] is None:
            cand = s["inc"]
        else:
            knob, cand = next_candidate(s)
            if cand is None:
                log("no valid move left; search stops")
                return
        try:
            submit(name, cand)
        except subprocess.CalledProcessError as e:
            log(f"{name}: submit failed ({e}); skipping move")
            advance(s, improved=False)
            save(s)
            continue
        s["pending"], s["cand"] = name, cand
        save(s)


def best():
    s = json.load(open(STATE))
    print(json.dumps({"J": s["inc_J"], "knobs": s["inc"]}, indent=1))


def score_cmd(names):
    """forge_tsearch.py score NAME [NAME..]: J and per-cell temporal-vs-base of finished jobs."""
    for n in names:
        r = result(n)
        if r is None:
            print(f"{n}: not finished")
            continue
        rc, res = r
        if res is None:
            print(f"{n}: FAILED rc={rc} (see .scratch/forge/tq/results/{n}.txt)")
            continue
        cells = "  ".join(f"{x['src'][:9]}/{x['chain'][:5]} {x['temporal'] - x['base']:+.3f}" for x in res)
        print(f"{n}: J={score(res):+.4f}   {cells}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "score":
        score_cmd(sys.argv[2:])
    else:
        {"run": run, "best": best}[cmd]()
