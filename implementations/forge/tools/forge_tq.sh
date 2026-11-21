#!/usr/bin/env bash
# forge_tq.sh — FORGE temporal test queue (one GPU test lane per process).
#
#   tools/forge_tq.sh lane [ID]      run THE lane: claim pending jobs oldest-first, forever
#   tools/forge_tq.sh submit NAME [--prio 0-9] [--ckpt X.pt] [--tpath DIR] [-- EVAL ARGS...]
#                                    (prio: lower runs first; default 5, filler search 9)
#   tools/forge_tq.sh status         queue depth, in-flight, ledger tail, idle/wait totals
#
# ONE lane only (user rule 2026-11-15: evals never run in parallel, so GPU
# timing and results stay honest; workers generate candidates in parallel).
# A job evaluates tools/forge_tdev.py on the non-gate dev set with checkpoint
# X.pt (default merged-v2.pt). DIR (optional) is a directory whose
# forge_temporal.py shadows tools/forge_temporal.py (FORGE_TPATH); EVAL ARGS
# pass through (e.g. --nclamp 0 --set foo=1). Gate scenes are never evaluated.
#
# Layout under .scratch/forge/tq/: pending/ running/ done/ results/<name>.{txt,json}
# results/LEDGER.tsv: name lane submitted started finished wait_s run_s rc mean_vs_spatial
#   mean_vs_base c10 c23 noisy  (per-chain temporal-vs-base means)
# results/IDLE.tsv:   lane idle_from idle_to idle_s   (tester idle = workers too few)
# wait_s large = workers too many / lanes too few; idle_s large = workers too few.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
Q="$REPO/.scratch/forge/tq"
CKPT_DEFAULT="$REPO/.scratch/forge/constructed/merged-v2.pt"
mkdir -p "$Q"/{pending,running,done,results}
now() { date +%s; }

submit() {
  local name="$1"; shift
  local tpath="" ckpt="$CKPT_DEFAULT" prio=5
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
      --prio) prio="$2"; [[ "$prio" =~ ^[0-9]$ ]] || { echo "prio 0-9" >&2; return 2; }; shift 2 ;;
      --tpath) tpath="$(cd "$2" && pwd)" || return 2; shift 2 ;;
      --ckpt) ckpt="$(realpath "$2")" || return 2; shift 2 ;;
      *) echo "unknown option $1" >&2; return 2 ;;
    esac
  done
  [[ -f "$ckpt" ]] || { echo "no checkpoint $ckpt" >&2; return 2; }
  [[ $# -gt 0 && "$1" == "--" ]] && shift
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad name: $name" >&2; return 2; }
  if ls "$Q"/{pending,running,done}/*"-$name.job" >/dev/null 2>&1; then
    echo "name already used: $name" >&2; return 2
  fi
  # lane takes the lowest sort key first: p0 (urgent) .. p9 (filler), then FIFO
  local tmp="$Q/.tmp.$$.$RANDOM" f="$Q/pending/p$prio-$(date +%s%N)-$name.job"
  {
    printf 'NAME=%q\n' "$name"
    printf 'TPATH=%q\n' "$tpath"
    printf 'CKPT=%q\n' "$ckpt"
    printf 'SUBMITTED=%s\n' "$(now)"
    printf 'ARGS=('; printf '%q ' "$@"; printf ')\n'
  } > "$tmp"
  mv "$tmp" "$f"
  echo "queued $name -> results/$name.txt (depth $(ls "$Q/pending" | wc -l))"
}

lane() {
  local id="${1:-0}" idle_from=""
  exec 9>"$Q/lane.lock"
  flock -n 9 || { echo "a lane is already running (serial evals only)" >&2; return 3; }
  echo "lane $id up (repo $REPO)"
  while :; do
    local job=""
    for j in $(ls "$Q/pending" 2>/dev/null | sort); do
      if mv "$Q/pending/$j" "$Q/running/$j" 2>/dev/null; then job="$j"; break; fi
    done
    if [[ -z "$job" ]]; then
      [[ -z "$idle_from" ]] && idle_from=$(now)
      sleep 2; continue
    fi
    if [[ -n "$idle_from" ]]; then
      local t=$(now); printf '%s\t%s\t%s\t%s\n' "$id" "$idle_from" "$t" $((t - idle_from)) >> "$Q/results/IDLE.tsv"
      idle_from=""
    fi
    local NAME TPATH CKPT SUBMITTED; local -a ARGS
    # shellcheck disable=SC1090
    source "$Q/running/$job"
    local out="$Q/results/$NAME.txt" js="$Q/results/$NAME.json" st=$(now)
    echo "[$(date +%T)] lane $id: $NAME"
    ( cd "$REPO" && FORGE_TPATH="$TPATH" python3 tools/forge_tdev.py eval "${CKPT:-$CKPT_DEFAULT}" \
        "${ARGS[@]}" --json "$js" ) > "$out.part" 2>&1
    local rc=$? fin=$(now)
    mv "$out.part" "$out"
    local m
    m=$(python3 -I - "$js" <<'EOF' 2>/dev/null
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    print("NA\tNA\tNA\tNA\tNA"); sys.exit()
mean = lambda v: sum(v) / len(v) if v else float("nan")
vs = mean([x["temporal"] - x["spatial"] for x in r]); vb = mean([x["temporal"] - x["base"] for x in r])
ch = [mean([x["temporal"] - x["base"] for x in r if x["chain"] == c]) for c in ("c10_lanczos", "c23_lanczos", "noisy10")]
print("\t".join(f"{v:+.4f}" for v in [vs, vb] + ch))
EOF
)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$NAME" "$id" "$SUBMITTED" "$st" "$fin" \
      $((st - SUBMITTED)) $((fin - st)) "$rc" "$m" >> "$Q/results/LEDGER.tsv"
    mv "$Q/running/$job" "$Q/done/$job"
    echo "[$(date +%T)] lane $id: $NAME done rc=$rc in $((fin - st))s (waited $((st - SUBMITTED))s)"
  done
}

status() {
  echo "pending: $(ls "$Q/pending" | sed 's/^p\([0-9]\)-[0-9]*-/[\1]/; s/\.job$//' | tr '\n' ' ')"
  echo "running: $(ls "$Q/running" | sed 's/^p[0-9]-[0-9]*-//; s/\.job$//' | tr '\n' ' ')"
  echo "name lane submitted started finished wait_s run_s rc vs_spatial vs_base c10 c23 noisy"
  tail -n "${1:-15}" "$Q/results/LEDGER.tsv" 2>/dev/null
  awk -F'\t' '{w+=$6; r+=$7; n++} END {if (n) printf "jobs %d  mean wait %.0fs  mean run %.0fs\n", n, w/n, r/n}' "$Q/results/LEDGER.tsv" 2>/dev/null
  awk -F'\t' '{i+=$4} END {printf "tester idle total %ds\n", i}' "$Q/results/IDLE.tsv" 2>/dev/null
}

case "${1:-}" in
  lane) shift; lane "$@" ;;
  submit) shift; submit "$@" ;;
  status) shift; status "$@" ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
