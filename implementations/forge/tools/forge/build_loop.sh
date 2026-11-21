#!/bin/bash
# build_loop.sh — overlap shard building with the source fetch.
# Each pass: sources.json from the finished mezzanines (mezz/*.json), then
# forge_data.py build (finished shards are skipped). Exits after a pass in
# which the fetch is no longer running and nothing new was built.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
SRC=.scratch/forge/sources
OUT=.scratch/forge/data
JOBS=${JOBS:-4}
fetch_running() {
  for d in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
    case "$c" in "python3 tools/forge_fetch.py"*) return 0;; esac
  done
  return 1
}
while true; do
  python3 - "$SRC" <<'EOF'
import glob, json, os, sys
src = sys.argv[1]
repo = os.getcwd()
out = []
for p in sorted(glob.glob(os.path.join(src, "mezz", "*.json"))):
    j = json.load(open(p))
    out.append({"id": j["id"], "path": os.path.join(repo, j["mezz"]), "split": j["split"],
                "license": j["license"], "url": j.get("canonical_url", j.get("url", j.get("path"))),
                "attribution": j["attribution"], "content": j["content"]})
json.dump(out, open(os.path.join(src, "sources-partial.json"), "w"), indent=1)
print(f"pass: {len(out)} mezzanines ready", flush=True)
EOF
  before=$(find "$OUT/shards" -maxdepth 2 -name done.json 2>/dev/null | wc -l)
  running=1; fetch_running || running=0
  python3 tools/forge_data.py build "$SRC/sources-partial.json" --out "$OUT" --jobs "$JOBS" \
    2>&1 | grep -v -e '^SvtMalloc' -e ': skip$'
  after=$(find "$OUT/shards" -maxdepth 2 -name done.json | wc -l)
  echo "shards: $before -> $after (fetch running: $running)"
  if [ "$running" = 0 ] && [ "$after" = "$before" ]; then echo "BUILD-LOOP DONE"; break; fi
  [ "$after" = "$before" ] && sleep 120
done
