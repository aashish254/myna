#!/bin/sh
# Wait for the second 3-epoch seed and copy its JSON into runs/.
set -u
cd "$(dirname "$0")/../.." || exit 1
export KAGGLE_API_TOKEN="$(cat "$HOME/.myna_kaggle_token")"
REF=aashishkumarmahato01/myna-baseline-3ep-s7
LOG=runs/overnight.log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if [ -f runs/baseline_seed7_3ep.json ]; then say "seed7 3ep already present"; exit 0; fi
i=0
while [ $i -lt 400 ]; do
    st=$(.venv/bin/kaggle kernels status "$REF" 2>/dev/null | tail -1)
    case "$st" in
        *COMPLETE*) say "$REF complete"; break ;;
        *ERROR*)    say "$REF ERRORED"; exit 1 ;;
    esac
    i=$((i + 1)); sleep 120
done
rm -rf /tmp/out7_3 && mkdir -p /tmp/out7_3
.venv/bin/kaggle kernels output "$REF" -p /tmp/out7_3 >>"$LOG" 2>&1
f=$(find /tmp/out7_3 -name "baseline_seed7.json" | head -1)
if [ -n "$f" ]; then cp "$f" runs/baseline_seed7_3ep.json; say "wrote runs/baseline_seed7_3ep.json"; else say "no JSON"; fi
rm -rf /tmp/out7_3
