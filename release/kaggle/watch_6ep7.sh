#!/bin/sh
set -u
cd "$(dirname "$0")/../.." || exit 1
export KAGGLE_API_TOKEN="$(cat "$HOME/.myna_kaggle_token2")"
REF=aashishmahato124/myna-baseline-6ep-s7
LOG=runs/conv6.log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
i=0
while [ $i -lt 400 ]; do
    st=$(.venv/bin/kaggle kernels status "$REF" 2>/dev/null | tail -1)
    case "$st" in
        *COMPLETE*) say "$REF complete"; break ;;
        *ERROR*)    say "$REF ERRORED"; exit 1 ;;
    esac
    i=$((i + 1)); sleep 120
done
rm -rf /tmp/c6s7 && mkdir -p /tmp/c6s7
.venv/bin/kaggle kernels output "$REF" -p /tmp/c6s7 >>"$LOG" 2>&1
f=$(find /tmp/c6s7 -name "baseline_seed7.json" | head -1)
if [ -n "$f" ]; then cp "$f" runs/baseline_seed7_6ep.json; say "wrote runs/baseline_seed7_6ep.json"; else say "no JSON"; fi
rm -rf /tmp/c6s7
