#!/bin/sh
# Poll both 6-epoch kernels (old account and backup account); capture whichever finishes first.
set -u
cd "$(dirname "$0")/../.." || exit 1
T1="$HOME/.myna_kaggle_token"
T2="$HOME/.myna_kaggle_token2"
OLD=aashishkumarmahato01/myna-baseline-6ep-s21
NEW=aashishmahato124/myna-baseline-6ep-s21-b
LOG=runs/conv6.log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

grab() {  # $1 token file, $2 ref, $3 label
    rm -rf /tmp/c6_$3 && mkdir -p /tmp/c6_$3
    KAGGLE_API_TOKEN=$(cat "$1") .venv/bin/kaggle kernels output "$2" -p /tmp/c6_$3 >>"$LOG" 2>&1
    f=$(find /tmp/c6_$3 -name "baseline_seed21.json" | head -1)
    if [ -n "$f" ]; then
        cp "$f" runs/baseline_seed21_6ep.json
        printf '{"source": "%s", "ref": "%s"}\n' "$3" "$2" > runs/conv6_source.json
        say "wrote runs/baseline_seed21_6ep.json from $3"
        return 0
    fi
    say "no JSON from $3"
    return 1
}

i=0
while [ $i -lt 300 ]; do
    so=$(KAGGLE_API_TOKEN=$(cat "$T1") .venv/bin/kaggle kernels status "$OLD" 2>/dev/null | tail -1)
    sn=$(KAGGLE_API_TOKEN=$(cat "$T2") .venv/bin/kaggle kernels status "$NEW" 2>/dev/null | tail -1)
    case "$so" in *COMPLETE*) grab "$T1" "$OLD" old && exit 0 ;; *ERROR*) say "old errored" ;; esac
    case "$sn" in *COMPLETE*) grab "$T2" "$NEW" new && exit 0 ;; *ERROR*) say "new errored" ;; esac
    i=$((i + 1)); sleep 120
done
say "timed out"
