#!/bin/sh
# Overnight driver: run the 3-epoch convergence check, then a fourth 2-epoch seed.
# Each job is pushed, polled to completion, and its JSON copied into runs/. Safe to re-run:
# a finished kernel is skipped if its JSON already exists.
set -u
cd "$(dirname "$0")/../.." || exit 1
export KAGGLE_API_TOKEN="$(cat "$HOME/.myna_kaggle_token")"
LOG=runs/overnight.log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run() {  # $1 dir, $2 ref, $3 seed, $4 out_json
    if [ -f "$4" ]; then say "skip $2 (already have $4)"; return 0; fi
    say "push $2"
    .venv/bin/kaggle kernels push -p "release/kaggle/$1" >>"$LOG" 2>&1 || { say "push failed for $2"; return 1; }
    i=0
    while [ $i -lt 360 ]; do
        st=$(.venv/bin/kaggle kernels status "$2" 2>/dev/null | tail -1)
        case "$st" in
            *COMPLETE*) say "$2 complete"; break ;;
            *ERROR*)    say "$2 ERRORED"; return 1 ;;
        esac
        i=$((i + 1)); sleep 120
    done
    rm -rf "/tmp/out_$3" && mkdir -p "/tmp/out_$3"
    .venv/bin/kaggle kernels output "$2" -p "/tmp/out_$3" >>"$LOG" 2>&1
    f=$(find "/tmp/out_$3" -name "baseline_seed$3.json" | head -1)
    if [ -n "$f" ]; then cp "$f" "$4"; say "wrote $4"; else say "no JSON for $2"; fi
    rm -rf "/tmp/out_$3"
}

run a3ep    aashishkumarmahato01/myna-baseline-3ep-s21  21 runs/baseline_seed21_3ep.json
run bseed27 aashishkumarmahato01/myna-baseline-2ep-s27  27 runs/baseline_seed27_2ep.json
say "overnight done"
