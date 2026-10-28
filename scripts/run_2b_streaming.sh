#!/usr/bin/env bash
# Run 2b on months as 2a finishes them, instead of waiting for all of 2a.
#
# 2b is cheap relative to 2a: no SAN replay, just a vectorized numpy filter
# over Parquet columns plus a per-kept-game memmap copy - measured at ~155s
# for a full ~68M-game month, single-threaded. So this dispatches one
# single-threaded build_2b.py process per eligible month rather than a worker
# pool per month (that's what 2a needs for SAN replay; 2b doesn't).
#
# A month is eligible once data/2a/<month>/_SUCCESS exists. Never dispatch on
# meta.parquet/moves.u16 existing alone - 2a writes those incrementally and
# they are not valid until _SUCCESS lands.
#
#   scripts/run_2b_streaming.sh [NPAR] [MIN_ELO]

set -uo pipefail

ROOT="."
PY="${PYTHON:-python3}"
INDIR="$ROOT/data/2a"
OUTDIR="$ROOT/data/2b"
BUILD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/build_2b.py"

NPAR="${1:-8}"           # concurrent months (single-threaded each)
MIN_ELO="${2:-1800}"
POLL=60

mkdir -p "$OUTDIR" "$ROOT/logs"

# --- crash recovery, same shape as run_2a_streaming.sh -------------------
# filter_month() opens moves.u16 and calls pq.write_table() at the end; a
# killed process leaves a meta.parquet with no footer (unreadable, same
# failure mode as build_2a.py's writer) or none at all. Nothing to salvage.
for c in "$OUTDIR"/.claim_*; do
    [ -e "$c" ] || continue
    rmdir "$c" 2>/dev/null || rm -rf "$c"
done
for d in "$OUTDIR"/*/; do
    [ -d "$d" ] || continue
    if [ ! -e "$d/_SUCCESS" ]; then
        echo "discarding partial 2b output: $(basename "$d")"
        rm -rf "$d"
    fi
done

dispatch() {
    local month="$1"
    local src="$INDIR/$month"
    [ -e "$src/_SUCCESS" ] || return 0
    [ -e "$OUTDIR/$month/_SUCCESS" ] && return 0
    mkdir "$OUTDIR/.claim_$month" 2>/dev/null || return 0
    echo "$(date +%H:%M:%S) dispatch $month"
    "$PY" "$BUILD" "$INDIR" -o "$OUTDIR" --months "$month" --min-elo "$MIN_ELO" \
        >> "$ROOT/logs/2b_$month.log" 2>&1 &
}

echo "streaming 2b: up to $NPAR concurrent months, min-elo $MIN_ELO, polling $INDIR every ${POLL}s"
while true; do
    mapfile -t ready < <(for d in "$INDIR"/*/; do
        m=$(basename "$d")
        [ -e "$d/_SUCCESS" ] && echo "$m"
    done | sort)

    for m in "${ready[@]}"; do
        running=$(jobs -rp | wc -l)
        [ "$running" -ge "$NPAR" ] && break
        dispatch "$m"
    done

    # Stop once 2a is done producing (both its feeder and the download behind
    # it are gone) and every month 2a ever finished has a matching 2b output.
    if ! pgrep -f 'run_2a_streaming.sh' > /dev/null && ! pgrep -f 'download_archive.sh' > /dev/null; then
        pending=0
        for m in "${ready[@]}"; do
            [ -e "$OUTDIR/$m/_SUCCESS" ] || pending=$((pending + 1))
        done
        if [ "$pending" -eq 0 ] && [ "$(jobs -rp | wc -l)" -eq 0 ]; then
            echo "$(date +%H:%M:%S) all available 2a months converted to 2b"
            break
        fi
    fi
    sleep "$POLL"
done
wait
echo "built: $(ls -1d "$OUTDIR"/*/ 2>/dev/null | wc -l) months"
