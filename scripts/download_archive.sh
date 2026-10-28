#!/usr/bin/env bash
# Download the full Lichess standard-rated archive (tier 1).
#
#   ~2.51 TB across 161 monthly .pgn.zst files.
#
# Resumable (wget -c) and idempotent: files already present with a matching
# sha256 are skipped, so re-running after an interruption costs one checksum
# pass over what is already on disk.
#
# Concurrency is deliberately modest (default 3). database.lichess.org is a
# free public service, our own link is a 1 Gbps external route that ~3 streams
# already saturates (measured: 4 and 16 streams both landed at 104 MB/s), and
# above some threshold in between the server starts answering with HTTP 429 -
# hit at 16 concurrent, each request denied outright, not merely slow. Going
# wider buys nothing and provokes exactly the rate limiting that then stalls
# everything.
#
#   scripts/download_archive.sh [DEST_DIR] [N_PARALLEL]

set -uo pipefail

DEST="${1:-./data/archive}"
NPAR="${2:-3}"
BASE="https://database.lichess.org/standard"
# Months already downloaded elsewhere are symlinked rather than re-fetched.
LEGACY_DIR="./data"

mkdir -p "$DEST"
cd "$DEST" || exit 1

curl -fsS -O "$BASE/list.txt"       || { echo "cannot fetch list.txt"; exit 1; }
curl -fsS -O "$BASE/sha256sums.txt" || { echo "cannot fetch sha256sums.txt"; exit 1; }

# Adopt any already-downloaded months from the pre-existing data/ directory.
for f in "$LEGACY_DIR"/lichess_db_standard_rated_*.pgn.zst; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ -e "$b" ] || ln -s "$f" "$b"
done

fetch_one() {
    url="$1"; f=$(basename "$url")
    want=$(awk -v n="$f" '$2==n {print $1}' sha256sums.txt)
    if [ -z "$want" ]; then echo "[skip] $f: no checksum published"; return 0; fi

    # A .verified marker newer than the file means we already hashed it and it
    # passed. Without this, every restart re-hashes the whole archive: 2.51 TB
    # of sha256 is ~1.4 h of pure overhead before a single byte is fetched.
    if [ -e "$f" ] && [ -e "$f.verified" ] && [ "$f.verified" -nt "$f" ]; then
        echo "[ok]   $f (verified earlier)"; return 0
    fi

    if [ -e "$f" ]; then
        have=$(sha256sum "$f" | cut -d' ' -f1)
        if [ "$have" = "$want" ]; then touch "$f.verified"; echo "[ok]   $f (already complete)"; return 0; fi
        # A symlink that fails the checksum belongs to someone else - do not
        # let wget -c append into the target.
        [ -L "$f" ] && { echo "[warn] $f: symlink checksum mismatch, re-fetching"; rm -f "$f"; }
    fi

    echo "[get]  $f"
    # No -O: wget's -c only resumes correctly when it derives the filename
    # itself. With -O it truncates and restarts from zero on every retry, which
    # silently throws away hours of a 30 GB transfer. basename(url) == $f, and
    # we are cd'd into DEST, so the default name is already what we want.
    #
    # --retry-on-http-error=429,...: wget's default --tries only covers
    # network-level failures: it treats an HTTP response as "the server
    # replied" and does not retry it, so a 429 without this flag comes back
    # as one immediate, permanent failure. --waitretry backs off 1,2,4,8,16s
    # (capped at 10 min) between attempts instead of hammering again straight
    # away, which is what earns the retries any chance of succeeding.
    wget -q -c -T 60 --tries=8 --waitretry=16 \
        --retry-on-http-error=429,500,502,503,504 \
        "$url" || { echo "[FAIL] $f: download error"; return 1; }

    have=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$have" = "$want" ]; then touch "$f.verified"; fi
    if [ "$have" != "$want" ]; then
        # -c is only safe when a checksum failure discards the file. Leaving a
        # corrupt partial in place would make every future -c resume append to
        # bad bytes and fail forever. Delete it so the next run starts clean.
        echo "[FAIL] $f: checksum mismatch, discarding partial"
        rm -f "$f"
        return 1
    fi
    echo "[ok]   $f"
}
export -f fetch_one

echo "downloading $(wc -l < list.txt) files into $DEST with $NPAR parallel streams"
# -P N runs N at a time; each fetch_one is self-contained and restartable.
xargs -a list.txt -P "$NPAR" -I{} bash -c 'fetch_one "$@"' _ {}
rc=$?

echo
echo "on disk: $(ls -1 lichess_db_standard_rated_*.pgn.zst 2>/dev/null | wc -l) files, $(du -shL . 2>/dev/null | cut -f1)"
[ $rc -eq 0 ] && echo "ALL DONE" || echo "COMPLETED WITH FAILURES - re-run to retry (already-good files are skipped)"
exit $rc
