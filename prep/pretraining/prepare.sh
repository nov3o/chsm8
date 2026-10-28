#!/usr/bin/env bash
# One-shot: published Parquet -> CHSM8 training records.
#
#   parquet (this repo) --parquet_to_u16--> u16 --to_factorized--> 3 --pack--> 3p [--shuffle--> 3p-shuffled]
#
# Every stage is optional to re-run and writes a _SUCCESS marker. Configure with
# environment variables (defaults shown):
#
#   SRC=data                 Parquet root (a local clone / snapshot of this dataset)
#   WORK=work                output root; stages go to $WORK/{u16,3,3p,3p-shuffled}
#   MONTHS=""                space-separated subset, e.g. "2025-11 2025-12" (default: all)
#   MIN_ELO=0                extra filter: min(white_elo, black_elo) >= MIN_ELO (corpus is already >= 1800)
#   MIN_TC_SECONDS=0         extra filter: base + 40*increment >= this (corpus is already >= 180)
#   SEQ_LEN=512              packed record length (records hold SEQ_LEN+1 rows)
#   VAL_FRACTION=0.01        trailing fraction of each month used for validation
#   LEGAL=1                  1 = write occ/pmask legality sidecars, 0 = skip
#   SHUFFLE=1                1 = bucket-shuffle training records, 0 = keep chronological order
#   BUCKETS=128              shuffled output shards
#   SEED=42
#   WORKERS=$(nproc)
#
#   SRC=data WORK=/scratch/chsm8 SEQ_LEN=512 LEGAL=1 SHUFFLE=1 bash scripts/prepare.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
SRC="${SRC:-data}"; WORK="${WORK:-work}"; MONTHS="${MONTHS:-}"
MIN_ELO="${MIN_ELO:-0}"; MIN_TC_SECONDS="${MIN_TC_SECONDS:-0}"
SEQ_LEN="${SEQ_LEN:-512}"; VAL_FRACTION="${VAL_FRACTION:-0.01}"
LEGAL="${LEGAL:-1}"; SHUFFLE="${SHUFFLE:-1}"; BUCKETS="${BUCKETS:-128}"; SEED="${SEED:-42}"
WORKERS="${WORKERS:-$(nproc)}"

months_arg=(); [[ -n "$MONTHS" ]] && months_arg=(--months $MONTHS)
legal_arg=(); [[ "$LEGAL" == 0 ]] && legal_arg=(--no-legal)
t() { local s=$SECONDS; "$@"; echo "  -> $(( SECONDS - s ))s"; }

echo "[1/4] Parquet -> u16 ($WORK/u16)"
t "$PY" "$here/parquet_to_u16.py" "$SRC" -o "$WORK/u16" "${months_arg[@]}" \
  --min-elo "$MIN_ELO" --min-tc-seconds "$MIN_TC_SECONDS" --workers "$WORKERS"

echo "[2/4] u16 -> factorized rows ($WORK/3)"
t "$PY" "$here/to_factorized.py" "$WORK/u16" -o "$WORK/3" "${legal_arg[@]}" --workers "$WORKERS"
"$PY" "$here/verify.py" factorized "$WORK/3" | tail -1

echo "[3/4] factorized -> packed seq_len=$SEQ_LEN ($WORK/3p)"
t "$PY" "$here/pack.py" "$WORK/3" -o "$WORK/3p" --seq-len "$SEQ_LEN" \
  --val-fraction "$VAL_FRACTION" "${legal_arg[@]}" --workers "$WORKERS"
"$PY" "$here/verify.py" packed "$WORK/3p" | tail -1

if [[ "$SHUFFLE" == 1 ]]; then
  echo "[4/4] shuffle -> $WORK/3p-shuffled"
  t "$PY" "$here/shuffle.py" "$WORK/3p" -o "$WORK/3p-shuffled" --buckets "$BUCKETS" --seed "$SEED" --link-val
  "$PY" "$here/verify.py" packed "$WORK/3p-shuffled" | tail -1
  echo "training data: $WORK/3p-shuffled"
else
  echo "[4/4] shuffle skipped; training data: $WORK/3p"
fi
