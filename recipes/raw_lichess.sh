#!/bin/bash
# chsm8-pre-training from scratch: the Lichess standard-rated archive (about 2.5 TB) -> 1.07B filtered games.
#   tier 1  data/archive/*.pgn.zst          scripts/download_archive.sh
#   tier 2a data/2a/<month>/ moves + meta   src/build_2a.py: blitz and slower (base + 40*incr >= 180 s), rated, no bots,
#                                           standard, decisive result, parseable Elos, SAN replays
#   tier 2b data/2b/<month>/                src/build_2b.py: both players >= 1800, no time forfeits, games cut where a
#                                           player had < 15 s left
#   parquet  data/hf/pre/data/<month>/      prep/pretraining/u16_to_parquet.py (= the published dataset)
# Then recipes/0_data.sh (pre) packs data/2b for training. CPU only; tier 2a is the expensive pass (SAN replay).
set -euo pipefail
W=${WORKERS:-$(nproc)}
bash scripts/download_archive.sh data/archive
for f in data/archive/lichess_db_standard_rated_*.pgn.zst; do
  python src/build_2a.py "$f" -o data/2a -w $W
done
python src/build_2b.py data/2a -o data/2b --min-elo 1800
python prep/pretraining/u16_to_parquet.py data/2b -o data/hf/pre/data --workers $W
