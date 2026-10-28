#!/bin/bash
# Label game sides with the distilled labeler (vLLM, one worker per GPU). Our exact command (slurm/labeler_label_prod.sbatch).
# Inputs: shard_NNN.jsonl of {key, user} from src/labeler/build_unlabeled.py (needs data/2b and data/4/player_game_index.parquet).
# CKPT = runs/labeler/ckpt_full_gemma3_1b_clean, or the released weights: huggingface-cli download LegumMagister/chsm8-labeler-gemma3-1b
set -euo pipefail
CKPT=${CKPT:-runs/labeler/ckpt_full_gemma3_1b_clean}; IN=${IN:-data/unlabeled}; OUT=${OUT:-data/labels_distilled}
N=${NGPU:-$(nvidia-smi -L | wc -l)}
if [ ! -d "$IN" ]; then
  # the sides to label (here: every side of data/chsm8-post-training-unlabeled) and the sides already labeled (none)
  [ -f data/4/player_game_index.parquet ] || python -c "
import os, pyarrow.dataset as ds, pyarrow.parquet as pq
t = ds.dataset('data/chsm8-post-training-unlabeled/data').to_table(columns=['month', 'row_idx', 'side', 'player_elo'])
os.makedirs('data/4', exist_ok=True); pq.write_table(t.rename_columns(['month', 'row_idx', 'side', 'elo']), 'data/4/player_game_index.parquet')"
  [ -f data/none.jsonl.gz ] || python -c "import gzip; gzip.open('data/none.jsonl.gz', 'wt').close()"
  python src/labeler/build_unlabeled.py --out "$IN" --labels "${DONE_LABELS:-data/none.jsonl.gz}" --workers $(nproc)
fi
for w in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$w python src/labeler/label_shards.py --ckpt $CKPT --in $IN --out $OUT --worker $w --n-workers $N &
done
wait
