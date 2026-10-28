#!/bin/bash
# Elo against Stockfish 17.1 at UCI_Elo 2000, 0.05 s per move, 240-ply cap, 1,000 games. Our exact commands:
#   games-only models (pt, player-base, players): src/benchmark_play.py   (slurm/elo1k.sbatch)
#   text models (sft, dpo), with or without a prompt: src/eval_sft.py      (slurm/eval1k.sbatch)
#   bash recipes/9_eval.sh games checkpoints/pt100/chsm8_latest.pt pt
#   bash recipes/9_eval.sh text  checkpoints/dpo_v21/v21_label_kl1.pt dpo "trade queens as early as possible"
set -euo pipefail
KIND=$1; CK=$2; TAG=$3; PROMPT=${4:-}
SF=${STOCKFISH:-stockfish}; N=${NGPU:-4}; G=$((1000 / N))
for k in $(seq 0 $((N - 1))); do
  if [ $KIND = games ]; then
    CUDA_VISIBLE_DEVICES=$k python src/benchmark_play.py "$CK" --engine "$SF" --games $G --engine-elo 2000 --engine-time 0.05 \
      --max-plies 240 --parallel-games 64 --out results/elo1k/$TAG/shard$k &
  else
    CUDA_VISIBLE_DEVICES=$k python src/eval_sft.py --prompt "$PROMPT" "$CK" --engine "$SF" --games $G --engine-elo 2000 \
      --engine-time 0.05 --max-plies 240 --parallel-games 64 --out results/sft_eval/$TAG/part$k &
  fi
done
wait
[ $KIND = games ] && python src/summarize_elo1k.py results/elo1k/$TAG || true
