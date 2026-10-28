#!/bin/bash
# chsm8-sft (v2.1): per-token EmbeddingGemma-300m conditioning in the top 6 layers, one pass over 27.1M described sides +
# player prompts (x3), 10% label dropout, pinned no-text term. 8 GPUs (GH200), 3.1 h. Our exact arguments
# (slurm/sft_v2.sbatch with DATA=data/sft30m_v21, the five labeler families, as logged in results/sft_v2.jsonl trial sft_v21).
# Multi-node: torchrun --nnodes 2 --nproc-per-node 4 ...
set -euo pipefail
NPROC=${NPROC:-8}
LAB=gemma-4-31b,gemma-4-31b-patched,gemma-3-1b-distill,gemma-3-1b-distill-patched,gemma-3-1b-distill-clean
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
torchrun --standalone --nproc-per-node $NPROC src/sft_ablate.py --trial sft_v21 --results results/sft_v2.jsonl \
  --data data/sft30m_v21 --checkpoint checkpoints/pt100/chsm8_latest.pt \
  --encoder google/embeddinggemma-300m --encoder-kind token --pool mean --meta-labels '' --labelers $LAB --eval-labelers $LAB \
  --compile --cond-layers 6 --lr 5e-7 --cond-lr 8e-4 --final-lr-frac 0.0 --labels-per-step 0 --games 128 --token-budget 98304 \
  --cfg-dropout 0.1 --freeze-dummy --player-data data/sft_players --player-epochs 3 --player-peak 1 --planned-games -1 \
  --time-budget 100000 --val-every 2000 --val-games 8000 --eval-games 8000 --eval-test --save --save-dir checkpoints/sft_v21 \
  --ckpt-every 600 --resume "$@"
