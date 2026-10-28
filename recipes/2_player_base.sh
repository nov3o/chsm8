#!/bin/bash
# chsm8-player-base: chsm8-pt continued on the 26.0M unique post-training games, no text. 4 GPUs, about 65 min.
# Our exact command (slurm/games_run.sbatch g100_drop_lr5e-05 6000 data/games30m + slurm/games/r100.txt).
set -euo pipefail
NPROC=${NPROC:-4}; RUN=g100_drop_lr5e-05
torchrun --standalone --nproc-per-node $NPROC src/train_pretrain.py \
  --prepacked-dir data/games30m --init-checkpoint checkpoints/pt100/chsm8_latest.pt \
  --architecture llama --llama-qk-norm --depth 12 --d-model 960 --n-heads 15 --d-ff 3520 --dropout 0.0 \
  --use-flash-attn --dtype bf16 --legal-penalty-weight 0.5 --fused-adamw --batch-size 32 --weight-decay 0.1 \
  --warmup-ratio 0.05 --lr-schedule cosine --cosine-min-lr-ratio 0.0 --time-budget 6000 \
  --eval-every-frac 0.1 --eval-fixed --eval-steps 50 --log-every 200 --checkpoint-every 1e9 \
  --run-name $RUN --checkpoint-dir checkpoints/games/$RUN --results-dir results/games \
  --no-use-cross-attention --xattn-mode drop --lr 5e-05 --max-train-positions 1946812928 --lr-schedule-by-positions \
  --milestone-checkpoint-percent 0.2 --milestone-checkpoint-extra-percent 0
