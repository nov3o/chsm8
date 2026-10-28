#!/bin/bash
# chsm8-pt: one pass over 78.32B positions, 4 GPUs (GH200), about 40 h. Our exact command (scripts/run_full_pretrain_4x12h.sh);
# we ran it as three 12-hour legs with --resume, which this loop reproduces.
set -euo pipefail
NPROC=${NPROC:-4}; RUN=${RUN:-full_llama_d12w960ff3520_qknorm}
for leg in 1 2 3; do
  torchrun --standalone --nproc_per_node=$NPROC src/train_pretrain.py \
    --prepacked-dir data/3p --prepacked-without-replacement --prepacked-sequential \
    --architecture llama --llama-qk-norm --depth 12 --d-model 960 --n-heads 15 --d-ff 3520 --dropout 0.0 \
    --use-cross-attention --fast-dummy-cross-attn --use-flash-attn --dtype bf16 \
    --legal-penalty-weight 0.5 --lr 5e-4 --weight-decay 0.1 --fused-adamw \
    --batch-size 32 --warmup-ratio .05 --time-budget $((leg * 43200)) \
    --max-train-positions 78320291328 --lr-schedule-by-positions \
    --checkpoint-dir checkpoints/$RUN --checkpoint-every 1800 --keep-checkpoints 2 \
    --eval-every 21600 --eval-steps 20 --log-every 200 --resume \
    --run-name $RUN --results-dir results/full_training
done
mkdir -p checkpoints/pt100 && cp checkpoints/$RUN/latest.pt checkpoints/pt100/chsm8_latest.pt
