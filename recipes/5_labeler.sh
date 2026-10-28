#!/bin/bash
# chsm8-labeler-gemma3-1b (clean): Gemma-3-1B full fine-tune on the 2.40M clean teacher sets, 1 epoch, 4 GPUs, about 5 h.
# Our exact command (slurm/labeler_train_full.sbatch, DATA = the clean distill set).
set -euo pipefail
NPROC=${NPROC:-4}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=true \
torchrun --standalone --nproc-per-node $NPROC src/labeler/train.py --trial full_gemma3_1b_clean --model google/gemma-3-1b-it \
  --lora-r 0 --lr 5e-5 --tag 0 --data data/labeler --out runs/labeler \
  --n-train 0 --epochs 1 --save 1 --save-every-min 60 --n-gen 384 --gen-split test --warmup 200 "$@"
