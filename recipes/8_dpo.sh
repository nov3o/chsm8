#!/bin/bash
# chsm8 (DPO): label pairs (same moves, own vs another game's description), beta 0.5 + NLL 1 + KL 1, pinned no-text term,
# 20 min on 1 GPU. Our exact command (slurm/dpo_v21.sbatch, trial v21_label_kl1 in eval/dpo/v21.tsv).
set -euo pipefail
LAB=gemma-4-31b,gemma-4-31b-patched,gemma-3-1b-distill,gemma-3-1b-distill-patched,gemma-3-1b-distill-clean
TOKENIZERS_PARALLELISM=false python src/dpo_sft.py --trial v21_label_kl1 \
  --pair-mode label --beta 0.5 --nll-weight 1.0 --kl-weight 1.0 --freeze-dummy \
  --init checkpoints/sft_v21/sft_v21.pt --data data/sft30m_v21 --label-emb= --labelers $LAB --time-budget 1200 \
  --eval-every 1000000 --eval-games 1000 --save-dir checkpoints/dpo_v21 --results results/dpo_v21.jsonl "$@"
