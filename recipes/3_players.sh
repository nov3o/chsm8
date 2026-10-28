#!/bin/bash
# The 14 player models: chsm8-player-base fine-tuned on one player's own moves (loss on the focal side), lr 3e-5,
# two passes over the unique training games. 1 GPU each, minutes. Our exact command (slurm/games/player_final.sbatch).
#   bash recipes/3_players.sh             all 14
#   bash recipes/3_players.sh 1503014     Carlsen only
set -euo pipefail
INIT=checkpoints/games/g100_drop_lr5e-05/latest.pt
for KEY in ${@:-$(ls data/players)}; do
  PACK=data/players/$KEY; NAME=final_$KEY
  POS=$(python -c "import json;d=json.load(open('$PACK/summary.json'));print(int(2*d['unique_train_games']*d['tokens_train']/d['train']['games']))")
  python src/train_pretrain.py --architecture llama --llama-qk-norm --depth 12 --d-model 960 \
    --n-heads 15 --d-ff 3520 --dropout 0.0 --use-flash-attn --dtype bf16 --legal-penalty-weight 0.5 --fused-adamw \
    --batch-size 32 --no-use-cross-attention --xattn-mode drop --lr 3e-5 --warmup-ratio 0.05 --lr-schedule cosine \
    --cosine-min-lr-ratio 0.0 --lr-schedule-by-positions --max-train-positions "$POS" --time-budget 3000 \
    --eval-every-frac 0.1 --eval-fixed --eval-steps 20 --log-every 50 --checkpoint-every 100000 --loss-side focal \
    --milestone-checkpoint-percent 0 --milestone-checkpoint-extra-percent 0 --results-dir results/player_final \
    --prepacked-dir "$PACK" --init-checkpoint "$INIT" --run-name "$NAME" --checkpoint-dir "checkpoints/players/$NAME"
  for sp in val test; do python scripts/eval_player_fit.py checkpoints/players/$NAME/latest.pt "$PACK" --split $sp --tag "$NAME" --out results/player_fit.jsonl; done
done
