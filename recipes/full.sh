#!/bin/bash
# CHSM8, the full pipeline: every stage of our real run, in order, with our exact commands (one recipe per stage).
# Run from the repo root:  bash recipes/full.sh              (DISTILL=1 bash recipes/full.sh: with distillation)
# GPU stages assume GH200-class GPUs; times are ours. Set NPROC / NGPU for your machine; multi-node: replace
# `torchrun --standalone` in the recipes with your launcher. Each stage can also be run on its own.
set -euo pipefail
cd "$(dirname "$0")/.."

# 0. Data (CPU). Hugging Face -> the formats we trained on: packed pretraining records (data/3p, about 1.5 TB), the
#    SFT/DPO corpus (data/sft30m_v21), the games-only corpus (data/games30m), the 14 player packs (data/players/<key>),
#    the player-prompt corpus (data/sft_players), and with DISTILL=1 the labeler's distillation set (data/labeler).
#    To rebuild chsm8-pre-training itself from the raw Lichess archive: recipes/raw_lichess.sh.
STAGES="pre post players$([ "${DISTILL:-0}" = 1 ] && echo " labeler")" bash recipes/0_data.sh

# 1. Pretraining: chsm8-pt. 78.3B positions, 4 GPUs, about 40 hours (three 12-hour legs with --resume). 1837 Elo.
bash recipes/1_pretrain.sh

# 2. Games-only continuation: chsm8-player-base. 26.0M post-training games, no text, 4 GPUs, about 65 minutes. 1890 Elo.
bash recipes/2_player_base.sh

# 3. Player models: 14 fine-tunes of chsm8-player-base on one player's own moves, 1 GPU, minutes each. 1850-1910 Elo.
bash recipes/3_players.sh

# 4-6. Distillation (optional, DISTILL=1): how the descriptions were made. Not needed to reproduce the models: the labels
#    (chsm8-post-training), the teacher sets (chsm8-teacher-labels, chsm8-distillation) and the labeler
#    (chsm8-labeler-gemma3-1b) are public, and stage 0 already downloads the labeled corpus.
if [ "${DISTILL:-0}" = 1 ]; then
  # 4. Teacher labels: Gemma-4-31B describes game sides, five descriptions each. Needs an OpenAI-compatible
  #    Gemma-4-31B endpoint (see the recipe). Cleaning the teacher's labels: recipes/label_cleaning.sh.
  bash recipes/4_teacher.sh
  # 5. Labeler: chsm8-labeler-gemma3-1b. Gemma-3-1B full fine-tune on the 2.40M clean teacher sets, 1 epoch,
  #    4 GPUs, about 5 hours.
  bash recipes/5_labeler.sh
  # 6. Labeling: the labeler describes the 30.1M game sides with vLLM, one worker per GPU. To choose the sides from
  #    the 1B games yourself: python prep/select_sides.py.
  bash recipes/6_label.sh
fi

# 7. SFT: chsm8-sft. One pass over 27.1M described sides + player prompts, 8 GPUs, 3.1 hours. 1792 Elo without a prompt.
bash recipes/7_sft.sh

# 8. DPO: chsm8. 20 minutes on 1 GPU. 1797 Elo without a prompt; style adherence up on every prompt.
bash recipes/8_dpo.sh

# 9. Elo: 1,000 games against Stockfish 17.1 at UCI_Elo 2000, 0.05 s per move. STOCKFISH = your binary.
#    All the evaluations of the paper: recipes/paper_eval.sh.
bash recipes/9_eval.sh games checkpoints/pt100/chsm8_latest.pt pt
bash recipes/9_eval.sh games checkpoints/games/g100_drop_lr5e-05/latest.pt player_base
bash recipes/9_eval.sh text checkpoints/sft_v21/sft_v21.pt sft
bash recipes/9_eval.sh text checkpoints/dpo_v21/v21_label_kl1.pt dpo
bash recipes/9_eval.sh text checkpoints/dpo_v21/v21_label_kl1.pt dpo_queens "trade queens as early as possible"
