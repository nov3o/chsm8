#!/bin/bash
# Hugging Face datasets -> the exact on-disk formats our runs trained on. CPU only. Run from the repo root.
#   bash recipes/0_data.sh            everything (about 2 TB, mostly the packed pretraining corpus)
#   STAGES="post players" bash recipes/0_data.sh
set -euo pipefail
W=${WORKERS:-$(nproc)}
STAGES=${STAGES:-"pre post players labeler"}
for s in $STAGES; do case $s in
pre)  # chsm8-pre-training -> data/2b (tier 2b) -> data/3 (factorized rows) -> data/3p (packed, shuffled)
  huggingface-cli download LegumMagister/chsm8-pre-training --repo-type dataset --include "data/*" --local-dir data/hf/pre
  SRC=data/hf/pre/data WORK=data/pre SEQ_LEN=512 LEGAL=1 SHUFFLE=1 WORKERS=$W bash prep/pretraining/prepare.sh
  ln -sfn pre/u16 data/2b; ln -sfn pre/3p-shuffled data/3p ;;
post)  # chsm8-post-training + moves -> data/sft30m_v21 (SFT, DPO) and data/games30m (player-base)
  python prep/posttraining.py --pre data/hf/pre
  python src/build_sft_corpus.py data/chsm8-post-training-unlabeled/data --out data/sft30m_v21 --workers $W
  python src/build_games_packed.py --source pool --out data/games30m --workers $W ;;
players)  # chsm8-top-players -> data/players/<key> (player fine-tunes) and data/sft_players (player prompts in SFT)
  python prep/players.py
  mkdir -p data && cp -r prep/openings data/   # the Lichess chess-openings table (CC0) as we used it
  for f in data/top_players/games/*.jsonl; do k=$(basename $f .jsonl)
    python src/build_games_packed.py --source players --inputs $f --out data/players/$k --holdout 2000 --repeat-to 20000; done
  python prep/players.py --sft   # SFT player prompts: the 99-player crawl of our run (after the 14 packs above)
  python src/build_player_corpus.py --out data/sft_players --workers $W ;;
labeler)  # chsm8-distillation -> data/labeler/{train,val,test}.jsonl
  python prep/labeler.py ;;
esac; done
