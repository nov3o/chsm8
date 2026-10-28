#!/bin/bash
# The paper's evaluations, as exact scripts of our runs. GPU unless noted.
#   Elo vs Stockfish (games-only / text models)       src/benchmark_play.py, src/eval_sft.py          (recipes/9_eval.sh)
#   style adherence: queens, pawns, queenside, Elo    src/summarize_sft_eval.py over results/sft_eval/<model>   (CPU)
#   pooling two 1,000-game runs                       scripts/r2_merge.py, scripts/dpo_full_keys.py            (CPU)
#   self-play style evaluation                        scripts/selfplay_style.py
#   1,000-game matches vs public models               src/benchmark_versus.py (opponent wrappers inside; DeepMind's
#                                                     searchless_chess and Karvonen's chess_gpt_eval as external
#                                                     checkouts, Searchless via scripts/searchless_server.py)
#   public models vs Stockfish                        src/benchmark_opponent_sf.py
#   Bradley-Terry strength of all 20 models           scripts/make_bt.py (CPU)
#   Elo confidence intervals                          src/elo_stats.py (CPU); time per move: src/time_per_move.py
#   player fit (move match, CE)                       scripts/eval_player_fit.py
#   player prompts on held-out games                  scripts/eval_player_prompts.py
#   labeler: distillation score, speed, similarity    src/labeler/score_ckpt.py, bench_vllm.py, bench_gen.py,
#                                                     similarity.py, judge_aggregate.py, summarize.py; teacher vs itself:
#                                                     src/labeler/teacher_resample.py
#   label diversity (distinct strings per band)       scripts/label_uniqueness_data.py (CPU)
#   text-encoder probe                                scripts/embed_probe.py
# All logged results of these are in results/.
set -euo pipefail
echo "see the comments in this file"
