# Results

Everything we logged, in four files. Each file has one layout for all of its rows.

| file | what it holds |
|---|---|
| [`autoresearch.csv`](autoresearch.csv) | 299 search runs over 6 stages: the selection metric, other metrics, and every setting of the run |
| [`stockfish.csv`](stockfish.csv) | 875 matches of our models against Stockfish 17.1: wins, draws, losses, Elo |
| [`versus.json`](versus.json) | 67 head-to-head pairs between 20 models (wins, draws, losses of `a`) |
| [`versus.csv`](versus.csv) | the same as a 20x20 matrix: `wins-draws-losses` of the row model, blank = not played |

`config` holds every setting we logged for a run. Two stages logged fewer: pretraining round 1 (39 runs) saved
no per-run config, so only the parameter count and steps are there (the run name encodes the architecture, e.g.
`xattn0_d4_w256` = no cross-attention, depth 4, width 256), and the 6 labeler runs have no logged settings.
Local paths are removed.

Elo = engine Elo + 400 log10(S / (1 - S)), S = (wins + draws/2) / games; Stockfish at `UCI_Elo` 2000, 0.05 s per move.

## Released models against Stockfish

| model | games | wins | draws | losses | score | Elo |
|---|--:|--:|--:|--:|--:|--:|
| chsm8-pt | 999 | 139 | 283 | 577 | 0.281 | 1837 |
| chsm8-sft | 2,000 | 237 | 452 | 1311 | 0.232 | 1792 |
| chsm8 | 1,999 | 248 | 451 | 1300 | 0.237 | 1797 |
| chsm8-player-base | 1,000 | 217 | 259 | 524 | 0.346 | 1890 |
| chsm8-player-abdusattorov | 998 | 225 | 285 | 488 | 0.368 | 1906 |
| chsm8-player-carlsen | 999 | 214 | 300 | 485 | 0.364 | 1903 |
| chsm8-player-caruana | 999 | 216 | 293 | 490 | 0.363 | 1902 |
| chsm8-player-firouzja | 999 | 186 | 306 | 507 | 0.339 | 1884 |
| chsm8-player-fischer | 999 | 203 | 248 | 548 | 0.327 | 1875 |
| chsm8-player-karpov | 997 | 236 | 258 | 503 | 0.366 | 1905 |
| chsm8-player-kasparov | 999 | 228 | 262 | 509 | 0.359 | 1900 |
| chsm8-player-keymer | 1,000 | 176 | 333 | 491 | 0.343 | 1887 |
| chsm8-player-nakamura | 1,000 | 185 | 223 | 592 | 0.296 | 1850 |
| chsm8-player-polgar | 1,000 | 219 | 247 | 534 | 0.343 | 1887 |
| chsm8-player-praggnanandhaa | 995 | 167 | 308 | 520 | 0.323 | 1871 |
| chsm8-player-sindarov | 998 | 222 | 266 | 510 | 0.356 | 1897 |
| chsm8-player-so | 1,000 | 226 | 294 | 480 | 0.373 | 1910 |
| chsm8-player-wei-yi | 999 | 206 | 294 | 499 | 0.353 | 1895 |

chsm8-sft and chsm8 pool two 1,000-game runs each.

## Search runs per stage

| stage | runs | metric | best run | best value |
|---|--:|---|---|--:|
| Pretraining (round 1: 1 GPU x 1 h; rounds 2-3: 4 GPUs) | 59 | val_loss (lower is better) | `ar4_shape_d12w960_b32_lr5e4_wu05_fastx_fused_ls10` | 3.004 |
| Games-only continuation (player-base) | 16 | val_loss (lower is better) | `g10_drop_lr5e-5` | 2.8724 |
| SFT (rounds 1-7: 1 GPU; 8+: 4 GPUs) | 146 | val_label_ce (lower is better) | `d13_120m_lr5e-7_c8e-4` | 2.9004 |
| DPO screening (200 games per prompt) | 51 | elo_no_prompt (higher is better) | `s30rn_kl05` | 1865 |
| DPO re-evaluation (1,000 games) | 21 | elo_no_prompt (higher is better) | `s30r_nllrest` | 1829 |
| Labeler model choice | 6 | distillation_score_pct (higher is better) | `Gemma-3-1B full FT` | 94 |

## Head-to-head

67 pairs, 64,746 games. Score of the first model:

| a | b | games | wins | draws | losses | score |
|---|---|--:|--:|--:|--:|--:|
| allie2 | stockfish | 1,000 | 707 | 170 | 123 | 0.792 |
| c1-4b | stockfish | 1,000 | 0 | 12 | 988 | 0.006 |
| chessgpt2-uci-12x12x768 | stockfish | 997 | 79 | 288 | 630 | 0.224 |
| chesslfm | stockfish | 997 | 80 | 123 | 794 | 0.142 |
| chesslfm-search | stockfish | 985 | 353 | 283 | 349 | 0.502 |
| chsm8-player-base | allie2 | 1,000 | 107 | 210 | 683 | 0.212 |
| chsm8-player-base | c1-4b | 1,000 | 985 | 15 | 0 | 0.993 |
| chsm8-player-base | chessgpt2-uci-12x12x768 | 1,000 | 816 | 145 | 39 | 0.888 |
| chsm8-player-base | chesslfm | 1,000 | 915 | 37 | 48 | 0.933 |
| chsm8-player-base | chesslfm-search | 1,000 | 489 | 172 | 339 | 0.575 |
| chsm8-player-base | deepmind-searchless-136m | 1,000 | 0 | 207 | 793 | 0.103 |
| chsm8-player-base | deepmind-searchless-270m | 1,000 | 0 | 142 | 858 | 0.071 |
| chsm8-player-base | deepmind-searchless-9m | 1,000 | 3 | 253 | 744 | 0.130 |
| chsm8-player-base | karvonen-chessgpt-lichess16 | 1,000 | 756 | 154 | 90 | 0.833 |
| chsm8-player-base | kempner-chessformer-50m-1500 | 1,000 | 861 | 78 | 61 | 0.900 |
| chsm8-player-base | maia2-blitz | 1,000 | 601 | 233 | 166 | 0.718 |
| chsm8-player-base | maia3-79m | 1,000 | 55 | 203 | 742 | 0.157 |
| chsm8-player-base | matilda-d12 | 1,000 | 1 | 80 | 919 | 0.041 |
| chsm8-player-base | otter-15m | 1,000 | 472 | 178 | 350 | 0.561 |
| chsm8-player-base | queen | 200 | 7 | 55 | 138 | 0.172 |
| chsm8-player-base | stockfish | 1,000 | 217 | 259 | 524 | 0.346 |
| chsm8-player-base | waterhorse-chessgpt-base-v1 | 1,000 | 895 | 83 | 22 | 0.936 |
| chsm8-player-carlsen | allie2 | 1,000 | 99 | 197 | 704 | 0.198 |
| chsm8-player-carlsen | c1-4b | 1,000 | 988 | 12 | 0 | 0.994 |
| chsm8-player-carlsen | chessgpt2-uci-12x12x768 | 1,000 | 844 | 133 | 23 | 0.910 |
| chsm8-player-carlsen | chesslfm | 1,000 | 894 | 68 | 38 | 0.928 |
| chsm8-player-carlsen | chesslfm-search | 1,000 | 499 | 196 | 305 | 0.597 |
| chsm8-player-carlsen | deepmind-searchless-136m | 1,000 | 0 | 127 | 873 | 0.064 |
| chsm8-player-carlsen | deepmind-searchless-270m | 1,000 | 0 | 244 | 756 | 0.122 |
| chsm8-player-carlsen | deepmind-searchless-9m | 1,000 | 7 | 319 | 674 | 0.167 |
| chsm8-player-carlsen | karvonen-chessgpt-lichess16 | 1,000 | 739 | 144 | 117 | 0.811 |
| chsm8-player-carlsen | kempner-chessformer-50m-1500 | 1,000 | 896 | 86 | 18 | 0.939 |
| chsm8-player-carlsen | maia2-blitz | 1,000 | 614 | 205 | 181 | 0.717 |
| chsm8-player-carlsen | maia3-79m | 1,000 | 114 | 199 | 687 | 0.213 |
| chsm8-player-carlsen | matilda-d12 | 1,000 | 0 | 32 | 968 | 0.016 |
| chsm8-player-carlsen | otter-15m | 1,000 | 511 | 199 | 290 | 0.611 |
| chsm8-player-carlsen | queen | 200 | 8 | 60 | 132 | 0.190 |
| chsm8-player-carlsen | stockfish | 999 | 214 | 300 | 485 | 0.364 |
| chsm8-player-carlsen | waterhorse-chessgpt-base-v1 | 1,000 | 903 | 81 | 16 | 0.944 |
| chsm8-pt | allie2 | 1,000 | 66 | 137 | 797 | 0.135 |
| chsm8-pt | c1-4b | 1,000 | 972 | 28 | 0 | 0.986 |
| chsm8-pt | chessgpt2-uci-12x12x768 | 1,200 | 863 | 258 | 79 | 0.827 |
| chsm8-pt | chesslfm | 1,000 | 881 | 67 | 52 | 0.914 |
| chsm8-pt | chesslfm-search | 1,000 | 384 | 186 | 430 | 0.477 |
| chsm8-pt | deepmind-searchless-136m | 1,000 | 0 | 205 | 795 | 0.102 |
| chsm8-pt | deepmind-searchless-270m | 1,000 | 0 | 160 | 840 | 0.080 |
| chsm8-pt | deepmind-searchless-9m | 1,000 | 31 | 237 | 732 | 0.149 |
| chsm8-pt | karvonen-chessgpt-lichess16 | 1,200 | 768 | 256 | 176 | 0.747 |
| chsm8-pt | kempner-chessformer-50m-1500 | 1,200 | 942 | 201 | 57 | 0.869 |
| chsm8-pt | maia2-blitz | 1,200 | 616 | 331 | 253 | 0.651 |
| chsm8-pt | maia3-79m | 1,000 | 52 | 176 | 772 | 0.140 |
| chsm8-pt | matilda-d12 | 1,000 | 0 | 39 | 961 | 0.019 |
| chsm8-pt | otter-15m | 1,000 | 367 | 249 | 384 | 0.491 |
| chsm8-pt | queen | 200 | 5 | 47 | 148 | 0.142 |
| chsm8-pt | stockfish | 999 | 139 | 283 | 577 | 0.281 |
| chsm8-pt | waterhorse-chessgpt-base-v1 | 1,200 | 970 | 168 | 62 | 0.878 |
| deepmind-searchless-136m | stockfish | 999 | 933 | 65 | 1 | 0.967 |
| deepmind-searchless-270m | stockfish | 982 | 880 | 102 | 0 | 0.948 |
| deepmind-searchless-9m | stockfish | 996 | 834 | 146 | 16 | 0.911 |
| karvonen-chessgpt-lichess16 | stockfish | 999 | 190 | 269 | 540 | 0.325 |
| kempner-chessformer-50m-1500 | stockfish | 1,000 | 62 | 230 | 708 | 0.177 |
| maia2-blitz | stockfish | 998 | 71 | 101 | 826 | 0.122 |
| maia3-79m | stockfish | 997 | 747 | 153 | 97 | 0.826 |
| matilda-d12 | stockfish | 1,000 | 994 | 6 | 0 | 0.997 |
| otter-15m | stockfish | 999 | 431 | 198 | 370 | 0.530 |
| queen | stockfish | 200 | 159 | 36 | 5 | 0.885 |
| waterhorse-chessgpt-base-v1 | stockfish | 999 | 52 | 246 | 701 | 0.175 |
