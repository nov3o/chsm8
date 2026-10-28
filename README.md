# CHSM8

**A chess model you can talk to.** Tell it how to play — *"attack the king at any cost"*, *"solid, safe, risk-free
chess"*, *"You are Magnus Carlsen in his prime"* — and it plays that way. 210M parameters, no search, every move legal.

```python
from chsm8.play import Player

p = Player.from_pretrained("LegumMagister/chsm8")
p.move("1. e4 c5 2. Nf3 d6 3. d4", prompt="attack the king at any cost")   # a legal move, in UCI
```

Everything is open: the weights, the 1.07B games, the 30M described game sides with all their labels, the labeler
that wrote them, and the exact code of our runs, which rebuilds each model from scratch.
Paper: [arXiv:2610.25356](https://arxiv.org/abs/2610.25356).

## Contents

1. [Install](#1-install)
2. [Play](#2-play)
3. [The tokenizer](#3-the-tokenizer)
4. [The nano model: the whole pipeline on a laptop](#4-the-nano-model-the-whole-pipeline-on-a-laptop)
5. [Data](#5-data)
6. [Pretraining](#6-pretraining)
7. [Writing the descriptions](#7-writing-the-descriptions)
8. [SFT: learning to follow text](#8-sft-learning-to-follow-text)
9. [DPO: making the text matter](#9-dpo-making-the-text-matter)
10. [Player models](#10-player-models)
11. [Evaluation](#11-evaluation)
12. [Models and datasets](#12-models-and-datasets)
13. [What's in the repo](#13-whats-in-the-repo)

## 1. Install

```bash
git clone https://github.com/nov3o/chsm8 && cd chsm8
pip install "git+https://huggingface.co/LegumMagister/chsm8-tokenizer"   # the tokenizer, written in Rust (https://rustup.rs)
pip install -e .
huggingface-cli login                                                     # for the text encoder, see below
```

For training on GPUs: `pip install -e ".[gpu]"` (flash-attention kernels); for labeling: `".[label]"` (vLLM); for the
teacher: `".[teacher]"`. Evaluation needs a [Stockfish 17.1](https://stockfishchess.org) binary.

The text encoder is [EmbeddingGemma-300m](https://huggingface.co/google/embeddinggemma-300m). It is gated: accept its
license on its page once, then log in. The games-only models (chsm8-pt, the player models) don't need it.

## 2. Play

```python
from chsm8.play import Player

p = Player.from_pretrained("LegumMagister/chsm8")
game = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6"

# every legal move with its probability, best first; the prompt changes the distribution
for prompt in ["", "attack the king at any cost", "play safe, trade pieces, aim for a draw"]:
    print(prompt or "(no prompt)", [m["san"] for m in p.scores(game, prompt)[:3]])

# one move: greedy, or sampled with a temperature
p.move(game, "sacrifice material for the initiative", temperature=1.0)
```

The game can be written in any notation: SAN, UCI, LAN, or a whole PGN with move numbers and comments.

To play a full game against Stockfish, with the settings we use for evaluation:

```python
from chsm8.play import play_stockfish

pgn = play_stockfish(p, "path/to/stockfish", elo=2000, model_white=True, prompt="attack the king at any cost")
print(pgn)
```

More in [`notebooks/play.ipynb`](notebooks/play.ipynb).

## 3. The tokenizer

Most chess models give every move string ("e4", "Nf3", "O-O") its own token. CHSM8 gives each move one row of four
small fields instead, and predicts each field with its own small head:

```
(src, dst, piece, promo)      e.g.  Ng1-f3  ->  (g1, f3, knight, none)
```

A move's score is the sum of its four log-probabilities, normalized over the legal moves only. So the model can't
play an illegal move, and in training it is penalized for any probability it puts on squares and pieces it doesn't
have.

```python
from chsm8 import tokenizer

tokenizer.tokenize("1. e4 e5 2. Nf3")      # any notation -> [1804, 2356, 1350]; illegal moves raise ValueError
rows, legal = tokenizer.analyze("1. e4 e5 2. Nf3")   # the rows so far + every legal next move
```

More in [`notebooks/tokenizer.ipynb`](notebooks/tokenizer.ipynb).

## 4. The nano model: the whole pipeline on a laptop

[`notebooks/nano.ipynb`](notebooks/nano.ipynb) runs every training stage on a CPU in about 15 minutes, with **the same
scripts as our real runs** (`src/`) on one month of Lichess and a 1M-parameter model. Only the sizes differ. Raise `LIMIT`
at the top for more data. The full-size version of each command is in [`recipes/`](recipes) and
[`notebooks/full.ipynb`](notebooks/full.ipynb).

## 5. Data

Our training scripts read packed binary formats, not the Hugging Face datasets directly. [`prep/`](prep) turns the
datasets back into exactly those formats, so the training code runs unchanged:

```bash
bash recipes/0_data.sh                    # everything, CPU only, about 2 TB
STAGES="post players" bash recipes/0_data.sh
```

| stage | from | to | read by |
|---|---|---|---|
| `pre` | chsm8-pre-training | `data/3p`: packed records of 513 rows `(kind, src, dst, piece, promo)` | `src/train_pretrain.py` |
| `post` | chsm8-post-training + moves | `data/sft30m_v21` (described sides), `data/games30m` (games) | `src/sft_ablate.py`, `src/dpo_sft.py`, `src/train_pretrain.py` |
| `players` | chsm8-top-players | `data/players/<key>` (one pack per player), `data/sft_players` | `src/train_pretrain.py`, `src/sft_ablate.py` |
| `labeler` | chsm8-distillation | `data/labeler/{train,val,test}.jsonl` | `src/labeler/train.py` |

The rebuilt player packs and SFT player corpus are byte-identical to the ones our models trained on.

To build chsm8-pre-training itself from the raw Lichess archive (download, filter, export), run `recipes/raw_lichess.sh`.
The 30.1M described sides (players rated 2200 or more) are the `(month, row_idx, side)` rows of chsm8-post-training; the
one-off script that selected them is not in this repo.

## 6. Pretraining

**chsm8-pt**: a 12-layer decoder (d=960) trained from scratch on 1.07B filtered Lichess games, games only, no text.
One pass over 78.3B positions on 4 GH200 GPUs, about 40 hours. Result: 1837 Elo.

```bash
bash recipes/1_pretrain.sh                # torchrun ... src/train_pretrain.py --prepacked-dir data/3p ... (our exact arguments)
```

## 7. Writing the descriptions

To follow text, the model needs games paired with descriptions of how they were played. We wrote them in three steps.

**1. The teacher.** Gemma-4-31B reads one side of a game (its moves plus a card of measurements computed from the board:
development, sacrifices, king pressure, trades, ...) and writes five descriptions of its style, from terse to long. A
rule-based check drops every set where a description contradicts the card. That gave 2.4M clean sets:
[chsm8-distillation](https://huggingface.co/datasets/LegumMagister/chsm8-distillation). The prompt builder and the
production script are in [`prompt-ablation/`](prompt-ablation); they talk to any OpenAI-compatible Gemma-4-31B endpoint.

```bash
bash recipes/4_teacher.sh                 # only needed to label new games
```

**2. The labeler.** A full fine-tune of Gemma-3-1B on those 2.4M sets: one epoch, 4 GPUs, about 5 hours.

```bash
bash recipes/5_labeler.sh                 # src/labeler/train.py
```

**3. Labeling at scale.** The labeler describes 30M game sides, five descriptions each, with vLLM:
[chsm8-post-training](https://huggingface.co/datasets/LegumMagister/chsm8-post-training).

```bash
bash recipes/6_label.sh                   # src/labeler/build_unlabeled.py, then src/labeler/label_shards.py per GPU
```

How the labels were cleaned (grounding check, clean filter, repair of contradicted bands, the Apertus relabel) is in
[`recipes/label_cleaning.sh`](recipes/label_cleaning.sh), with the exact scripts.

## 8. SFT: learning to follow text

**chsm8-sft**: chsm8-pt plus cross-attention to the frozen EmbeddingGemma-300m in its top 6 layers. One pass over 27.1M
described training sides, plus player games with prompts like *"play like Magnus Carlsen"*. The loss is on the described
side's moves only; the opponent's moves are context. 8 GPUs, 3.1 hours.

```bash
bash recipes/7_sft.sh                     # src/sft_ablate.py
```

The player prompts use the online games of 99 FIDE top-100 players, all in [chsm8-top-players](https://huggingface.co/datasets/LegumMagister/chsm8-top-players)
(`prep/players.py --sft` rebuilds our corpus byte for byte).

## 9. DPO: making the text matter

**chsm8** (the final model): DPO on chsm8-sft. No sampling and no reward model: the chosen sample is a game under its
own description, the rejected one is the same game under another game's description. So the only thing the
preference can teach is what the text should change. The SFT loss and a KL term to the SFT model keep the chess intact.
One GPU, 20 minutes.

```bash
bash recipes/8_dpo.sh                     # src/dpo_sft.py
```

Style adherence rises on every prompt we tested (queens traded early 53% → 67%, queenside castling 97.3% → 99.7%),
and strength without a prompt stays the same (1797 vs. 1792 Elo for SFT).

## 10. Player models

Fourteen models that each play like one player, without any prompt: Carlsen, Kasparov, Fischer, Karpov, Nakamura,
Caruana, Firouzja, and others. First chsm8-pt continues on the 26M games of the post-training set (chsm8-player-base,
1890 Elo), then one short fine-tune per player on that player's own moves.

```bash
bash recipes/2_player_base.sh             # src/train_pretrain.py, no text
bash recipes/3_players.sh                 # all 14; or: bash recipes/3_players.sh 1503014   (Carlsen)
```

## 11. Evaluation

Elo against Stockfish 17.1 at `UCI_Elo` 2000, 0.05 s per move, 240-ply cap, 1,000 games with alternating colours:

```bash
bash recipes/9_eval.sh games checkpoints/pt100/chsm8_latest.pt pt
bash recipes/9_eval.sh text checkpoints/dpo_v21/v21_label_kl1.pt dpo "trade queens as early as possible"
```

| model | Elo |
|---|---|
| chsm8-pt | 1837 |
| chsm8-sft | 1792 |
| chsm8 | 1797 |
| chsm8-player-base | 1890 |

The paper's other evaluations (matches against 16 public models, Bradley–Terry, self-play style, player prompts, the
labeler's distillation score, label diversity) are listed with their scripts in [`recipes/paper_eval.sh`](recipes/paper_eval.sh).

Every match we logged, all head-to-head games against the public models, and all 299 search runs with their
settings are in [`results/`](results/README.md).

## 12. Models and datasets

| model | what it is |
|---|---|
| [chsm8](https://huggingface.co/LegumMagister/chsm8) | the final model: SFT + DPO |
| [chsm8-sft](https://huggingface.co/LegumMagister/chsm8-sft) | after SFT |
| [chsm8-pt](https://huggingface.co/LegumMagister/chsm8-pt) | pretrained, games only |
| [chsm8-player-base](https://huggingface.co/LegumMagister/chsm8-player-base) | chsm8-pt continued on the post-training games |
| chsm8-player-{carlsen, kasparov, fischer, ...} | 14 player models |
| [chsm8-labeler-gemma3-1b](https://huggingface.co/LegumMagister/chsm8-labeler-gemma3-1b) | game side → five descriptions |
| [chsm8-tokenizer](https://huggingface.co/LegumMagister/chsm8-tokenizer) | any notation → moves, legality checked |

| dataset | what it is |
|---|---|
| [chsm8-pre-training](https://huggingface.co/datasets/LegumMagister/chsm8-pre-training) | 1.07B Lichess games, by month |
| [chsm8-post-training](https://huggingface.co/datasets/LegumMagister/chsm8-post-training) | 30M game sides × five descriptions |
| [chsm8-distillation](https://huggingface.co/datasets/LegumMagister/chsm8-distillation) | 2.4M clean teacher sets, for the labeler |
| [chsm8-game-labels](https://huggingface.co/datasets/LegumMagister/chsm8-game-labels) | every teacher set |
| [chsm8-top-players](https://huggingface.co/datasets/LegumMagister/chsm8-top-players) | games of the 14 public players |

## 13. What's in the repo

```
src/                    the training and evaluation code of our runs, as it ran
  train_pretrain.py       pretraining, the games-only continuation and the player fine-tunes
  sft_ablate.py           SFT
  dpo_sft.py              DPO
  model.py                the decoder, factorized heads, text cross-attention
  benchmark_play.py       games against Stockfish; eval_sft.py: the same for text models
  build_*.py              packed-data builders
  labeler/                labeler training (train.py), labeling (label_shards.py), prompts (build_data.py)
prompt-ablation/        the teacher: prompt and card builder (autolabel/), production labeling script
prep/                   Hugging Face datasets -> the formats above
recipes/                0_data.sh ... 9_eval.sh: our exact commands, one per stage; raw_lichess.sh,
                        label_cleaning.sh, paper_eval.sh
notebooks/              nano (the pipeline on a CPU), full (the recipes), play, tokenizer
chsm8/                  a small library to load the released models and play with them
results/                every logged match and search run
```

The scripts in `src/` and `prompt-ablation/` are copies of what ran. Changes: local paths made repo-relative, and
CPU fallbacks in `dpo_sft.py`, `eval_sft.py`, `benchmark_play.py` and `scripts/eval_player_fit.py` so the nano notebook
runs without a GPU (GPU behaviour unchanged). Runs on other hardware give models of the same quality, not the same weights.

The architecture and hyperparameter searches (autoresearch) ran on a copy of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted to chess; the harness is not included, its
runs and settings are in [`results/autoresearch.csv`](results/autoresearch.csv). Matches against public models use
[searchless_chess](https://github.com/google-deepmind/searchless_chess) and
[chess_gpt_eval](https://github.com/adamkarvonen/chess_gpt_eval) as external checkouts.

## License

MIT. The games come from the [Lichess database](https://database.lichess.org) (CC0).
