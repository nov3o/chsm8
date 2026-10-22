# CHSM8

**A chess model you can talk to.** Tell it how to play — *"attack the king at any cost"*, *"solid, safe, risk-free
chess"*, *"You are Magnus Carlsen in his prime"* — and it plays that way. 210M parameters, no search, every move legal.

```python
from chsm8.play import Player

p = Player.from_pretrained("LegumMagister/chsm8")
p.move("1. e4 c5 2. Nf3 d6 3. d4", prompt="attack the king at any cost")   # a legal move, in UCI
```

Everything is open: the weights, the 1.07B games, the 30M described game sides with all their labels, the labeler
that wrote them, and the code below, which rebuilds each model from scratch.

## Contents

1. [Install](#1-install)
2. [Play](#2-play)
3. [The tokenizer](#3-the-tokenizer)
4. [The nano model: the whole recipe on a laptop](#4-the-nano-model-the-whole-recipe-on-a-laptop)
5. [Pretraining](#5-pretraining)
6. [Writing the descriptions](#6-writing-the-descriptions)
7. [SFT: learning to follow text](#7-sft-learning-to-follow-text)
8. [DPO: making the text matter](#8-dpo-making-the-text-matter)
9. [Player models](#9-player-models)
10. [Evaluation](#10-evaluation)
11. [Models and datasets](#11-models-and-datasets)
12. [What's in the repo](#12-whats-in-the-repo)

## 1. Install

```bash
git clone https://github.com/nov3o/chsm8 && cd chsm8
pip install "git+https://huggingface.co/LegumMagister/chsm8-tokenizer"   # the tokenizer, written in Rust (https://rustup.rs)
pip install -e .
huggingface-cli login                                                     # for the text encoder, see below
```

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

## 4. The nano model: the whole recipe on a laptop

[`notebooks/nano.ipynb`](notebooks/nano.ipynb) runs pretraining and SFT on a CPU in a few minutes with a 1M-parameter
model. It's the same code and the same loss as the real thing; only the sizes in `configs/nano*.yaml` are small.
Raise `LIMIT` at the top for more data.

The same steps from the command line:

```bash
# 3,000 games from one month of Lichess -> packed training records
python scripts/data/prepare_pretraining.py --out data/nano --months 2026-06 --max-games 3000

# pretrain a 4-layer, 128-wide model on them
python scripts/train/pretrain.py configs/nano.yaml

# 300 described game sides per split -> SFT with text
python scripts/data/prepare_posttraining.py --sides data/nano_sides --max-sides 300
python scripts/train/sft.py configs/nano_sft.yaml
```

## 5. Pretraining

**chsm8-pt**: a 12-layer decoder (d=960) trained from scratch on 1.07B filtered Lichess games, games only, no text.
One pass over 78.3B positions on 4 GH200 GPUs, about 40 hours. Result: 1837 Elo.

```bash
# download chsm8-pre-training (162 months of Lichess) and turn every game into rows; ~1.5 TB
python scripts/data/prepare_pretraining.py --out data/pt

# train; the config holds our exact settings
torchrun --nproc-per-node 4 scripts/train/pretrain.py configs/pt.yaml
```

Every config value can be overridden from the command line, e.g. `--batch_size=16`.

## 6. Writing the descriptions

To follow text, the model needs games paired with descriptions of how they were played. We wrote them in three steps.

**1. The teacher.** Gemma-4-31B reads one side of a game — its moves, plus a card of measurements computed from the
board (development, sacrifices, king pressure, trades, ...) — and writes five descriptions of its style, from terse to
long. A rule-based check drops every set where a description contradicts the card (a "sacrifice" in a game with no
sacrifices). That gave 2.4M clean sets: [chsm8-distillation](https://huggingface.co/datasets/LegumMagister/chsm8-distillation).

```bash
pip install vllm
python scripts/label/teacher.py --games games.parquet --out teacher.jsonl --tp 4
```

**2. The labeler.** A full fine-tune of Gemma-3-1B on those 2.4M sets: one epoch, 4 GPUs, about 10 hours.

```bash
torchrun --nproc-per-node 4 scripts/label/train_labeler.py --out runs/labeler
```

**3. Labeling at scale.** The labeler describes 30M game sides, five descriptions each:
[chsm8-post-training](https://huggingface.co/datasets/LegumMagister/chsm8-post-training).

```bash
python scripts/label/label.py --games games.parquet --out labels.jsonl   # uses LegumMagister/chsm8-labeler-gemma3-1b
```

The prompt and the measurement card are in [`scripts/label/prompt.py`](scripts/label/prompt.py), the check in
[`scripts/label/grounding.py`](scripts/label/grounding.py).

## 7. SFT: learning to follow text

**chsm8-sft**: chsm8-pt plus cross-attention to the text encoder in its top 6 layers. One pass over 27.1M described
training sides, each with all five descriptions, plus player games with prompts like *"play like Magnus Carlsen"*.
The loss is on the described side's moves only; the opponent's moves are context. 8 GPUs, about 3 hours.

```bash
# described sides from chsm8-post-training, moves joined from chsm8-pre-training
python scripts/data/prepare_posttraining.py --sides data/sides --games data/games

# player games from chsm8-top-players
python scripts/data/prepare_players.py --packed data/players --sides data/player_sides

torchrun --nproc-per-node 4 scripts/train/sft.py configs/sft.yaml
```

Our SFT used the games of 99 top players; 14 of them are public, so this reproduces SFT with those 14.

## 8. DPO: making the text matter

**chsm8** (the final model): DPO on chsm8-sft. No sampling and no reward model: the chosen sample is a game under its
own description, the rejected one is the same game under another game's description. So the only thing the
preference can teach is what the text should change. Two anchors keep the chess intact: the SFT loss and a KL term to
the SFT model. One GPU, 20 minutes.

```bash
python scripts/train/dpo.py configs/dpo.yaml
```

Style adherence rises on every prompt we tested (queens traded early 53% → 67%, queenside castling 97.3% → 99.7%),
and strength without a prompt stays the same (1792 vs. 1797 Elo for SFT).

## 9. Player models

Fourteen models that each play like one player, without any prompt: Carlsen, Kasparov, Fischer, Karpov, Nakamura,
Caruana, Firouzja, and others. First, chsm8-pt continues on the 26M games of the post-training set (chsm8-player-base),
then one short fine-tune per player on that player's own moves.

```bash
torchrun --nproc-per-node 4 scripts/train/pretrain.py configs/player_base.yaml
python scripts/train/pretrain.py configs/player.yaml --data=data/players/kasparov --out=runs/player_kasparov
```

## 10. Evaluation

Elo against Stockfish 17.1 at `UCI_Elo` 2000, 0.05 s per move, 1,000 games with alternating colours:

```bash
python scripts/eval/elo.py --model LegumMagister/chsm8 --stockfish path/to/stockfish --games 1000
python scripts/eval/elo.py --model LegumMagister/chsm8 --prompt "attack the king at any cost" --games 200
python scripts/eval/elo.py --model LegumMagister/chsm8-player-carlsen --games 200
```

| model | Elo |
|---|---|
| chsm8-pt | 1837 |
| chsm8-sft | 1797 |
| chsm8 | 1792 |

## 11. Models and datasets

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

## 12. What's in the repo

```
chsm8/                  the library
  model.py                the model: decoder, factorized heads, text cross-attention, loss
  tokenizer.py            games -> rows (wraps chsm8_tok)
  text.py                 the frozen text encoder
  play.py                 choosing moves, playing Stockfish, Elo
  data.py                 packed records and game sides on disk
  checkpoint.py           saving, loading, converting checkpoints
  style.py                what SFT and DPO share
  players.py              the player prompts
  config.py               yaml configs with command-line overrides
scripts/
  data/                   Hugging Face datasets -> training files
  label/                  teacher, labeler training, labeling
  train/                  pretrain.py, sft.py, dpo.py
  eval/                   elo.py
configs/                our exact settings, one file per model
notebooks/              play, tokenizer, nano
```

Note: the batch order in SFT and DPO is close to our runs but not identical, so a rerun gives a model of the same
quality, not the same weights.

## License

MIT. The games come from the [Lichess database](https://database.lichess.org) (CC0).
