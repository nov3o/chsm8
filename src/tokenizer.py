"""
Chess move tokenizer with legal-move masking.

Vocabulary: special tokens + all unique SAN moves from corpus.
Legal-move masking: at each decoding step, logits for illegal moves are set to -inf
using python-chess to track board state.

Token sequence modes:
  plain    — "e4 e5 Nf3 Nc6"
  numbered — "1. e4 e5 2. Nf3 Nc6"   (move numbers take ~33% of tokens)
  colored  — "<w> e4 <b> e5 <w> Nf3 <b> Nc6"
"""

import json
from pathlib import Path
from typing import Literal

import chess
import torch

SequenceMode = Literal["plain", "numbered", "colored"]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<w>", "<b>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID, WHITE_ID, BLACK_ID = range(6)


class ChessTokenizer:
    def __init__(self, vocab_path: str | Path):
        with open(vocab_path) as f:
            vocab: list[str] = json.load(f)
        self.tok2id: dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.id2tok: list[str] = vocab
        self.vocab_size = len(vocab)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def encode_game(
        self,
        moves: list[str],          # SAN strings, e.g. ["e4","e5","Nf3"]
        mode: SequenceMode = "plain",
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        tokens: list[int] = []
        if add_bos:
            tokens.append(BOS_ID)

        for i, san in enumerate(moves):
            if mode == "numbered":
                move_num = i // 2 + 1
                dot = "." if i % 2 == 0 else "..."
                num_tok = f"{move_num}{dot}"
                tokens.append(self.tok2id.get(num_tok, UNK_ID))
            elif mode == "colored":
                tokens.append(WHITE_ID if i % 2 == 0 else BLACK_ID)

            tokens.append(self.tok2id.get(san, UNK_ID))

        if add_eos:
            tokens.append(EOS_ID)
        return tokens

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.id2tok[i] for i in ids if i < self.vocab_size]

    # ------------------------------------------------------------------ #
    # Legal-move masking
    # ------------------------------------------------------------------ #

    def legal_move_mask(self, board: chess.Board) -> torch.Tensor:
        """Return additive logit mask: 0 for legal moves, -inf elsewhere."""
        mask = torch.full((self.vocab_size,), float("-inf"))
        for move in board.legal_moves:
            san = board.san(move)
            if san in self.tok2id:
                mask[self.tok2id[san]] = 0.0
        mask[EOS_ID] = 0.0  # always allow ending the game
        return mask

    # ------------------------------------------------------------------ #
    # Build vocabulary from a move list file
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_vocab(
        moves_file: str | Path,
        out_path: str | Path,
        max_move_number: int = 300,
    ) -> "ChessTokenizer":
        """
        Build vocab from a file of unique SAN moves (one per line).
        Also adds move-number tokens for numbered mode.
        """
        with open(moves_file) as f:
            moves = sorted({l.strip() for l in f if l.strip()})

        num_tokens = [f"{n}." for n in range(1, max_move_number + 1)] + \
                     [f"{n}..." for n in range(1, max_move_number + 1)]
        vocab = SPECIAL_TOKENS + num_tokens + moves

        with open(out_path, "w") as f:
            json.dump(vocab, f, ensure_ascii=False)

        return ChessTokenizer(out_path)
