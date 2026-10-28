#!/usr/bin/env python3
"""Direct, colour-balanced match: CHSM8 versus another chess model.

No engine is involved in move selection or scoring. Both sides are greedy:
each plays its single most likely *legal* move (for language models: the legal
continuation with the highest total log-probability). Each opening line from
the book is played twice with colours swapped, so the result measures the
models rather than the book. Games reaching --max-plies are scored as draws.

Opponent adapters (--opponent):
  nanogpt-char   character-level PGN nanoGPT (Karvonen ChessGPT, Kempner ChessFormer)
  hf-uci         Hugging Face causal LM over space-separated UCI (e.g. ChessGPT-2 UCI)
  hf-pgn         Hugging Face causal LM over PGN movetext (e.g. Waterhorse ChessGPT)
  maia2          Maia-2 board model, conditioned on --maia-elo for both players
  chesslfm       ChessLFM (mlabonne/LFM2.5-230M-Chess): value token, then legal-masked move token
  chesslfm-search  the same with the author's depth-3 search (root top 12, then top 6, value-head leaves, minimax)
  searchless     DeepMind searchless action-value transformer, 9M / 136M / 270M (scripts/searchless_server.py)
  lc0            Leela Chess Zero network with one node (also Maia-1 networks)
  maia3          Maia-3 / Chessformer 79M, argmax policy, --opp-elo for both players
  otter          Otter 15M, argmax policy, --opp-elo and --opp-tc
  allie2         Allie 2.0 (5.6B MoE, 0.69B active), argmax policy ("strongest" mode, no search), --opp-elo, --opp-tc
  matilda        Matilda: Maia-3 23M candidates + Stockfish (--matilda-depth) + re-ranker, argmax
  c1             C1-4B reasoning LM (vLLM, greedy, FINAL_ANSWER; illegal answers re-decoded constrained to legal moves)
  queen          QUEEN / PAWN-8 (SmolLM3-3B + Lc0 cross-attention, vLLM, greedy, the release's move policy)

CHSM8 side: a pretrained checkpoint (default) or, with --sft, a style-conditioned SFT/DPO checkpoint played without
a prompt (e.g. checkpoints of LegumMagister/chsm8).

    python src/benchmark_versus.py checkpoints/chsm8/chsm8_latest.pt \
        --opponent maia2 --opponent-path models/opponents/maia2_blitz/original/model.pt \
        --name maia2-blitz --games 200 --out results/versus/maia2
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_play import choose_model_moves_batched, load_checkpoint  # noqa: E402
from factored_tokenizer import ROW_BOS, move_to_row  # noqa: E402


def san_movetext(board, compact):
    """Movetext of board.move_stack: '1.e4 e5 2.Nf3' (compact) or '1. e4 e5 2. Nf3'."""
    replay, parts = chess.Board(), []
    for mv in board.move_stack:
        if replay.turn == chess.WHITE:
            parts.append(f"{replay.fullmove_number}." if compact else f"{replay.fullmove_number}.")
        parts.append(replay.san(mv))
        replay.push(mv)
    text = " ".join(parts)
    return re.sub(r"(\d+\.) ", r"\1", text) if compact else text


def next_prefix_marker(board, compact):
    """Text that precedes the side to move's next SAN move in the movetext."""
    if board.turn == chess.WHITE:
        return (" " if board.move_stack else "") + f"{board.fullmove_number}." + ("" if compact else " ")
    return " "


# ---- opponents ---------------------------------------------------------------

class NanoGptChar:
    """Character-level PGN nanoGPT; prompt ';1.e4 e5 2.Nf3 ...' (32-char vocab)."""

    def __init__(self, path, device, meta_path):
        sys.path.insert(0, str(ROOT / "external" / "chess_gpt_eval"))
        from nanogpt.model import GPT, GPTConfig
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
        self.model = GPT(GPTConfig(**ck["model_args"]))
        self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.block = ck["model_args"]["block_size"]
        self.stoi = pickle.load(open(meta_path, "rb"))["stoi"]
        self.device = device

    @torch.no_grad()
    def choose(self, boards, batch_size=512):
        cands = []
        for gi, b in boards:
            prefix = [self.stoi[c] for c in ";" + san_movetext(b, compact=True) + next_prefix_marker(b, compact=True)]
            for mv in b.legal_moves:
                cands.append({"g": gi, "mv": mv, "rest": [self.stoi[c] for c in b.san(mv)], "pre": list(prefix), "s": 0.0})
        todo, every = list(cands), list(cands)
        while todo:
            groups, nxt = {}, []
            for c in todo:
                groups.setdefault(len(c["pre"]), []).append(c)
            for grp in groups.values():
                for o in range(0, len(grp), batch_size):
                    ch = grp[o:o + batch_size]
                    x = torch.tensor([c["pre"][-self.block:] for c in ch], device=self.device)
                    lp = torch.log_softmax(self.model(x)[0][:, -1].float(), -1)
                    for r, c in enumerate(ch):
                        t = c["rest"].pop(0)
                        c["s"] += float(lp[r, t]); c["pre"].append(t)
                        if c["rest"]:
                            nxt.append(c)
            todo = nxt
        return best_per_game(every)


class HfCausal:
    """Hugging Face causal LM scoring every legal continuation of a text prompt."""

    def __init__(self, path, device, fmt):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).eval().to(device)
        self.fmt, self.device = fmt, device
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.max_len = getattr(self.model.config, "n_positions", None) or self.model.config.max_position_embeddings

    def texts(self, b, mv):
        if self.fmt == "uci":
            prefix = " ".join(m.uci() for m in b.move_stack)
            return prefix, (" " if prefix else "") + mv.uci()
        return san_movetext(b, compact=False), next_prefix_marker(b, compact=False) + b.san(mv)

    @torch.no_grad()
    def choose(self, boards, batch_tokens=65536):
        items = []
        for gi, b in boards:
            for mv in b.legal_moves:
                prefix, cont = self.texts(b, mv)
                p_ids = self.tok(prefix).input_ids if prefix else self.tok("").input_ids
                f_ids = self.tok(prefix + cont).input_ids
                k = 0
                while k < min(len(p_ids), len(f_ids)) and p_ids[k] == f_ids[k]:
                    k += 1
                k = max(k, 1)                          # token 0 is never scored
                if len(f_ids) > self.max_len:
                    raise RuntimeError(f"prompt of {len(f_ids)} tokens exceeds context {self.max_len}")
                items.append({"g": gi, "mv": mv, "ids": f_ids, "k": k, "s": 0.0})
        items.sort(key=lambda it: len(it["ids"]))
        i = 0
        while i < len(items):
            L = len(items[i]["ids"])
            n = max(1, batch_tokens // max(L, 1))
            ch = items[i:i + n]
            L = max(len(c["ids"]) for c in ch)
            x = torch.full((len(ch), L), self.pad, device=self.device)
            att = torch.zeros((len(ch), L), dtype=torch.long, device=self.device)
            for r, c in enumerate(ch):
                x[r, :len(c["ids"])] = torch.tensor(c["ids"], device=self.device)
                att[r, :len(c["ids"])] = 1
            lp = torch.log_softmax(self.model(input_ids=x, attention_mask=att).logits.float(), -1)
            for r, c in enumerate(ch):
                ids = torch.tensor(c["ids"], device=self.device)
                pos = torch.arange(c["k"], len(c["ids"]), device=self.device)
                c["s"] = float(lp[r, pos - 1, ids[pos]].sum())
            i += len(ch)
        return best_per_game(items)


class Maia2:
    """Maia-2 board policy; picks its highest-probability legal move."""

    def __init__(self, path, device, elo):
        from importlib.resources import files
        from maia2 import inference
        from maia2.main import MAIA2Model
        from maia2.utils import create_elo_dict, get_all_possible_moves, parse_args
        cfg = parse_args(str(files("maia2.configs").joinpath("maia2-training.yaml")))
        self.model = MAIA2Model(len(get_all_possible_moves()), create_elo_dict(), cfg)
        sd = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        self.model.eval().to(device)
        self.inf, self.prep, self.elo, self.device = inference, inference.prepare(), elo, device

    @torch.no_grad()
    def choose(self, boards):
        all_moves, elo_dict, rev = self.prep
        xs, es, eo, legal = [], [], [], []
        for _, b in boards:
            bi, s, o, lm = self.inf.preprocessing(b.fen(), self.elo, self.elo, elo_dict, all_moves)
            xs.append(bi); es.append(s); eo.append(o); legal.append(lm)
        logits, _, _ = self.model(torch.stack(xs).to(self.device), torch.tensor(es, device=self.device),
                                  torch.tensor(eo, device=self.device))
        logits = logits.float().masked_fill(torch.stack(legal).to(self.device) == 0, float("-inf"))
        out = {}
        for r, (gi, b) in enumerate(boards):
            uci = rev[int(logits[r].argmax())]
            if b.turn == chess.BLACK:                  # Maia-2 sees black positions mirrored
                uci = self.inf.mirror_move(uci)
            out[gi] = chess.Move.from_uci(uci)
        return out


class ChessLFM:
    """ChessLFM (Maxime Labonne, Liquid AI; mlabonne/LFM2.5-230M-Chess), reimplemented from the author's demo app:
    an 80-token board prompt, the argmax value token, then the move token masked to legal moves. With search=True,
    the app's depth-3 search: the 12 most likely root moves, the 6 most likely replies below, leaves scored by the
    value token, minimax (the app uses alpha-beta, which picks the same move; here the full tree is evaluated in
    batches across games). Like the app, it avoids repeating a position when 3+ points of material ahead."""

    CAST = ["-", "q", "k", "kq", "Q", "Qq", "Qk", "Qkq", "K", "Kq", "Kk", "Kkq", "KQ", "KQq", "KQk", "KQkq"]
    MAT = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

    def __init__(self, path, device, search=False, root_k=12, k=6):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval().to(device)
        self.ids = json.load(open(Path(path) / "token_ids.json"))
        self.vbins = torch.tensor([self.ids[f"<v:{i}>"] for i in range(64)], device=device)
        self.search, self.root_k, self.k, self.device = search, root_k, k, device

    def prompt(self, b, hist, rep):
        cells = []
        for row in b.board_fen().split("/"):
            for ch in row:
                cells += ["."] * int(ch) if ch.isdigit() else [ch]
        bits = (8 * b.has_kingside_castling_rights(chess.WHITE) + 4 * b.has_queenside_castling_rights(chess.WHITE)
                + 2 * b.has_kingside_castling_rights(chess.BLACK) + b.has_queenside_castling_rights(chess.BLACK))
        ep = b.ep_square
        toks = (["<|pos|>"] + [f"<c:{c}>" for c in cells] + [f"<stm:{'w' if b.turn else 'b'}>", f"<cast:{self.CAST[bits]}>",
                f"<ep:{chess.square_name(ep)[0] if ep is not None and b.has_legal_en_passant() else '-'}>",
                f"<hm:{min(b.halfmove_clock, 100) // 4}>", f"<rep:{min(max(rep, 0), 2)}>", "<|hist|>"]
                + ["<m:0000>"] * (8 - len(hist[-8:])) + [f"<m:{u}>" for u in hist[-8:]] + ["<|eval|>"])
        assert len(toks) == 80
        return [self.ids[t] for t in toks]

    @torch.no_grad()
    def last_logits(self, seqs, bs=4096):
        out = []
        for i in range(0, len(seqs), bs):
            x = torch.tensor(seqs[i:i + bs], device=self.device)
            out.append(self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[:, -1].float())
        return torch.cat(out)

    def value(self, prompts):
        """Win probability of the side to move, from the argmax value bin."""
        lg = self.last_logits(prompts)
        bins = lg[:, self.vbins].argmax(-1)
        return ((bins.float() + 0.5) / 64).tolist(), bins.tolist()

    def policy(self, nodes):
        """nodes: [(board, hist, rep)] -> (win prob, [(move, prob)] sorted by prob)."""
        prompts = [self.prompt(b, h, r) for b, h, r in nodes]
        v, bins = self.value(prompts)
        lg = self.last_logits([p + [self.ids[f"<v:{k}>"], self.ids["<|bestmove|>"]] for p, k in zip(prompts, bins)])
        out = []
        for i, (b, _, _) in enumerate(nodes):
            moves = list(b.legal_moves)
            l = lg[i, [self.ids[f"<m:{m.uci()}>"] for m in moves]]
            pr = torch.softmax(l, 0).tolist()
            out.append((v[i], sorted(zip(moves, pr), key=lambda t: -t[1])))
        return out

    @staticmethod
    def terminal(b):
        if b.is_checkmate():
            return 0.0
        if b.is_stalemate() or b.is_insufficient_material() or b.can_claim_draw():
            return 0.5
        return None

    def searched(self, roots):
        """Depth-3 minimax over the policy's top moves; returns candidates [(move, value)] best first, per root."""
        rp = self.policy(roots)
        level1 = []                                   # (root i, move, board, hist)
        for i, ((b, h, _), (_, pri)) in enumerate(zip(roots, rp)):
            for mv, _ in pri[:self.root_k]:
                c = b.copy(); c.push(mv); level1.append((i, mv, c, h + [mv.uci()]))
        def expand(nodes):                            # nodes [(board, hist)] -> children per node (or terminal value)
            live = [j for j, (b, _) in enumerate(nodes) if self.terminal(b) is None]
            pol = dict(zip(live, self.policy([(nodes[j][0], nodes[j][1], 0) for j in live]))) if live else {}
            kids = []
            for j, (b, h) in enumerate(nodes):
                if j not in pol:
                    kids.append(None); continue
                ch = []
                for mv, _ in pol[j][1][:self.k]:
                    c = b.copy(); c.push(mv); ch.append((c, h + [mv.uci()]))
                kids.append(ch)
            return kids
        k1 = expand([(c, h) for _, _, c, h in level1])
        l2 = [x for ch in k1 if ch for x in ch]
        k2 = expand(l2)
        leaves = [x for ch in k2 if ch for x in ch]
        lv_live = [j for j, (b, _) in enumerate(leaves) if self.terminal(b) is None]
        vals = dict(zip(lv_live, self.value([self.prompt(leaves[j][0], leaves[j][1], 0) for j in lv_live])[0])) if lv_live else {}
        leaf_score = [vals[j] if j in vals else self.terminal(leaves[j][0]) for j in range(len(leaves))]
        it = iter(leaf_score)
        s2 = []                                       # score of each level-2 node (side to move there)
        for j, (b, _) in enumerate(l2):
            ch = k2[j]
            s2.append(self.terminal(b) if ch is None else max([1 - next(it) for _ in ch] + [0.0]))
        it = iter(s2)
        s1 = []
        for j, (_, _, c, _) in enumerate(level1):
            ch = k1[j]
            s1.append(self.terminal(c) if ch is None else max([1 - next(it) for _ in ch] + [0.0]))
        res = [[] for _ in roots]
        for (i, mv, _, _), sc in zip(level1, s1):
            res[i].append((mv, 1 - sc))
        return [sorted(r, key=lambda t: -t[1]) for r in res]   # stable: ties keep the policy order

    def avoid_repeats(self, b, cands):
        mat = sum((1 if p.color == b.turn else -1) * self.MAT.get(p.piece_type, 0) for p in b.piece_map().values())
        if mat < 3:
            return cands
        seen = {self.key(x) for x in self.history_boards(b)}
        keep = []
        for mv in cands:
            c = b.copy(); c.push(mv)
            if self.key(c) not in seen:
                keep.append(mv)
        return keep or cands

    @staticmethod
    def key(b):
        return " ".join(b.fen().split()[:4])

    @staticmethod
    def history_boards(b):
        c = b.copy(); out = []
        while c.move_stack:
            c.pop(); out.append(c.copy(stack=False))
        return out

    def choose(self, boards):
        roots = []
        for _, b in boards:
            hist = [m.uci() for m in b.move_stack]
            rep = sum(self.key(x) == self.key(b) for x in self.history_boards(b))
            roots.append((b, hist, rep))
        if self.search:
            cands = [[mv for mv, _ in r] for r in self.searched(roots)]
        else:
            cands = [[mv for mv, _ in pri[:5]] for _, pri in self.policy(roots)]
        return {gi: self.avoid_repeats(b, c)[0] for (gi, b), c in zip(boards, cands)}


def load_reviewed(name, path, package_dir=None):
    """Import one reviewed reference file from a downloaded release (models/opponents/*), without putting the download
    on sys.path. package_dir: register it as a package first, for files with relative imports (the package's other
    modules then load only when the reviewed code imports them)."""
    import importlib.util
    if package_dir is not None and name.split(".")[0] not in sys.modules:
        pkg = name.split(".")[0]
        spec = importlib.util.spec_from_loader(pkg, loader=None, is_package=True)
        mod = importlib.util.module_from_spec(spec)
        mod.__path__ = [str(package_dir)]
        sys.modules[pkg] = mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mirror_sq(sq):
    return sq ^ 56


class Maia3:
    """Maia-3 (Chessformer; Monroe et al., 2026; UofTCSSLab/Maia3-79M), the authors' reference network
    (CSSLab/maia3, maia3/models.py) and the inputs of their UCI engine (maia3/uci.py, maia3/dataset.py): 8 board
    planes of 12 pieces, each position from its own mover's view (mirrored for black), oldest first and padded with the
    earliest; both Elos set to `elo`. With history=False, the current position fills all 8 slots (their UCI default).
    Plays the argmax legal move of the policy (their Temperature=0), no search."""

    BASE = dict(history=8, use_padding=True, include_time_info=False, dim_emb=128, num_blocks=8, mlp_ratio=2.0,
                dropout=0.0, use_gab=True, use_relative_bias=False, use_absolute_pe=False, use_rms_norm=True,
                omit_qkv_biases=True, activation="gelu", gab_gen_size=128, gab_per_square_dim=32, gab_intermediate_dim=128)
    SIZES = {"79m": dict(dim_vit=1024, head_hid_dim=1024, num_heads=32),   # maia3/model_registry.py
             "23m": dict(dim_vit=512, head_hid_dim=512, num_heads=16)}

    def __init__(self, path, device, elo, history=True, size="79m", amp=True):
        from types import SimpleNamespace
        src = ROOT / "models/opponents/maia3_src/repo/maia3/models.py"
        M = sys.modules.get("maia3_reviewed_models") or load_reviewed("maia3_reviewed_models", src)
        self.cfg = SimpleNamespace(**self.BASE, **self.SIZES[size])
        self.amp = amp
        self.model = M.MAIA3Model(self.cfg)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = self.model.load_state_dict({k.replace("smolgen", "gab"): v for k, v in sd.items()},
                                                         strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        self.model.eval().to(device)
        self.device, self.elo, self.history = device, elo, history

    @staticmethod
    def planes(b):
        t = torch.zeros(64, 12)
        mirror = b.turn == chess.BLACK
        for sq, p in b.piece_map().items():
            white = p.color == chess.WHITE
            if mirror:
                sq, white = mirror_sq(sq), not white
            t[sq, p.piece_type - 1 + (0 if white else 6)] = 1
        return t

    def tokens(self, b):
        H = self.cfg.history
        if self.history:
            c, hist = b.copy(), [self.planes(b)]
            while c.move_stack and len(hist) < H:
                c.pop(); hist.append(self.planes(c))
            hist = hist[::-1]
        else:
            hist = [self.planes(b)]
        hist = [hist[0]] * (H - len(hist)) + hist
        return torch.cat(hist + [torch.zeros(64, 1)], 1)

    @staticmethod
    def index(mv, turn):
        f, t = (mv.from_square, mv.to_square) if turn == chess.WHITE else (mirror_sq(mv.from_square), mirror_sq(mv.to_square))
        if mv.promotion:
            return 4096 + (chess.square_file(f) * 8 + chess.square_file(t)) * 4 + "qrbn".index(chess.piece_symbol(mv.promotion))
        return f * 64 + t

    @torch.no_grad()
    def logits(self, boards):
        x = torch.stack([self.tokens(b) for _, b in boards]).to(self.device)
        elo = torch.full((len(boards),), self.elo, dtype=torch.long, device=self.device)
        with torch.autocast("cuda", enabled=self.amp and str(self.device).startswith("cuda")):
            return self.model(x, elo, elo)[0].float().cpu()

    def choose(self, boards):
        logits = self.logits(boards)
        out = {}
        for r, (gi, b) in enumerate(boards):
            legal = list(b.legal_moves)
            out[gi] = legal[int(logits[r, [self.index(m, b.turn) for m in legal]].argmax())]
        return out


class Otter:
    """Otter (Kumar 2026; peargentlabs/otter-chess), the authors' reference network (PeargentLabs/otter-chess,
    package/otter_chess/model.py) and the inputs of their API (api.py) with the board planes of their web app
    (docs/lib/play/chess-utils.ts boardToTensor; the Python package calls a C extension for it): 18x8x8 planes from
    the mover's view, the last 20 moves (mirrored for black's), both Elos `elo` (bucket 10 is 2000+), time control
    `tc` and clock fraction `clock` (the API's default 0.5). Plays the argmax legal move, no search."""

    def __init__(self, path, device, elo, tc="600+0", clock=0.5):
        from safetensors.torch import load_file
        repo = ROOT / "models/opponents/otter-chess_src/repo/package/otter_chess"
        M = load_reviewed("otter_reviewed_model", repo / "model.py")
        self.model = M.StrongPolicyModel(history_k=20)
        self.model.load_state_dict(load_file(path))
        self.model.eval().to(device)
        self.policy = json.load(open(repo / "vocab/policy_move_to_id.json"))
        self.hist = json.load(open(repo / "vocab/history_move_to_id.json"))
        bucket = 0 if elo < 1100 else 10 if elo >= 2000 else 1 + (elo - 1100) // 100
        base, inc = map(int, tc.split("+"))
        eff = base + 40 * inc
        tcb = 0 if eff < 60 else 1 if eff < 180 else 2 if eff < 600 else 3 if eff < 1800 else 4
        self.bucket, self.tcb, self.clock, self.device = bucket, tcb, clock, device

    @staticmethod
    def canon(uci, white):
        return uci if white else uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]

    @staticmethod
    def planes(b):
        t = torch.zeros(18, 8, 8)
        white = b.turn == chess.WHITE
        for sq, p in b.piece_map().items():
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            row = 7 - rank if white else rank
            own = (p.color == chess.WHITE) == white
            t[p.piece_type - 1 + (0 if own else 6), row, file] = 1
        us, them = (chess.WHITE, chess.BLACK) if white else (chess.BLACK, chess.WHITE)
        for ch, (c, side) in enumerate([(us, "k"), (us, "q"), (them, "k"), (them, "q")]):
            ok = b.has_kingside_castling_rights(c) if side == "k" else b.has_queenside_castling_rights(c)
            if ok:
                t[12 + ch] = 1
        if b.ep_square is not None:       # the FEN's en-passant field: python-chess's FEN shows it only if legal
            ep = b.ep_square if b.has_legal_en_passant() else None
            if ep is not None:
                rank = chess.square_rank(ep)
                t[16, 7 - rank if white else rank, chess.square_file(ep)] = 1
        if white:
            t[17] = 1
        return t

    @torch.no_grad()
    def choose(self, boards):
        n = len(boards)
        ids = torch.zeros(n, 20, dtype=torch.long)
        mask = torch.zeros(n, 20, dtype=torch.bool)
        for r, (_, b) in enumerate(boards):
            replay, canon = chess.Board(), []
            for mv in b.move_stack:
                canon.append(self.canon(mv.uci(), replay.turn == chess.WHITE)); replay.push(mv)
            win = canon[-20:]
            for i, m in enumerate(win, start=20 - len(win)):
                ids[r, i], mask[r, i] = self.hist.get(m, 0), True
        x = torch.stack([self.planes(b) for _, b in boards]).to(self.device)
        e = torch.full((n,), self.bucket, dtype=torch.long, device=self.device)
        tc = torch.full((n,), self.tcb, dtype=torch.long, device=self.device)
        clock = torch.tensor([[self.clock, 0.0]] * n, device=self.device)
        logits = self.model(x, ids.to(self.device), mask.to(self.device), e, e, tc, clock)[0].float().cpu()
        out = {}
        for r, (gi, b) in enumerate(boards):
            legal = list(b.legal_moves)
            idx = [self.policy[self.canon(m.uci(), b.turn == chess.WHITE)] for m in legal]
            out[gi] = legal[int(logits[r, idx].argmax())]
        return out


class Allie2:
    """Allie 2.0 (Zhang et al.; the successor of Allie, ICLR 2025; yimingzhang/allie-2.0), through the authors'
    plain-PyTorch reference (model.py, tokens.py of the release; backend="torch", so the C++/CUDA-graph fast.py is
    never loaded), BF16 on the GPU. Header: time control `tc`, both Elos `elo`; no clocks (read as unknown, like the
    model's clockless training games). The authors' bot in its "strongest" mode with search 0: the argmax legal move.
    Each game keeps its key-value cache while it is being played, so a move costs the two new tokens; all games run
    in one batched step."""

    def __init__(self, path, device, elo, tc="600+0"):
        d = Path(path)
        load_reviewed("allie_reviewed.tokens", d / "tokens.py", package_dir=d)
        M = load_reviewed("allie_reviewed.model", d / "model.py", package_dir=d)
        self.T, self.M = sys.modules["allie_reviewed.tokens"], M
        self.model = M.Model(d, torch.device(device), torch.bfloat16, backend="torch")
        base, inc = map(int, tc.split("+"))
        self.head = self.T.header(base, inc, elo, elo)
        self.games, self.last = {}, set()

    @torch.no_grad()
    def choose(self, boards):
        T, items, meta = self.T, [], []
        for gi, b in boards:
            toks = self.head + [T.MOVE_ID[m.uci()] for m in b.move_stack]
            g = self.games.get(gi)
            if g is None or g["tokens"] != toks[:len(g["tokens"])]:
                g = self.games[gi] = dict(cache=self.M.Cache(self.model), tokens=[], boards=[T.START])
            n = min(g["cache"].n, len(toks) - 1)
            g["cache"].truncate(n)
            bs = g["boards"][:max(n, 1)]
            while len(bs) < len(toks):
                p = len(bs)
                bs.append(T.advance(bs[-1], toks[p]) if p >= T.HEADER else T.START)
            st = torch.tensor(np.frombuffer(b"".join(bs[n:]), np.uint8).reshape(-1, 68))
            items.append((g["cache"], torch.tensor(toks[n:]), torch.full((len(toks) - n, 3), -1.0), st))
            g["tokens"], g["boards"] = toks, bs
            meta.append((gi, b))
        z = self.M.step(self.model, items).float().cpu()
        seen = {gi for gi, _ in boards}            # free the caches of games absent from this call and the last
        self.games = {k: v for k, v in self.games.items() if k in seen or k in self.last}
        self.last = seen
        out = {}
        for r, (gi, b) in enumerate(meta):
            legal = list(b.legal_moves)
            out[gi] = legal[int(z[r, [T.MOVE_ID[m.uci()] for m in legal]].argmax())]
        return out


class Matilda:
    """Matilda (Carlson 2026; GarryChess/matilda-uci): frozen Maia-3 23M proposes its 16 most likely legal moves,
    Stockfish scores exactly those (one multipv search restricted to them, the release's UciSearchController), and
    the trained set-transformer re-ranker (TXTC, base_3k.pt) adds a residual to Maia-3's logits; plays the argmax
    (the release's Temperature 0). Reference code reviewed and imported from the release (matilda/model.py,
    features.py) with the featurization of matilda/inference.py and maia3_wrapper.py re-done in batch: Maia-3 in FP32
    with the real 8-position history, the pooled pre-logits state, the last block's per-square attention-bias map,
    the fp16 round-trip of the training shards. Both Elos `elo` (the release clamps to 1000-3200), time control `tc`.
    The release searches to depth 22 within a time-control cap (30 s in rapid); here depth `depth` (default 12), one
    thread, 16 MB hash, so 1,000-game runs fit a node. Stockfish 17.1, one process per game slot."""

    def __init__(self, path, device, elo, tc="600+0", depth=12, n=32,
                 engine=str(ROOT / "external/stockfish-sf17.1/src/stockfish")):
        import threading
        d = ROOT / "models/opponents/matilda_src/repo/src/matilda_uci/matilda"
        self.MM = load_reviewed("matilda_reviewed.model", d / "model.py", package_dir=d)
        self.F = load_reviewed("matilda_reviewed.features", d / "features.py", package_dir=d)
        sd = torch.load(Path(path) / "base_3k.pt", map_location="cpu", weights_only=True)
        self.txtc = self.MM.TXTC(n_players=sd["spe.weight"].shape[0] - 1)
        self.txtc.load_state_dict(sd, strict=True)
        self.txtc.eval().to(device)
        self.maia = Maia3(ROOT / "models/opponents/maia3_23m/maia3-23m.pt", device, max(1000, min(3200, elo)),
                          size="23m", amp=False)
        m = self.maia.model
        self.buf = {}
        m.last_ln.register_forward_hook(lambda mod, i, o: self.buf.__setitem__("hidden", o.detach()))
        self.mha = m.transformer.layers[-1].self_attn
        self.mha.register_forward_pre_hook(lambda mod, a, k: self.buf.__setitem__("query", k.get("query", a[0] if a else None)),
                                           with_kwargs=True)
        base, inc = map(float, tc.split("+"))
        self.tc, self.depth, self.engine, self.device = (base, inc), depth, engine, device
        self.engines, self.locks, self.n = {}, [threading.Lock() for _ in range(n)], n
        import atexit
        atexit.register(self.close)

    def close(self):
        for e in self.engines.values():
            try:
                e.quit()
            except Exception:
                pass
        self.engines = {}

    def score(self, k, b, cands):
        """Release's UciSearchController.score: (cp from the mover's view, mate = +-32000, 1-based multipv rank)."""
        import chess.engine
        with self.locks[k]:
            if k not in self.engines:
                self.engines[k] = chess.engine.SimpleEngine.popen_uci(self.engine)
            infos = self.engines[k].analyse(b, chess.engine.Limit(depth=self.depth), root_moves=cands,
                                            multipv=len(cands))
        infos = [infos] if isinstance(infos, dict) else infos
        out = {}
        for r, info in enumerate(infos):
            if info.get("pv"):
                sc = info["score"].pov(b.turn).score(mate_score=32000)
                out[info["pv"][0]] = (max(-32000, min(32000, sc)), r + 1)
        return out

    @torch.no_grad()
    def choose(self, boards):
        import concurrent.futures
        N, K = len(boards), 16
        logits = self.maia.logits(boards)                                     # (N, 4352), FP32
        hidden = self.buf["hidden"].float().cpu()
        bias = self.mha._sq_bias(self.buf["query"]).float().mean(1).mean(-1).cpu()   # (N, 64) mover's frame
        lm = torch.zeros(N, 4352)
        imp = torch.zeros(N, 64)
        ci = np.full((N, K), -1, np.int64); cl = np.zeros((N, K), np.float32); cv = np.zeros((N, K), np.float32)
        cands, legals = [], []
        for r, (_, b) in enumerate(boards):
            legal = list(b.legal_moves)
            idx = [self.maia.index(m, b.turn) for m in legal]
            lm[r, idx] = 1
            p = torch.softmax(logits[r, idx], 0)
            top = sorted(range(len(legal)), key=lambda i: -float(p[i]))[:K]
            cands.append([legal[i] for i in top]); legals.append((legal, idx))
            for j, i in enumerate(top):
                ci[r, j], cl[r, j], cv[r, j] = idx[i], math.log(max(float(p[i]), 1e-9)), 1
            v = bias[r] if b.turn == chess.WHITE else bias[r][[chess.square_mirror(s) for s in range(64)]]
            lo, hi = float(v.min()), float(v.max())
            imp[r] = (v - lo) / (hi - lo) if hi > lo else v
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n) as pool:
            scores = list(pool.map(lambda t: self.score(t[0] % self.n, t[1], t[2]),
                                   [(gi, gb, c) for (gi, gb), c in zip(boards, cands)]))
        sf_cp = np.full((N, K), -32001, np.int32); sf_rank = np.zeros((N, K), np.int32); sf_valid = np.zeros(N, np.int8)
        for r, (c, sc) in enumerate(zip(cands, scores)):
            for j, mv in enumerate(c):
                if mv in sc:
                    sf_cp[r, j], sf_rank[r, j] = sc[mv]; sf_valid[r] = 1
        d = dict(maia_logits=logits.numpy().astype(np.float16), maia_hidden=hidden.numpy().astype(np.float16),
                 maia_importance=imp.numpy().astype(np.float16), legal_mask=lm.numpy().astype(np.int8),
                 cand_idx=ci, cand_logit=cl, cand_valid=cv, tc_base=np.full(N, self.tc[0], np.float32),
                 tc_inc=np.full(N, self.tc[1], np.float32), sf_cp=sf_cp, sf_rank=sf_rank, sf_valid=sf_valid)
        T = {k: v.to(self.device) for k, v in self.F.tensors_sf(d, use_sf=True).items()}
        lp = self.txtc(T["ml"], T["hid"], T["imp"], T["lm"], T["ci"], T["tok"], T["val"], T["tcf"], style=False).cpu()
        return {gi: legal[int(lp[r, idx].argmax())] for r, ((gi, _), (legal, idx)) in enumerate(zip(boards, legals))}


class C1:
    """C1-4B (Tang et al., 2026; UofTCSSLab/C1-4B, Qwen3-4B after SFT on Stockfish-grounded explanations and RL), with
    the prompt of its model card (FEN, piece lists, legal moves, "Find the best move ...", "FINAL_ANSWER: <uci>"),
    greedy decoding (the card's recommendation) and at most 1,024 new tokens, batched in vLLM. A FINAL_ANSWER that is
    missing or illegal is replaced by decoding the answer again after the model's own reasoning, constrained to the
    legal moves (vLLM structured outputs); the count of such repairs is logged. No search."""

    NAMES = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}

    def __init__(self, path, device, gpu_mem=0.25, max_tokens=1024):
        # model.py's _env_fixes stubs torchvision/torchaudio for a broken venv; vLLM's container has working ones
        os.environ["LLM_CHESS_NO_ENV_FIXES"] = "1"           # for vLLM's spawned workers, which re-import __main__
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_BlockedFinder"]
        for k in [k for k in sys.modules if k.split(".")[0] in ("torchvision", "torchaudio")]:
            del sys.modules[k]
        from vllm import LLM, SamplingParams
        try:
            from vllm.sampling_params import StructuredOutputsParams as SO
            self.so = lambda ch: dict(structured_outputs=SO(choice=ch))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams as SO
            self.so = lambda ch: dict(guided_decoding=SO(choice=ch))
        self.SP = SamplingParams
        self.llm = LLM(model=str(path), dtype="bfloat16", gpu_memory_utilization=gpu_mem, max_model_len=4096, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.greedy = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        self.calls = self.repaired = 0

    def prompt(self, b):
        pieces = {}
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if p:
                pieces.setdefault(f"{'White' if p.color else 'Black'} {self.NAMES[p.piece_type]}", []).append(chess.square_name(sq))
        order = [f"{c} {t}" for c in ("White", "Black") for t in ("King", "Queen", "Rook", "Bishop", "Knight", "Pawn")]
        arrangement = ", ".join(f"{k}: {sorted(pieces[k])}" for k in order if k in pieces)
        legal = ", ".join(m.uci() for m in b.legal_moves)
        return (f"You are given a chess position in FEN: {b.fen()}.\n"
                f"Piece positions: {arrangement}\n"
                f"Legal moves: {legal}\n"
                "Find the best move for the side to play.\n"
                "Analyze step by step and explain your reasoning.\n"
                "Finish with a single line formatted EXACTLY as:\n"
                "FINAL_ANSWER: <answer>\n"
                "Use UCI notation (e.g., e2e4, c2b1q) for the final answer.")

    def choose(self, boards):
        convs = [[{"role": "user", "content": self.prompt(b)}] for _, b in boards]
        outs = self.llm.chat(convs, sampling_params=self.greedy, use_tqdm=False)
        res, redo = {}, []
        for (gi, b), conv, o in zip(boards, convs, outs):
            text = o.outputs[0].text
            m = re.findall(r"FINAL_ANSWER:\s*([a-h][1-8][a-h][1-8][qrbn]?)", text)
            try:
                mv = chess.Move.from_uci(m[-1]) if m else None
            except ValueError:
                mv = None
            if mv is not None and mv in b.legal_moves:
                res[gi] = mv
            else:
                head = text.rsplit("FINAL_ANSWER", 1)[0].rstrip()
                pre = self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                redo.append((gi, b, pre + head + "\nFINAL_ANSWER: "))
        if redo:
            sps = [self.SP(temperature=0.0, max_tokens=8, **self.so([m.uci() for m in b.legal_moves])) for _, b, _ in redo]
            fix = self.llm.generate([p for _, _, p in redo], sampling_params=sps, use_tqdm=False)
            for (gi, b, _), o in zip(redo, fix):
                u = o.outputs[0].text.strip()
                try:
                    mv = chess.Move.from_uci(u) if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", u) else None
                except ValueError:
                    mv = None
                res[gi] = mv if mv is not None and mv in b.legal_moves else next(iter(b.legal_moves))
        self.calls += len(boards); self.repaired += len(redo)
        if self.calls // 500 != (self.calls - len(boards)) // 500:
            print(f"C1: {self.repaired}/{self.calls} moves repaired (missing or illegal FINAL_ANSWER)", file=sys.stderr, flush=True)
        return res


class Queen:
    """QUEEN (Bhaskar, Cheng & Chen, 2026; princeton-nlp/queen_pawn-8, i.e. PAWN-8, the checkpoint renamed QUEEN, 2697
    in the paper): SmolLM3-3B with 16 cross-attention sublayers over an Lc0 BT5 encoder of the position and its history,
    served in vLLM by the release's own reviewed code (models/vllm/flamingo*.py, models/encoder.py, utils/*), loaded
    from the download directory (no other import path). The model card's prompt (inference_config.json), at most 2,048
    new tokens, greedy decoding (the card samples at 0.6; greedy matches this harness), and the release's move policy
    (utils/eval_policy.py: structured BEST_MOVE, else first legal move of the critical line, else first legal promising
    move, else a seeded random legal move); how often each source was used is logged. No search."""

    def __init__(self, path, device, max_tokens=2048, gpu_mem=0.5, max_seqs=200):
        os.environ["LLM_CHESS_NO_ENV_FIXES"] = "1"
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_BlockedFinder"]
        for k in [k for k in sys.modules if k.split(".")[0] in ("torchvision", "torchaudio")]:
            del sys.modules[k]
        # as the release's infer.py: in-process engine (the board rides a module-global registry), no FlashInfer
        # sampler, stable request ids; the model directory on the path for its `models`/`utils` packages
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.dir = Path(path).resolve()
        clash = [m for m in ("models", "utils") if m in sys.modules]
        assert not clash, f"modules {clash} already imported; Queen's packages would collide"
        sys.path.insert(0, str(self.dir))
        os.environ["PYTHONPATH"] = str(self.dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
        import torch as _t
        _load = _t.load
        _t.load = lambda *a, **k: _load(*a, **{**k, "weights_only": True})   # xattn.pt: tensors only, no pickled code
        from models.vllm.flamingo_generate import ChessFlamingoGenerator
        from utils.eval_policy import choose_move
        self.choose_move = choose_move
        self.settings = json.loads((self.dir / "inference_config.json").read_text())
        self.gen = ChessFlamingoGenerator(self.dir, self.dir / "lc0", dtype="bfloat16", gpu_memory_utilization=gpu_mem,
                                          max_model_len=max_tokens + 256, max_num_seqs=max_seqs, seed=self.settings["seed"],
                                          enforce_eager=True, use_v1_vllm=True)
        _t.load = _load
        self.max_tokens, self.max_seqs = max_tokens, max_seqs
        self.sources, self.calls, self.truncated, self.tokens = {}, 0, 0, 0

    def choose(self, boards):
        import hashlib
        fens = [b.fen() for _, b in boards]
        hists = []
        for _, b in boards:
            c, h = b.copy(stack=True), []
            while c.move_stack:
                c.pop(); h.append(c.fen())
            hists.append(h[::-1])
        seeds = [int.from_bytes(hashlib.blake2b(f"{self.settings['seed']}|{gi}|{b.ply()}".encode(), digest_size=8).digest(),
                                "big") % (2**31 - 1) for gi, b in boards]
        outs = []
        for o in range(0, len(boards), self.max_seqs):
            outs += self.gen.generate(fens[o:o + self.max_seqs], [self.settings["prompt"]] * len(fens[o:o + self.max_seqs]),
                                      hists[o:o + self.max_seqs], max_tokens=self.max_tokens, temperature=0.0,
                                      repetition_penalty=1.0, seeds=seeds[o:o + self.max_seqs])
        res = {}
        for (gi, b), fen, seed, out in zip(boards, fens, seeds, outs):
            try:
                mv, src, _ = self.choose_move(fen, out["text"], seed, self.settings["move_policy"])
            except ValueError:                      # unparseable UCI in the critical line
                mv, src = None, "parse_error"
            if mv is None or mv not in b.legal_moves:
                import random
                mv, src = random.Random(seed).choice(list(b.legal_moves)), f"{src}->random_legal"
            res[gi] = mv
            self.sources[src] = self.sources.get(src, 0) + 1
            self.truncated += out["finish_reason"] == "length"
            self.tokens += len(out["token_ids"])
        n0, self.calls = self.calls, self.calls + len(boards)
        if self.calls // 500 != n0 // 500 or n0 == 0:
            print(f"Queen: {self.calls} moves, {self.tokens / self.calls:.0f} tokens/move, {self.truncated} truncated, "
                  f"move sources {self.sources}", file=sys.stderr, flush=True)
        return res


def best_per_game(cands):
    best = {}
    for c in cands:
        if c["g"] not in best or c["s"] > best[c["g"]][1]:
            best[c["g"]] = (c["mv"], c["s"])
    return {g: mv for g, (mv, _) in best.items()}


# ---- match -------------------------------------------------------------------

class Searchless:
    """DeepMind's searchless 270M action-value transformer (Ruoss et al., 2024), served by scripts/searchless_server.py
    in a separate JAX process (environment .pydeps_jax) on the same GPU."""

    def __init__(self, path):
        import subprocess
        env = dict(os.environ, PYTHONPATH=str(ROOT / ".pydeps_jax"), XLA_PYTHON_CLIENT_PREALLOCATE="false")
        self.p = subprocess.Popen([sys.executable, str(ROOT / "scripts/searchless_server.py"), path], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        assert json.loads(self.p.stdout.readline()).get("ready")
        import atexit
        atexit.register(self.p.kill)

    def choose(self, boards):
        self.p.stdin.write(json.dumps({"games": [[m.uci() for m in b.move_stack] for _, b in boards]}) + "\n")
        moves = json.loads(self.p.stdout.readline())["moves"]
        return {i: chess.Move.from_uci(u) for (i, _), u in zip(boards, moves)}


class Lc0Policy:
    """Leela Chess Zero with one node of search (the network's own choice, no tree), as the searchless Lc0 baseline of
    Ruoss et al. (2024). A pool of lc0 processes on this GPU; each game always uses the same process."""

    def __init__(self, weights, n=None):
        import chess.engine
        import threading
        n = n or int(os.environ.get("LC0_ENGINES", 32))  # each process holds its own copy of the net on the GPU
        self.cmd = [str(ROOT / "external/lc0/build/gh200/lc0"), f"--weights={weights}", "--backend=cuda-fp16",
                    "--backend-opts=max_batch=8"]  # one-node play never batches more than a few positions
        self.engines, self.locks, self.n = {}, [threading.Lock() for _ in range(n)], n
        import atexit
        atexit.register(self.close)

    def close(self):
        for e in self.engines.values():
            try:
                e.quit()
            except Exception:
                pass
        self.engines = {}

    def _engine(self, k):
        import chess.engine
        if k not in self.engines:
            self.engines[k] = chess.engine.SimpleEngine.popen_uci(self.cmd)
        return self.engines[k]

    def choose(self, boards):
        import chess.engine
        import concurrent.futures

        def one(item):
            gi, b = item
            k = gi % self.n
            with self.locks[k]:
                return gi, self._engine(k).play(b, chess.engine.Limit(nodes=1), game=gi).move
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n) as pool:
            return dict(pool.map(one, boards))


OPPONENTS = ["nanogpt-char", "hf-uci", "hf-pgn", "maia2", "chesslfm", "chesslfm-search", "searchless", "lc0",
             "maia3", "otter", "allie2", "matilda", "c1", "queen"]


def add_opponent_args(ap):
    ap.add_argument("--opponent", required=True, choices=OPPONENTS)
    ap.add_argument("--opponent-path", required=True)
    ap.add_argument("--meta", default=str(ROOT / "external/chess_gpt_eval/nanogpt/out/meta.pkl"))
    ap.add_argument("--maia-elo", type=int, default=2000, help="Maia-2's Elo for both players")
    ap.add_argument("--opp-elo", type=int, default=2800, help="Maia-3, Otter, Allie 2.0: Elo for both players")
    ap.add_argument("--opp-tc", default="600+0", help="Otter, Allie 2.0: time control in the input")
    ap.add_argument("--maia3-no-history", action="store_true", help="Maia-3: current position only (UCI default)")
    ap.add_argument("--matilda-depth", type=int, default=12, help="Matilda: Stockfish depth over its 16 candidates")


def build_opponent(a, device):
    return {"nanogpt-char": lambda: NanoGptChar(a.opponent_path, device, a.meta),
            "hf-uci": lambda: HfCausal(a.opponent_path, device, "uci"),
            "hf-pgn": lambda: HfCausal(a.opponent_path, device, "pgn"),
            "maia2": lambda: Maia2(a.opponent_path, device, a.maia_elo),
            "chesslfm": lambda: ChessLFM(a.opponent_path, device),
            "chesslfm-search": lambda: ChessLFM(a.opponent_path, device, search=True),
            "searchless": lambda: Searchless(a.opponent_path),
            "lc0": lambda: Lc0Policy(a.opponent_path),
            "maia3": lambda: Maia3(a.opponent_path, device, a.opp_elo, history=not a.maia3_no_history),
            "otter": lambda: Otter(a.opponent_path, device, a.opp_elo, a.opp_tc),
            "allie2": lambda: Allie2(a.opponent_path, device, a.opp_elo, a.opp_tc),
            "matilda": lambda: Matilda(a.opponent_path, device, a.opp_elo, a.opp_tc, a.matilda_depth),
            "c1": lambda: C1(a.opponent_path, device),
            "queen": lambda: Queen(a.opponent_path, device)}[a.opponent]()


def play(chsm8, cfg, opp, device, specs, max_plies):
    games = []
    for gi, chsm8_white, opening in specs:
        b, rows = chess.Board(), [ROW_BOS]
        for tok in opening.split():
            san = tok.rsplit(".", 1)[-1]
            if not san:
                continue
            mv = b.parse_san(san)
            rows.append(move_to_row(mv, b)); b.push(mv)
        games.append({"gi": gi, "chsm8_white": chsm8_white, "opening": opening, "board": b, "rows": rows,
                      "book_plies": len(b.move_stack), "done": False})
    while True:
        live = [g for g in games if not g["done"]]
        for g in live:
            b = g["board"]
            if b.is_game_over(claim_draw=True) or len(b.move_stack) >= max_plies:
                g["done"] = True
        live = [g for g in games if not g["done"]]
        if not live:
            break
        mine = [(g["gi"], {"rows": g["rows"], "board": g["board"]}) for g in live
                if (g["board"].turn == chess.WHITE) == g["chsm8_white"]]
        theirs = [(g["gi"], g["board"]) for g in live if (g["board"].turn == chess.WHITE) != g["chsm8_white"]]
        moves = {}
        if mine:
            moves.update({i: mv for i, (mv, _) in choose_model_moves_batched(chsm8, cfg, device, mine).items()})
        if theirs:
            moves.update(opp.choose(theirs))
        for g in live:
            mv = moves.get(g["gi"])
            if mv is None or mv not in g["board"].legal_moves:
                raise RuntimeError(f"game {g['gi']}: no legal move chosen")
            g["rows"].append(move_to_row(mv, g["board"])); g["board"].push(mv)
    return games


def load_chsm8(path, device, sft):
    """Pretrained checkpoint via benchmark_play, or an SFT/DPO checkpoint via eval_sft.SFTPolicy with no prompt."""
    if not sft:
        return load_checkpoint(path, device)
    from types import SimpleNamespace
    from eval_sft import SFTPolicy
    pol = SFTPolicy(path, "", torch.device(device))
    return pol, SimpleNamespace(seq_len=512, d_model=0, use_cross_attention=False), {"step": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    add_opponent_args(ap)
    ap.add_argument("--sft", action="store_true", help="CHSM8 checkpoint is an SFT/DPO model (played without a prompt)")
    ap.add_argument("--chsm8-name", default="chsm8-pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=178)
    ap.add_argument("--parallel-games", type=int, default=100)
    ap.add_argument("--openings", default=str(ROOT / "external/chess_gpt_eval/logs/openings.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.games % 2 == 0
    device = "cuda:0"
    chsm8, cfg, ck = load_chsm8(a.checkpoint, device, a.sft)
    opp = build_opponent(a, device)
    book = [l.strip() for l in Path(a.openings).read_text().splitlines()[1:] if l.strip()]
    specs = [(i, i % 2 == 0, book[i // 2]) for i in range(a.games)]
    games = []
    for o in range(0, len(specs), a.parallel_games):
        games += play(chsm8, cfg, opp, device, specs[o:o + a.parallel_games], a.max_plies)
        print(f"{len(games)}/{a.games} games", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, pgns = [], []
    for g in sorted(games, key=lambda g: g["gi"]):
        b = g["board"]
        res = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "1/2-1/2"
        term = "normal" if b.is_game_over(claim_draw=True) else "max plies"
        pts = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
        pts = pts if g["chsm8_white"] else 1 - pts
        rows.append({"game": g["gi"] + 1, "chsm8_color": "white" if g["chsm8_white"] else "black",
                     "result": res, "chsm8_points": pts, "termination": term, "plies": len(b.move_stack),
                     "opening": g["opening"]})
        pg = chess.pgn.Game.from_board(b)
        pg.headers.update({"White": a.chsm8_name if g["chsm8_white"] else a.name,
                           "Black": a.name if g["chsm8_white"] else a.chsm8_name, "Result": res,
                           "Round": str(g["gi"] + 1), "Termination": term})
        pgns.append(str(pg))
    n = len(rows); s = sum(r["chsm8_points"] for r in rows) / n
    sd = math.sqrt(sum((r["chsm8_points"] - s) ** 2 for r in rows) / (n - 1)) / math.sqrt(n)
    diff = lambda p: 400 * math.log10(p / (1 - p)) if 0 < p < 1 else float("inf") * (1 if p >= 1 else -1)
    summary = {"chsm8_checkpoint": a.checkpoint, "opponent": a.name, "opponent_path": a.opponent_path,
               "protocol": "greedy legal move both sides, book openings played with both colours, "
                           f"max {a.max_plies} plies (then draw), no engine",
               "games": n, "chsm8_wins": sum(r["chsm8_points"] == 1 for r in rows),
               "draws": sum(r["chsm8_points"] == 0.5 for r in rows),
               "chsm8_losses": sum(r["chsm8_points"] == 0 for r in rows),
               "max_ply_draws": sum(r["termination"] == "max plies" for r in rows),
               "chsm8_score": s, "elo_diff": diff(s),
               "elo_diff_95ci": [diff(max(1e-9, s - 1.96 * sd)), diff(min(1 - 1e-9, s + 1.96 * sd))],
               "results": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "games.pgn").write_text("\n\n".join(pgns) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
    # Results are written; skip interpreter shutdown, where engine pools (python-chess event-loop threads, e.g.
    # Matilda's Stockfish processes) could wait on each other forever. Child engines get EOF and exit.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
