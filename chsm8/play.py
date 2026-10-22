"""Choosing moves: score every legal move, play against Stockfish.

The model scores a legal move as log p(src) + log p(dst) + log p(piece) + log p(promo) at the last position.
Only the legal moves (from the tokenizer) are scored, so every move it plays is legal. No search.
"""
import torch
import torch.nn.functional as F

from chsm8 import tokenizer

MAX_ROWS = 512


class Player:
    """A model and, for text models, its text encoder.

        p = Player.from_pretrained("LegumMagister/chsm8")
        p.scores("1. e4 c5 2. Nf3", "trade queens as early as possible")  # [{"uci", "san", "logp"}, ...] best first
        p.move("1. e4 c5 2. Nf3")                                          # greedy move, UCI
    """

    def __init__(self, model, encoder=None):
        self.model, self.encoder, self._text = model.eval(), encoder, {}

    @classmethod
    def from_pretrained(cls, repo="LegumMagister/chsm8", filename="model.pt", device=None):
        from chsm8.model import CHSM8
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = CHSM8.from_pretrained(repo, filename, device)
        encoder = None
        if model.cfg.text_layers:
            from chsm8.text import TextEncoder
            encoder = TextEncoder(device=device)
        return cls(model, encoder)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def text(self, prompt):
        if not prompt or self.encoder is None:
            return None
        if prompt not in self._text:
            self._text[prompt] = self.encoder([prompt])
        return self._text[prompt]

    @torch.no_grad()
    def scores(self, game="", prompt=""):
        """Game so far (any notation) -> every legal move with its log-probability over legal moves, best first."""
        rows, legal = tokenizer.analyze(game)
        if not legal:
            return []
        x = torch.tensor([rows[-MAX_ROWS:]], device=self.device)
        logits = self.model(x, self.text(prompt.strip()))
        lp = {k: F.log_softmax(v[0, -1].float(), -1) for k, v in logits.items()}
        r = torch.tensor([row for _, _, row in legal], device=self.device)
        s = (lp["src"][r[:, 1]] + lp["dst"][r[:, 2]] + lp["piece"][r[:, 3]] + lp["promo"][r[:, 4]]).log_softmax(0)
        out = [{"uci": u, "san": san, "logp": v} for (u, san, _), v in zip(legal, s.tolist())]
        return sorted(out, key=lambda m: -m["logp"])

    def move(self, game="", prompt="", temperature=0.0):
        sc = self.scores(game, prompt)
        if temperature <= 0:
            return sc[0]["uci"]
        p = torch.tensor([m["logp"] for m in sc]).div(temperature).softmax(0)
        return sc[int(torch.multinomial(p, 1))]["uci"]


def play_stockfish(player, stockfish="stockfish", elo=2000, seconds=0.05, model_white=True, prompt="", max_plies=240):
    """One game against Stockfish 17.1 at UCI_Elo `elo`, `seconds` per move, 1 thread: our evaluation setting.
    Returns a chess.pgn.Game; a game still running after max_plies gets result "*" and is left out of the Elo."""
    import chess
    import chess.engine
    import chess.pgn
    board = chess.Board()
    with chess.engine.SimpleEngine.popen_uci(stockfish) as engine:
        engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": elo})
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if board.turn == model_white:
                board.push_uci(player.move(" ".join(m.uci() for m in board.move_stack), prompt))
            else:
                board.push(engine.play(board, chess.engine.Limit(time=seconds)).move)
    game = chess.pgn.Game.from_board(board)
    game.headers["White"], game.headers["Black"] = ("chsm8", "stockfish") if model_white else ("stockfish", "chsm8")
    game.headers["Result"] = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    return game


def elo(games, engine_elo=2000):
    """Elo against the engine from finished games (chess.pgn.Game with White/Black = chsm8/stockfish):
    engine_elo + 400 log10(S / (1 - S)) for score S, with a 95% interval from the per-game standard error."""
    import math
    per = []
    for g in games:
        r, white = g.headers["Result"], g.headers["White"] == "chsm8"
        if r != "*":
            per.append(0.5 if r == "1/2-1/2" else float((r == "1-0") == white))
    n, s = len(per), sum(per) / len(per)
    se = math.sqrt(sum((x - s) ** 2 for x in per) / (n - 1) / n)
    f = lambda x: engine_elo + 400 * math.log10(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4)))
    return {"games": n, "score": s, "elo": round(f(s)), "ci95": (round(f(s - 1.96 * se)), round(f(s + 1.96 * se)))}
