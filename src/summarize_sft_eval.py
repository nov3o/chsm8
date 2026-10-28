#!/usr/bin/env python3
"""Summarize slurm/sft_eval_all.sbatch output: Elo per prompt + behaviour of the model's side, measured from the PGNs.

    python3 src/summarize_sft_eval.py results/sft_eval/<tag>
Writes report.md, elo.png (strong red / weak blue / style purple / no prompt gray) and behaviour.png
(change vs no prompt per metric) into that directory."""
import io
import json
import math
import sys
from pathlib import Path

import chess
import chess.pgn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"strong": "#d63a2a", "weak": "#2a78d6", "style": "#8a5cd1", "none": "#8a8a85"}
METRICS = [  # key, label, format
    ("plies", "game length (plies)", "{:.0f}"),
    ("capt", "captures / own move", "{:.3f}"),
    ("checks", "checks / own move", "{:.3f}"),
    ("pawn", "pawn moves / own move", "{:.3f}"),
    ("q_off40", "queens traded by ply 40", "{:.0%}"),
    ("castled", "castled", "{:.0%}"),
    ("ooo", "castled queenside (of castled)", "{:.0%}"),
    ("early_q", "queen moved in first 5 own moves", "{:.0%}"),
    ("variety", "distinct first-8-own-move lines / game", "{:.2f}"),
    ("sicilian", "Sicilian as Black (1.e4 c5)", "{:.0%}"),
    ("cpl", "mean centipawn loss", "{:.0f}"),
]


def elo(w, d, l, engine=2000):
    n = w + d + l
    s = (w + d / 2) / n
    se = math.sqrt(max((w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n, 1e-12) / n)
    f = lambda x: engine + 400 * math.log10(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4)))
    return round(f(s)), (round(f(s - 1.96 * se)), round(f(s + 1.96 * se))), n


def behaviour(pgn_path, summary):
    games = []
    with open(pgn_path) as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            games.append(g)
    acc = {k: [] for k, _, _ in METRICS}; lines = set(); n_black_e4 = n_sic = 0
    for g in games:
        me = chess.WHITE if g.headers.get("White") == "model" else chess.BLACK
        b = g.board(); own = capt = checks = pawn = 0; castled = ooo = early_q = False; q_off40 = False; first = []
        for ply, mv in enumerate(g.mainline_moves()):
            mine = b.turn == me
            if mine:
                own += 1
                capt += b.is_capture(mv); pawn += b.piece_type_at(mv.from_square) == chess.PAWN
                if b.is_castling(mv):
                    castled = True; ooo = b.is_queenside_castling(mv)
                if b.piece_type_at(mv.from_square) == chess.QUEEN and own <= 5:
                    early_q = True
                if len(first) < 8:
                    first.append(mv.uci())
            if me == chess.BLACK and ply == 1:
                prev = b.peek().uci() if b.move_stack else ""
                if prev == "e2e4":
                    n_black_e4 += 1; n_sic += mv.uci() == "c7c5"
            b.push(mv)
            if mine and b.is_check():
                checks += 1
            if ply == 39:
                q_off40 = not b.pieces(chess.QUEEN, chess.WHITE) and not b.pieces(chess.QUEEN, chess.BLACK)
        if own == 0:
            continue
        acc["plies"].append(len(b.move_stack)); acc["capt"].append(capt / own); acc["checks"].append(checks / own)
        acc["pawn"].append(pawn / own); acc["q_off40"].append(q_off40); acc["castled"].append(castled)
        if castled:
            acc["ooo"].append(ooo)
        acc["early_q"].append(early_q); lines.add(tuple(first))
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items() if k not in ("variety", "sicilian", "cpl")}
    out["variety"] = len(lines) / max(len(games), 1)
    out["sicilian"] = n_sic / n_black_e4 if n_black_e4 else float("nan")
    cp = [x["mean_cp_loss"] for x in summary["games"] if x.get("mean_cp_loss") is not None]
    out["cpl"] = float(np.mean(cp)) if cp else float("nan")
    return out


def main(root):
    root = Path(root); prompts = json.load(open(root / "prompts.json")); rows = []
    for tag, prompt in prompts.items():
        d = root / tag / "shard0"
        if not (d / "summary.json").exists():
            continue
        s = json.load(open(d / "summary.json"))
        e, ci, n = elo(s["wins"], s["draws"], s["losses"])
        rows.append(dict(tag=tag, grp=tag.split("_")[0], prompt=prompt or "(no prompt)", elo=e, ci=ci, n=n,
                         **behaviour(d / "games.pgn", s)))
    base = next((r for r in rows if r["tag"] == "none"), None)

    # Elo plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white"); X = {"weak": 0, "none": 1, "style": 2, "strong": 3}
    for grp in X:
        pts = sorted([r for r in rows if r["grp"] == grp], key=lambda r: r["elo"])
        for i, r in enumerate(pts):
            x = X[grp] + (i - (len(pts) - 1) / 2) * 0.05
            ax.errorbar(x, r["elo"], yerr=[[r["elo"] - r["ci"][0]], [r["ci"][1] - r["elo"]]], fmt="o", color=COL[grp], ms=8, capsize=3)
            left = grp in ("weak",)
            ax.annotate(r["prompt"][:48], (x, r["elo"]), xytext=(-9 if left else 9, 0), textcoords="offset points",
                        va="center", ha="right" if left else "left", fontsize=7, color="#1f1f1e")
    if base:
        ax.axhline(base["elo"], color=COL["none"], lw=1, ls="--")
    ax.set_xticks(list(X.values())); ax.set_xticklabels(["weak prompts", "no prompt", "style prompts", "strong prompts"])
    ax.set_xlim(-1.2, 4.2); ax.set_ylabel("Elo vs Stockfish 17.1 (UCI_Elo 2000), 95% CI")
    ax.grid(axis="y", color="#e6e5df"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{root.name}: playing strength by prompt", loc="left", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(root / "elo.png", dpi=150); plt.close(fig)

    # behaviour heatmap: change vs no prompt, in units of the spread across prompts
    if base:
        keys = [k for k, _, _ in METRICS]; R = [r for r in rows if r["tag"] != "none"]
        M = np.array([[r[k] - base[k] for k in keys] for r in R], dtype=float)
        sd = np.nanstd(np.array([[r[k] for k in keys] for r in rows], dtype=float), axis=0); sd[sd == 0] = 1
        Z = M / sd
        fig, ax = plt.subplots(figsize=(14, 0.38 * len(R) + 2.5), facecolor="white")
        im = ax.imshow(np.nan_to_num(Z), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l, _ in METRICS], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(R))); ax.set_yticklabels([f"[{r['grp']}] {r['prompt'][:60]}" for r in R], fontsize=7)
        for i, r in enumerate(R):
            for j, (k, _, f) in enumerate(METRICS):
                if not math.isnan(r[k]):
                    ax.text(j, i, f.format(r[k]), ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="change vs no prompt (in SDs across prompts)", shrink=0.6)
        ax.set_title("Behaviour of the model's side by prompt (cell text = value; colour = change vs no prompt)",
                     loc="left", fontsize=11, weight="bold")
        fig.tight_layout(); fig.savefig(root / "behaviour.png", dpi=150); plt.close(fig)

    with open(root / "report.md", "w") as f:
        f.write(f"# Post-SFT eval: {root.name}\n\nvs Stockfish 17.1 UCI_Elo 2000, 0.05 s/move, greedy legal moves. "
                "Behaviour is measured on the model's own moves.\n\n![elo](elo.png)\n\n![behaviour](behaviour.png)\n\n")
        f.write("| group | prompt | games | Elo (95% CI) | " + " | ".join(l for _, l, _ in METRICS) + " |\n")
        f.write("|---|---|---:|---|" + "---:|" * len(METRICS) + "\n")
        for r in sorted(rows, key=lambda r: (list(COL).index(r["grp"]) if r["grp"] in COL else 9, -r["elo"])):
            vals = " | ".join("" if math.isnan(r[k]) else fm.format(r[k]) for k, _, fm in METRICS)
            f.write(f"| {r['grp']} | {r['prompt']} | {r['n']} | {r['elo']} ({r['ci'][0]}–{r['ci'][1]}) | {vals} |\n")
    print(open(root / "report.md").read())


if __name__ == "__main__":
    main(sys.argv[1])
