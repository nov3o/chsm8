#!/usr/bin/env python3
"""tables/bt.tex: Bradley-Terry strength of every model from all logged matches (run from repo root, CPU only).

Matches: each model against Stockfish 17.1 at UCI_Elo 2000 (results/opp_sf1k, and our three models' 1,000-game runs),
the 1,000-game head-to-head matches of our models (results/versus1k) and the earlier 200-game chsm8-pt matches
(results/versus). Draws count as half a win each way; Stockfish is fixed at 2000.
"""
import glob, json, os

import numpy as np

OUT = os.environ.get("REPORT_OUT", "techreport")
R = "results"
NAME = {"stockfish": "Stockfish (anchor)", "chsm8-pt": "\\pt", "chsm8-player-base": "\\textsc{player-base}",
        "chsm8-player-carlsen": "\\textsc{player-carlsen}", "allie2": "Allie 2.0", "c1-4b": "C1-4B",
        "chessgpt2-uci-12x12x768": "chess-gpt2-uci", "chesslfm": "ChessLFM", "chesslfm-search": "ChessLFM with search",
        "deepmind-searchless-9m": "Searchless 9M", "deepmind-searchless-136m": "Searchless 136M",
        "deepmind-searchless-270m": "Searchless 270M", "karvonen-chessgpt-lichess16": "Chess-GPT (Karvonen)",
        "kempner-chessformer-50m-1500": "ChessFormer (Transcendence)", "maia2-blitz": "Maia-2 blitz",
        "maia3-79m": "Maia-3 79M", "matilda-d12": "Matilda", "otter-15m": "Otter 15M", "queen": "QUEEN (PAWN-8)",
        "waterhorse-chessgpt-base-v1": "ChessGPT-base 2.8B"}
OURS = {"chsm8-pt", "chsm8-player-base", "chsm8-player-carlsen"}
# Icon per group, as in Table 19: supervised move prediction, strong recent models without search, search.
GROUP = {**{k: r"\grpsup" for k in ("chsm8-pt", "chsm8-player-base", "chsm8-player-carlsen", "maia2-blitz", "chesslfm",
                                    "kempner-chessformer-50m-1500", "chessgpt2-uci-12x12x768", "karvonen-chessgpt-lichess16",
                                    "waterhorse-chessgpt-base-v1")},
         **{k: r"\grpadv" for k in ("otter-15m", "allie2", "maia3-79m", "deepmind-searchless-9m", "deepmind-searchless-136m",
                                    "deepmind-searchless-270m", "c1-4b", "queen")},
         **{k: r"\grpsearch" for k in ("chesslfm-search", "matilda-d12", "stockfish")}}


def matches():
    M = []  # (a, b, wins of a, draws, losses of a)
    for f in glob.glob(f"{R}/opp_sf1k/*/elo.json"):
        o = f.split("/")[-2]
        if "smoke" not in o:
            d = json.load(open(f)); M.append((o, "stockfish", d["wins"], d["draws"], d["losses"]))
    for tag, f in [("chsm8-pt", "playing_benchmark/chsm8_1k/latest"), ("chsm8-player-base", "elo1k/g100_final"),
                   ("chsm8-player-carlsen", "elo1k/final_1503014")]:
        d = json.load(open(f"{R}/{f}/elo.json")); M.append((tag, "stockfish", d["wins"], d["draws"], d["losses"]))
    for f in glob.glob(f"{R}/versus1k/*/summary.json") + glob.glob(f"{R}/versus1k/*/*/summary.json"):
        us, opp = f.split("versus1k/")[1].split("/")[0].split("__")
        if "smoke" not in opp:
            d = json.load(open(f)); M.append((us, opp, d["chsm8_wins"], d["draws"], d["chsm8_losses"]))
    for f in glob.glob(f"{R}/versus/*/summary.json"):
        d = json.load(open(f)); M.append(("chsm8-pt", d["opponent"], d["chsm8_wins"], d["draws"], d["chsm8_losses"]))
    return M


def fit(M):
    """Newton's method on the Bradley-Terry log-likelihood (Elo scale), Stockfish fixed; standard errors from the
    observed information."""
    P = sorted({x for m in M for x in m[:2]}); ix = {p: i for i, p in enumerate(P)}; n = len(P)
    c = np.log(10) / 400; r = np.zeros(n); fix = ix["stockfish"]; keep = [k for k in range(n) if k != fix]
    for _ in range(100):
        g = np.zeros(n); H = np.zeros((n, n))
        for a, b, w, d, l in M:
            i, j = ix[a], ix[b]; N = w + d + l; s = w + d / 2
            p = 1 / (1 + np.exp(-c * (r[i] - r[j])))
            g[i] += c * (s - N * p); g[j] -= c * (s - N * p)
            h = c * c * N * p * (1 - p); H[i, i] += h; H[j, j] += h; H[i, j] -= h; H[j, i] -= h
        step = np.linalg.solve(H[np.ix_(keep, keep)], g[keep]); r[keep] += step
        if np.abs(step).max() < 1e-9:
            break
    se = np.zeros(n); se[keep] = np.sqrt(np.diag(np.linalg.inv(H[np.ix_(keep, keep)])))
    return P, r + 2000 - r[fix], se


M = matches()
P, r, se = fit(M)
games = {p: sum(w + d + l for a, b, w, d, l in M if p in (a, b)) for p in P}
elo = {"stockfish": 2000.0}
for a, b, w, d, l in M:
    if b == "stockfish":
        s = (w + d / 2) / (w + d + l); elo[a] = 2000 + 400 * np.log10(s / (1 - s))
bt_rank = {P[i]: k + 1 for k, i in enumerate(np.argsort(-r))}
elo_rank = {p: k + 1 for k, p in enumerate(sorted(P, key=lambda p: -elo[p]))}

lines = [r"\begin{tabular}{@{}rlr@{\,}lrrl@{}}", r"\toprule",
         r"\# & model & \multicolumn{2}{c}{BT (95\% CI)} & games & Elo & Elo rank \\", r"\midrule"]
for p in sorted(P, key=lambda p: bt_rank[p]):
    i = P.index(p); d = elo_rank[p] - bt_rank[p]
    move = r"\rankup{%d}" % d if d > 0 else (r"\rankdown{%d}" % -d if d < 0 else r"\rankeq")
    nm = r"\textbf{%s}" % NAME[p] if p in OURS else NAME[p]
    nm = GROUP[p] + r"\," + nm
    ci = "(fixed)" if p == "stockfish" else r"$\pm$\,%.0f" % (1.96 * se[i])
    el = "---" if p == "stockfish" else "%.0f" % elo[p]
    lines.append(f"{bt_rank[p]} & {nm} & {r[i]:.0f} & {ci} & {games[p]:,} & {el} & \\makebox[0.9em][r]{{{elo_rank[p]}}}\\,{move} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tables/bt.tex", "w").write("\n".join(lines) + "\n")
json.dump({"matches": len(M), "games": sum(w + d + l for *_, w, d, l in M), "players": len(P)},
          open(f"{OUT}/tables/bt_meta.json", "w"))
print(f"wrote tables/bt.tex: {len(M)} matches, {sum(w + d + l for *_, w, d, l in M):,} games, {len(P)} players")
