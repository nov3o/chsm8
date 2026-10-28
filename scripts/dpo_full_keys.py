"""Key play metrics of the full DPO run's checkpoints (1,000-game evals) -> results/dpo_full_keys.json (read by the dashboard).
0% = SFT v2.1 (results/sft_eval/e1k_sft_v21)."""
import json, sys
PRE = sys.argv[1] if len(sys.argv) > 1 else "dpofull"; OUT = sys.argv[2] if len(sys.argv) > 2 else "results/dpo_full_keys.json"
sys.path.insert(0, "src")
from summarize_sft_eval import elo, behaviour

TAGS = ("none", "style_00", "style_07", "style_09", "style_11")
out = []
for pct, d in [(0, "e1k_sft_v21")] + [(p, f"e1k_{PRE}_p{p}") for p in (25, 50, 75, 100)]:
    r = {}
    try:
        for t in TAGS:
            s = json.load(open(f"results/sft_eval/{d}/{t}/shard0/summary.json"))
            r[t] = dict(elo=elo(s["wins"], s["draws"], s["losses"]), **behaviour(f"results/sft_eval/{d}/{t}/shard0/games.pgn", s))
    except FileNotFoundError:
        continue
    out.append({"pct": pct, "queens": r["style_07"]["q_off40"], "pawns": r["style_11"]["pawn"], "queenside": r["style_09"]["ooo"],
                "elo_none": r["none"]["elo"][0], "elo_none_ci": r["none"]["elo"][1], "elo_attack": r["style_00"]["elo"][0],
                "elo_attack_ci": r["style_00"]["elo"][1], "gap": r["style_00"]["elo"][0] - r["none"]["elo"][0],
                "checks_attack": r["style_00"]["checks"], "games": [r[t]["elo"][2] for t in TAGS]})
json.dump(out, open(OUT, "w"), indent=1)
print(len(out), "points")
