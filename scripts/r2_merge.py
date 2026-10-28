"""Pool two independent 1,000-game evals (e1k_<m> + e1k_<m>_r2) -> 2,000-game key metrics + z-scores vs the 20 short DPO runs."""
import csv, json, sys, numpy as np
sys.path.insert(0, "src"); from summarize_sft_eval import elo, behaviour
TAGS = ("none", "style_00", "style_07", "style_09", "style_11")
MODELS = [("SFT v2.1", "sft_v21"), ("10-min dpo_label_kl1", "dpo_label_kl1"), ("random band 100% (published)", "dpofull_p100"),
          ("long band 25%", "long25_p25"), ("long band 100%", "long25_p100")]
def one(d):
    r = {}
    for t in TAGS:
        p = f"results/sft_eval/{d}/{t}/shard0"; s = json.load(open(p + "/summary.json"))
        r[t] = dict(wdl=(s["wins"], s["draws"], s["losses"]), **behaviour(p + "/games.pgn", s))
    return r
def keys(rs):
    W = {t: [sum(r[t]["wdl"][i] for r in rs) for i in range(3)] for t in TAGS}
    m = lambda t, k: float(np.mean([r[t][k] for r in rs]))
    en, ea = elo(*W["none"])[0], elo(*W["style_00"])[0]
    return [100 * m("style_07", "q_off40"), m("style_11", "pawn"), 100 * m("style_09", "ooo"), en, ea - en]
rows = [r for r in csv.DictReader(open("autoresearch_dpo/eval1k_results.csv")) if r["model"] != "sft_v21"]
C = ["queens_traded_pct", "pawn_moves_per_move", "queenside_castling_pct", "elo_no_prompt", "elo_attack_minus_no_prompt"]
X = np.array([[float(r[c]) for c in C] for r in rows]); mu, sd = X.mean(0), X.std(0, ddof=1)
sg = np.array([1, 1, 1, 1, -1]); w = np.array([1, 1, .2, 1, 1]); Z = sg * (X - mu) / sd
a = np.linalg.solve(.5 * np.corrcoef(Z.T) + .5 * np.eye(5), w); a *= w.sum() / a.sum()
out = []
for name, m in MODELS:
    r1, r2 = one(f"e1k_{m}"), one(f"e1k_{m}_r2")
    k1, k2, kp = keys([r1]), keys([r2]), keys([r1, r2]); z = sg * (np.array(kp) - mu) / sd
    out.append(dict(model=name, run1=k1, run2=k2, pooled=kp, z=z.tolist(), sum_z=float(z @ w), sum_z_corr=float(z @ a)))
json.dump(out, open("results/r2_pooled.json", "w"), indent=1)
f = lambda k: f"{k[0]:.1f}% / {k[1]:.3f} / {k[2]:.1f}% / {k[3]:.0f} / {k[4]:+.0f}"
for o in out: print(f"{o['model']:30s} r1 {f(o['run1'])} | r2 {f(o['run2'])} | pooled {f(o['pooled'])} | z {np.round(o['z'],1)} sum {o['sum_z']:+.2f} corr {o['sum_z_corr']:+.2f}")
