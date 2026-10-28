#!/usr/bin/env python3
"""Claim-level distillation metric: extract style claims from label sets, then score
grounding (claims vs the game's measured card) and fidelity (claims vs the teacher's).

  extract  (GPU, vLLM): a fixed reader model maps each 5-label set to 12 style axes.
  score    (CPU):       grounding precision, teacher agreement (kappa), recall of the
                        teacher's grounded claims, and ratio to a teacher-vs-teacher ceiling.

usage:
  claims.py extract --sources teacher,w3_t5gemma_2b2b_gemma,... [--n 384]
  claims.py score   --sources ...  [--ceiling teacher_gemma_resample]
A source is `teacher` (original labels from test.jsonl) or a samples_<name>.jsonl file.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

R = Path("results/labeler")
TEST = Path("data/labeler_gemma/test.jsonl")
READER = "Qwen/Qwen3-32B"
BANDS = ("terse_1", "terse_2", "short", "medium", "long")

AXES = {
    "material": ["sacrifice", "hoard"],             # gives material away / grabs and keeps material
    "trades": ["seek", "avoid"],                    # trades pieces a lot / avoids exchanges
    "king_attack": ["yes", "no"],                   # goes after the enemy king / does not
    "development": ["fast", "slow"],
    "castling": ["short", "long", "late", "never"],
    "queens": ["trade_early", "keep"],
    "simplify_endgame": ["yes", "no"],              # aims for simplification / an endgame grind
    "pawn_storm": ["yes", "no"],                    # throws flank pawns at the king
    "center": ["dominate", "cede"],
    "risk": ["reckless", "cautious"],
    "posture": ["attacking", "defensive"],
    "tempo": ["fast", "slow"],                      # plays quickly / slowly (clock), not development
}

PROMPT = """You read chess style instructions and record which style claims they make.

Instructions (5 of them, all describing the same player):
{labels}

For each axis below, output the value the instructions claim, or "none" if they do not say
anything about that axis. Only record what is stated or clearly implied; do not guess.
If the instructions contradict each other on an axis, use the claim made most strongly.

Axes and allowed values:
{axes}

Answer with a single JSON object mapping every axis to one allowed value or "none". No prose."""


def sources_rows(sources, n):
    test = {json.loads(l)["key"]: json.loads(l) for l in open(TEST)}
    out = {}
    for s in sources:
        if s == "teacher":
            out[s] = {k: v["labels"] for k, v in test.items()}
        else:
            out[s] = {json.loads(l)["key"]: json.loads(l)["pred"] for l in open(R / f"samples_{s}.jsonl")}
    keys = sorted(set.intersection(*[set(v) for v in out.values()]))[:n] if n else None
    return test, out, keys


def extract(a):
    from vllm import LLM, SamplingParams
    test, rows, keys = sources_rows(a.sources.split(","), a.n)
    llm = LLM(model=READER, max_model_len=4096, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    axes_txt = "\n".join(f"- {k}: {' | '.join(v)} | none" for k, v in AXES.items())
    sp = SamplingParams(temperature=0.0, max_tokens=300)
    for s, by_key in rows.items():
        ks = keys or sorted(by_key)
        prompts = []
        for k in ks:
            labels = "\n".join(f"- {by_key[k].get(b, '')}" for b in BANDS)
            prompts.append(tok.apply_chat_template([{"role": "user", "content": PROMPT.format(labels=labels, axes=axes_txt)}],
                                                   tokenize=False, add_generation_prompt=True, enable_thinking=False))
        outs = llm.generate(prompts, sp)
        bad = 0
        with open(R / f"claims_{s}.jsonl", "w") as f:
            for k, o in zip(ks, outs):
                txt = o.outputs[0].text
                m = re.search(r"\{.*\}", txt, re.S)
                try:
                    d = json.loads(m.group(0)) if m else {}
                except json.JSONDecodeError:
                    d = {}
                claims = {ax: (d.get(ax) if d.get(ax) in vals else "none") for ax, vals in AXES.items()}
                bad += not d
                f.write(json.dumps({"key": k, "claims": claims}) + "\n")
        print(f"{s}: {len(ks)} label sets, {bad} unparsed", flush=True)


# ---------------------------------------------------------------- game truth from the card

def card_facts(user: str) -> dict:
    g = lambda rx, cast=str: (lambda m: cast(m.group(1)) if m else None)(re.search(rx, user))  # noqa: E731
    f = {
        "sac": g(r"sacrifices \(captures.*?\): (\d+)", int),
        "invested": g(r"total invested ([\d.]+) pawns", float),
        "deficit": g(r"worst material deficit endured: ([\d.]+)", float),
        "cap_label": g(r"captures: \d+ \([\d.]+/move, ([\w ]+)\)"),
        "even_trades": g(r"even trades initiated: (\d+)", int),
        "press": g(r"pressure on \w+'s king ring: [\d.]+ attackers/ply — (\w+)"),
        "pushes_at_king": g(r"pawn pushes at the enemy king: (\d+)", int),
        "dev": g(r"development: ([\w ]+) \("),
        "king": g(r"- king: ([^\n]+)"),
        "queens": g(r"- queens: ([^\n]+)"),
        "endgame": g(r"endgame share: (\d+)%", int),
        "center": g(r"central presence: .*? — (\w+)"),
    }
    k = f["king"] or ""
    f["castled"] = None if not k else ("short" if k.startswith("kingside") else "long" if k.startswith("queenside")
                                       else "never" if k.startswith("none") else None)
    m = re.search(r"on ply (\d+)", k)
    f["castle_ply"] = int(m.group(1)) if m else None
    q = f["queens"] or ""
    m = re.search(r"traded on ply (\d+)", q)
    f["queen_trade_ply"] = int(m.group(1)) if m else (None if not q else 10**6)
    return f


def contradicted(ax, val, f) -> bool | None:
    """True = the card contradicts the claim, False = consistent, None = card can't decide."""
    if ax == "material":
        if f["sac"] is None:
            return None
        if val == "sacrifice":
            return f["sac"] == 0 and (f["deficit"] or 0) < 2
        return f["sac"] >= 1 and (f["invested"] or 0) >= 2  # "hoard" vs a real sacrifice
    if ax == "trades" and f["cap_label"]:
        busy = f["cap_label"] in ("high contact", "very bloody") or (f["even_trades"] or 0) >= 3
        quiet = f["cap_label"] == "low contact" and (f["even_trades"] or 0) == 0
        return quiet if val == "seek" else busy
    if ax == "king_attack" and f["press"]:
        if val == "yes":
            return f["press"] == "none" and (f["pushes_at_king"] or 0) == 0
        return f["press"] in ("real", "heavy", "strong")
    if ax == "pawn_storm" and f["pushes_at_king"] is not None:
        return f["pushes_at_king"] == 0 if val == "yes" else f["pushes_at_king"] >= 3
    if ax == "development" and f["dev"]:
        return f["dev"] in ("slow", "very slow") if val == "fast" else f["dev"] in ("very fast", "brisk")
    if ax == "castling" and f["castled"]:
        if val == "never":
            return f["castled"] in ("short", "long")
        if val in ("short", "long"):
            return f["castled"] != val
        if val == "late" and f["castle_ply"]:
            return f["castle_ply"] <= 16
    if ax == "queens" and f["queen_trade_ply"] is not None:
        return f["queen_trade_ply"] > 40 if val == "trade_early" else f["queen_trade_ply"] <= 30
    if ax == "simplify_endgame" and f["endgame"] is not None:
        if val == "yes":
            return f["endgame"] == 0 and f["queen_trade_ply"] == 10**6
        return f["endgame"] >= 30
    if ax == "center" and f["center"]:
        return f["center"] in ("cedes", "abandons", "ignores") if val == "dominate" else f["center"] in ("dominates", "controls")
    return None


def kappa(pairs):
    """Cohen's kappa over categorical pairs."""
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[x] * cb[x] for x in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(a):
    test = {json.loads(l)["key"]: json.loads(l) for l in open(TEST)}
    load = lambda s: {json.loads(l)["key"]: json.loads(l)["claims"] for l in open(R / f"claims_{s}.jsonl")}  # noqa: E731
    srcs = a.sources.split(",")
    C = {s: load(s) for s in srcs + ([a.ceiling] if a.ceiling else [])}
    keys = sorted(set.intersection(*[set(v) for v in C.values()]))
    T = C["teacher"]
    print(f"{len(keys)} games; claims per label set and grounding use the card; agreement is vs the original teacher\n")
    rows = []
    for s in ([a.ceiling] if a.ceiling else []) + srcs:
        c = C[s]
        n_claims = checked = bad = 0
        agree_pairs, other_pairs = [], []
        t_true = t_true_hit = 0
        for i, k in enumerate(keys):
            f = card_facts(test[k]["user"])
            other = keys[(i + 1) % len(keys)]
            for ax in AXES:
                v = c[k][ax]
                if v != "none":
                    n_claims += 1
                    x = contradicted(ax, v, f)
                    if x is not None:
                        checked += 1
                        bad += x
                if s != "teacher":
                    agree_pairs.append((v, T[k][ax]))
                    other_pairs.append((v, T[other][ax]))
                    tv = T[k][ax]
                    if tv != "none" and contradicted(ax, tv, f) is False:  # teacher claim the card confirms
                        t_true += 1
                        t_true_hit += v == tv
        rows.append((s, n_claims / len(keys), 1 - bad / max(1, checked), checked / max(1, n_claims),
                     kappa(agree_pairs) if agree_pairs else None, kappa(other_pairs) if other_pairs else None,
                     t_true_hit / max(1, t_true) if t_true else None))
    ceil = next((r[4] - r[5] for r in rows if r[0] == a.ceiling), None)  # game-specific agreement of the ceiling
    print("| source | claims / set | grounding precision | checkable share | kappa vs teacher (same game) | "
          "kappa vs teacher (other game) | recall of teacher's grounded claims | distillation (game-specific) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s, cps, gp, chk, kap, kap_o, rec in rows:
        fmt = lambda x, p=False: "—" if x is None else (f"{x:.0%}" if p else f"{x:.3f}")  # noqa: E731
        dist = "—" if kap is None or not ceil else f"{(kap - kap_o) / ceil:.0%}"
        print(f"| {s} | {cps:.1f} | {gp:.1%} | {chk:.0%} | {fmt(kap)} | {fmt(kap_o)} | {fmt(rec, True)} | {dist} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "score"])
    ap.add_argument("--sources", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ceiling", default="")
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else score(a)


if __name__ == "__main__":
    main()
