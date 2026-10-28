#!/usr/bin/env python3
"""Print the labeler ablation table from results/labeler/results.jsonl (best val first)."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/labeler/results.jsonl"
rows = [json.loads(l) for l in open(path)]
rows = [r for r in rows if not r["trial"].startswith("smoke")]
rows.sort(key=lambda r: r["val_nats_per_byte"])
cols = ["trial", "val", "terse", "short", "med", "long", "ex", "ex/s", "parse", "lenOK", "leak", "gen rec/s"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    b = r["val_per_band"]
    print("| " + " | ".join([
        r["trial"], f"{r['val_nats_per_byte']:.4f}", f"{(b['terse_1'] + b['terse_2']) / 2:.3f}",
        f"{b['short']:.3f}", f"{b['medium']:.3f}", f"{b['long']:.3f}", f"{r['examples']}",
        f"{r['examples'] / (r['minutes'] * 60):.1f}", f"{r['gen_parse_ok']:.2f}", f"{r['gen_band_len_ok']:.2f}",
        f"{r['gen_leak_rate']:.2f}", f"{r['gen_records_per_s']:.2f}"]) + " |")
