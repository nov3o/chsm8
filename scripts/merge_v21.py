#!/usr/bin/env python3
"""data/sft30m_v21/index.parquet = data/sft30m/index.parquet with every Apertus record's labels replaced by the clean
distilled relabel (labels/distilled_relabel_apertus), labeler = 'gemma-3-1b-distill-clean', label_correct from check.jsonl.
rows.i16 / legal.bin are symlinks (same games, same row offsets)."""
import json, glob
import pyarrow as pa, pyarrow.parquet as pq
I = "labels/distilled_relabel_apertus"
B = ["terse_1", "terse_2", "short", "medium", "long"]
new = {}
for f in sorted(glob.glob(f"{I}/labels_*.jsonl")):
    for l in open(f):
        d = json.loads(l); new[d["key"]] = [d["labels"][b] for b in B]
corr = {}
for l in open(f"{I}/check.jsonl"):
    d = json.loads(l); corr[d["key"]] = d["correct"]
t = pq.read_table("data/sft30m/index.parquet")
keys = [f"{m}:{r}:{s}" for m, r, s in zip(t["month"].to_pylist(), t["row_idx"].to_pylist(), t["side"].to_pylist())]
lab, labeler, lc = t["labels"].to_pylist(), t["labeler"].to_pylist(), t["label_correct"].to_pylist()
n_ap = n_rep = 0
for i, k in enumerate(keys):
    if labeler[i] and labeler[i].startswith("apertus"):
        n_ap += 1
        if k in new:
            lab[i], labeler[i], lc[i] = new[k], "gemma-3-1b-distill-clean", corr[k]; n_rep += 1
t = t.set_column(t.schema.get_field_index("labels"), "labels", pa.array(lab, type=t.schema.field("labels").type))
t = t.set_column(t.schema.get_field_index("labeler"), "labeler", pa.array(labeler, type=t.schema.field("labeler").type))
t = t.set_column(t.schema.get_field_index("label_correct"), "label_correct", pa.array(lc, type=t.schema.field("label_correct").type))
pq.write_table(t, "data/sft30m_v21/index.parquet")
print(json.dumps({"records": len(keys), "apertus_records": n_ap, "replaced": n_rep, "relabels_available": len(new)}))
