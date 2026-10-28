#!/bin/bash
# How the published labels were cleaned (chsm8-distillation, chsm8-post-training). Our labels are public; this is the
# record of the steps, each an exact script of our run (paths relative to the repo root).
#   1. grounding check: every band against the measured card                        src/labeler/grounding.py
#      (rules in prompt-ablation/autolabel/contradictions.py; claims extraction for the audit: src/labeler/claims.py)
#   2. the clean distillation set: teacher sets with no contradicted band            src/labeler/filter_clean.py
#      export for Hugging Face                                                       src/labeler/export_distill.py
#   3. repair: regenerate contradicted bands with the distilled labeler, conditioned on the game and the other bands
#      src/labeler/repair_prep.py -> src/labeler/repair_labels.py; leftovers: fix_todo.py -> fix_loop.py; merge_final.py
#   4. relabel the Apertus-70B sets with the clean labeler: src/labeler/build_unlabeled.py --select apertus, then
#      src/labeler/label_shards.py; check: scripts/check_relabel.py; merge into the SFT index: scripts/merge_v21.py
#   5. export the 30M labels                                                        src/labeler/export_labels30m.py
# Each script's docstring gives its inputs and outputs.
set -euo pipefail
echo "see the comments in this file"
