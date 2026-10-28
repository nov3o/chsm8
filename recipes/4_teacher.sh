#!/bin/bash
# Teacher labels: Gemma-4-31B writes five descriptions per game side. We ran prompt-ablation/production_label.py against a
# hosted OpenAI-compatible Gemma-4-31B endpoint (provider "swissai" in prompt-ablation/autolabel/providers.py; point
# PROVIDERS["swissai"] at your own server, e.g. `vllm serve google/gemma-4-31B-it`, and set SWISSAI_API_KEY).
# Inputs: data/2b (from 0_data.sh pre) and data/4/player_game_index.parquet (the sides to label).
# Our labels are public (chsm8-teacher-labels, chsm8-distillation); you only need this to label new games.
# One side with the exact production messages, on local vLLM:   python src/labeler/teacher_example.py --tp 4
set -euo pipefail
python prompt-ablation/production_label.py "$@"
