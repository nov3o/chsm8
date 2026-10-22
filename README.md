# CHSM8

A chess move model that plays in the style you describe in plain text. Code for data preparation, pretraining,
SFT, DPO and the style labeler. Models and data load from Hugging Face.

```
pip install -e .
python scripts/data/prepare_pretraining.py ...      # chsm8-pre-training -> packed games
python scripts/train/pretrain.py configs/pt.yaml     # games-only pretraining
python scripts/data/prepare_posttraining.py ...     # chsm8-post-training -> described game sides
python scripts/data/prepare_players.py --packed data/players --sides data/player_sides
torchrun --nproc-per-node 4 scripts/train/sft.py configs/sft.yaml
python scripts/train/dpo.py configs/dpo.yaml
torchrun --nproc-per-node 4 scripts/label/train_labeler.py --out runs/labeler
```

`configs/nano.yaml` and `configs/nano_sft.yaml` train a small model on any machine.

Work in progress: notebooks, labeling and evaluation scripts are coming.

MIT license.
