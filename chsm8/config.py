"""Configs are YAML files (configs/*.yaml); any key can be overridden on the command line:

    torchrun --nproc-per-node 4 scripts/train/pretrain.py configs/pt.yaml --batch_size=16 --data=/fast/pt
"""
import ast
import sys
from types import SimpleNamespace

import yaml


def load(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    assert argv and argv[0].endswith((".yaml", ".yml")), "usage: script.py config.yaml [--key=value ...]"
    cfg = yaml.safe_load(open(argv[0]))
    for arg in argv[1:]:
        key, _, value = arg.removeprefix("--").partition("=")
        assert key in cfg, f"unknown config key {key!r} (keys: {', '.join(cfg)})"
        try:
            cfg[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            cfg[key] = value  # plain string
    return SimpleNamespace(**cfg)
