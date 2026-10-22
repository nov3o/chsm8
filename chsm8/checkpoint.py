"""Read the released checkpoints (and the ones our training scripts write) into a CHSM8 model.

Released files come in three layouts:

    chsm8-pt/chsm8_latest.pt            {"model_state_dict", "cfg"}: every layer has the frozen pretraining
                                         cross-attention scaffold, which only ever saw a zero context
    chsm8-player-*/model.pt             {"model_state_dict", "cfg"}: scaffold already folded into `xbias`
    chsm8-sft/model.pt, chsm8/model.pt  {"chess", "cond", "frozen_dummy"}: decoder, text projection + gates,
                                         and the pinned no-text term of every layer

Our own scripts save {"config", "model"}: a Config and a CHSM8 state dict.
"""
import torch

from chsm8.model import CHSM8, Config


def scaffold_constant(sd, i, bf16=True):
    """What layer i's pretraining cross-attention added for a zero context: out_proj(v_proj(0)) = W_o b_v + b_o.
    Pretraining computed it under bf16 autocast; SFT re-computed it in fp32 when it pinned it."""
    w_o, b_o, b_v = (sd[f"layers.{i}.cross_attn.{k}"].float() for k in ("out_proj.weight", "out_proj.bias", "v_proj.bias"))
    if bf16:
        b = torch.bfloat16
        return (w_o.to(b).float() @ b_v.to(b).float() + b_o.to(b).float()).to(b).float()
    return w_o @ b_v + b_o


def convert(ck, text_layers=None, bf16_constant=True):
    """Any checkpoint dict above -> (Config, CHSM8 state dict). text_layers overrides the layout's own value, e.g.
    6 to start SFT from chsm8-pt: the top layers then keep the scaffold's cross-attention weights as their start."""
    if "config" in ck:
        cfg = Config(**ck["config"])
        if text_layers is not None and text_layers != cfg.text_layers:
            assert not cfg.text_layers, "can only add text layers to a games-only checkpoint"
            cfg.text_layers = text_layers
        return cfg, ck["model"]
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck["chess"]
    sd = {k.replace("._orig_mod", "").removeprefix("module."): v for k, v in sd.items()}
    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    d = sd["norm.weight"].shape[0]
    has_text = "cond" in ck
    if text_layers is None:
        text_layers = len(ck["cond"]["gate"]) if has_text else 0
    cfg = Config(n_layers=n_layers, d_model=d, n_heads=d // sd["layers.0.self_attn.q_norm.weight"].shape[0],
                 d_ff=sd["layers.0.ff.gate_proj.weight"].shape[0], text_layers=text_layers)
    out = {}
    for k, v in sd.items():
        if k.startswith("head."):
            k = k.replace("_head.", ".")
        if ".cross_attn." in k or ".n2." in k or k.endswith(".xbias"):
            i = int(k.split(".")[1])
            if i < n_layers - text_layers or k.endswith(".xbias"):
                continue  # layers without text keep only their constant (below)
        out[k] = v
    for i in range(n_layers):
        if has_text:
            out[f"layers.{i}.no_text"] = ck["frozen_dummy"][i].reshape(-1).float()
        elif f"layers.{i}.xbias" in sd:
            out[f"layers.{i}.no_text"] = sd[f"layers.{i}.xbias"].float()
        else:
            out[f"layers.{i}.no_text"] = scaffold_constant(sd, i, bf16_constant)
    if has_text:
        c = ck["cond"]
        out["text_proj.0.weight"], out["text_proj.0.bias"] = c["proj.0.weight"], c["proj.0.bias"]
        out["text_proj.1.weight"], out["gate"] = c["proj.1.weight"], c["gate"]
    return cfg, out


def build(ck, device="cpu", **kw):
    cfg, sd = convert(ck, **kw)
    model = CHSM8(cfg)
    if cfg.text_layers and "gate" not in sd:  # new text path on a games-only checkpoint: fresh weights where missing
        new = model.state_dict()
        sd = {**{k: v for k, v in new.items() if k.startswith(("text_proj.", "gate")) or ".cross_attn." in k
                 or ".n2." in k}, **sd}
    model.load_state_dict(sd, strict=True)
    return model.to(device)


def load(repo, filename="model.pt", device="cpu", **kw):
    """Download one file from a Hugging Face model repo (or read a local path) and build the model."""
    import os
    path = repo if os.path.exists(repo) else __import__("huggingface_hub").hf_hub_download(repo, filename)
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return build(ck, device, **kw).eval()


def save(model, path, **extra):
    from dataclasses import asdict
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)
