"""Train the character-level GPT on Tiny Shakespeare.

    python src/train.py                 # full run (~15 min on a free Colab T4)
    FAST=1 python src/train.py          # 400-iteration smoke test

Writes the best checkpoint to model/best_ckpt.pt (plus config.json, vocab.json, loss_history.csv).
"""
import json
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn

from model import GPT, GPTConfig

# ============================== settings ==============================
OPTIMIZER = "adamw"          # "adamw"  -> the standard baseline, works out of the box
                             # "ours"   -> something special, see build_optimizer() below
FAST = bool(int(os.environ.get("FAST", "0")))

cfg = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2, bias=False)   # ~10.7M parameters
batch_size = 64
max_iters = 400 if FAST else 5000
eval_interval = 100 if FAST else 250
eval_iters = 50 if FAST else 100
learning_rate, min_lr, warmup_iters = 1e-3, 1e-4, 100       # linear warm-up, then cosine decay
weight_decay, grad_clip, seed = 0.1, 1.0, 1337

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
out_dir = os.path.join(ROOT, "model"); os.makedirs(out_dir, exist_ok=True)
data_file = os.path.join(ROOT, "input.txt")
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"


# ============================== optimizer ==============================
def build_optimizer(model):
    decay = [p for p in model.parameters() if p.dim() >= 2]      # weight matrices get weight decay
    no_decay = [p for p in model.parameters() if p.dim() < 2]    # biases / norms do not
    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]

    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.99), fused=use_amp)

    if OPTIMIZER == "ours":
        # ✨ ──────────────────────────────────────────────────────────────── ✨
        #
        #     Something special is brewing right here.
        #
        #     The model in this repo was trained with a brand-new optimizer
        #     of our own. We are keeping the recipe under wraps for now ...
        #     but not for long.   Surprise coming soon.   Stay tuned!  🎭
        #
        # ✨ ──────────────────────────────────────────────────────────────── ✨
        raise NotImplementedError("Our optimizer is not public yet: surprise coming soon! Use OPTIMIZER = 'adamw'.")

    raise ValueError(f"unknown optimizer: {OPTIMIZER}")


# ============================== data ==============================
if not os.path.exists(data_file):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, data_file)
text = open(data_file, encoding="utf-8").read()
chars = sorted(set(text))
cfg.vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_tr = int(0.9 * len(data))
train_data, val_data = data[:n_tr].to(device), data[n_tr:].to(device)
offs = torch.arange(cfg.block_size, device=device)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (batch_size,), device=device)
    return d[ix[:, None] + offs], d[ix[:, None] + offs + 1]


# ============================== train ==============================
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
model = GPT(cfg).to(device)
print(f"{len(text):,} characters | vocab {cfg.vocab_size} | {model.num_params() / 1e6:.2f}M parameters | optimizer: {OPTIMIZER}")
opt = build_optimizer(model)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)          # fp16 + loss scaling (T4 has no bf16)


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    r = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval(); out = {}
    for split in ("train", "val"):
        ls = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                ls[k] = model(X, Y)[1].item()
        out[split] = ls.mean().item()
    model.train()
    return out


history, best_val, t0 = [], float("inf"), time.time()
X, Y = get_batch("train")
for it in range(max_iters + 1):
    for g in opt.param_groups:
        g["lr"] = get_lr(it)
    if it % eval_interval == 0:
        L = estimate_loss(); history.append((it, L["train"], L["val"]))
        print(f"step {it:5d} | train {L['train']:.4f} | val {L['val']:.4f} | {(time.time() - t0) / 60:.1f} min")
        if L["val"] < best_val:                                  # keep only the best-validation checkpoint
            best_val = L["val"]
            torch.save(dict(model=model.state_dict(), iter=it, val_loss=best_val, chars=chars),
                       os.path.join(out_dir, "best_ckpt.pt"))
    if it == max_iters:
        break
    with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
        _, loss = model(X, Y)
    X, Y = get_batch("train")
    scaler.scale(loss).backward()
    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

# ============================== save ==============================
json.dump(dict(**cfg.to_dict(), best_val_loss=best_val, params_M=round(model.num_params() / 1e6, 2)),
          open(os.path.join(out_dir, "config.json"), "w"), indent=2)
json.dump(dict(chars=chars), open(os.path.join(out_dir, "vocab.json"), "w"))
np.savetxt(os.path.join(ROOT, "assets", "loss_history.csv"), np.array(history), delimiter=",",
           header="iter,train_loss,val_loss", comments="", fmt=["%d", "%.4f", "%.4f"])
print(f"done in {(time.time() - t0) / 60:.1f} min | best val loss {best_val:.4f}")
