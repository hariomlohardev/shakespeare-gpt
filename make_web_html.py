"""Pack the trained checkpoint into ONE self-contained HTML file that runs in the browser.

    python make_web_html.py                       # uses model/best_ckpt.pt -> shakespeare_gpt_offline.html
    python make_web_html.py CKPT OUT.html         # custom paths
    python make_web_html.py --bin-only            # only write model.bin (e.g. for docs/ on GitHub Pages)
    python make_web_html.py --act relu            # if your model uses ReLU in the MLP (default: gelu, as in nanoGPT)

Only the model weights (float16), the character vocabulary and the architecture sizes are exported.
Nothing about the training procedure / optimizer is read or written.
Needs: torch, numpy.  Put shakespeare_template.html next to this script.
"""
import argparse
import base64
import json
import os
import struct

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(HERE, "model", "best_ckpt.pt"))
    ap.add_argument("out", nargs="?", default=os.path.join(HERE, "shakespeare_gpt_offline.html"))
    ap.add_argument("--bin-only", action="store_true")
    ap.add_argument("--act", choices=["gelu", "relu"], default="gelu")
    ap.add_argument("--n-head", type=int, default=None)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    clean = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", "module."):
            if k.startswith(pre):
                k = k[len(pre):]
        if torch.is_tensor(v) and v.dtype.is_floating_point:      # skips attention-mask buffers etc.
            clean[k] = v.detach().cpu().float()
    sd = clean

    need = ["transformer.wte.weight", "transformer.wpe.weight", "transformer.ln_f.weight"]
    missing = [k for k in need if k not in sd]
    if missing:
        raise SystemExit(f"Unexpected parameter names (missing {missing}).\nFound: {list(sd)[:25]}\n"
                         "This exporter expects nanoGPT-style names; send me this list and I'll adapt it.")

    # ---- architecture from shapes (+ n_head from config.json) ----
    vocab, C = sd["transformer.wte.weight"].shape
    block = sd["transformer.wpe.weight"].shape[0]
    n_layer = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.h."))
    cfg_path = os.path.join(HERE, "model", "config.json")
    n_head = a.n_head or (json.load(open(cfg_path)).get("n_head") if os.path.exists(cfg_path) else None) or 6
    assert C % n_head == 0, f"n_embd {C} not divisible by n_head {n_head}; pass --n-head"

    # ---- characters ----
    chars = ck.get("chars") if isinstance(ck, dict) else None
    if chars is None:
        vp = os.path.join(HERE, "model", "vocab.json")
        v = json.load(open(vp))
        chars = v["chars"] if "chars" in v else [c for c, _ in sorted(v["stoi"].items(), key=lambda kv: kv[1])]
    chars = "".join(chars)
    assert len(chars) == vocab, f"vocab size mismatch: {len(chars)} chars vs {vocab} embeddings"

    # tied output head: don't store it twice
    if "lm_head.weight" in sd and torch.equal(sd["lm_head.weight"], sd["transformer.wte.weight"]):
        del sd["lm_head.weight"]

    tensors, blobs, off = [], [], 0
    for name, t in sd.items():
        b = t.numpy().astype("<f2").tobytes()
        tensors.append({"name": name, "shape": list(t.shape), "offset": off})
        blobs.append(b); off += len(b)
    n_params = sum(int(np.prod(t["shape"])) for t in tensors)

    header = {"config": {"n_layer": n_layer, "n_head": int(n_head), "n_embd": int(C), "block_size": int(block),
                         "vocab_size": int(vocab), "act": a.act, "params_M": n_params / 1e6},
              "chars": chars, "tensors": tensors}
    hb = json.dumps(header, ensure_ascii=False).encode("utf-8")
    hb += b" " * ((-(4 + len(hb))) % 4)                           # align the data section
    blob = struct.pack("<I", len(hb)) + hb + b"".join(blobs)

    bin_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), "model.bin")
    if a.bin_only:
        open(bin_path, "wb").write(blob)
        print(f"wrote {bin_path} ({len(blob) / 1e6:.1f} MB) - put it next to your web page (e.g. docs/model.bin)")
        return

    tpl = open(os.path.join(HERE, "shakespeare_template.html"), encoding="utf-8").read()
    assert "__MODEL_B64__" in tpl
    html = tpl.replace("__MODEL_B64__", base64.b64encode(blob).decode("ascii"))
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"{n_layer} layers, {n_head} heads, {C}-dim, context {block}, vocab {vocab}, {n_params / 1e6:.2f}M params")
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB) - just double-click it.")


if __name__ == "__main__":
    main()
