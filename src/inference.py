"""Load the trained model and generate text.

    python src/inference.py --prompt "ROMEO:" --tokens 400 --temperature 0.8
"""
import argparse
import json
import os
import time

import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "model")


class Generator:
    """Wraps the model + character vocabulary behind one simple `generate()` call."""

    def __init__(self, ckpt_path=None, config_path=None, device=None):
        ckpt_path = ckpt_path or os.path.join(DEFAULT_DIR, "best_ckpt.pt")
        config_path = config_path or os.path.join(os.path.dirname(ckpt_path), "config.json")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.chars = ckpt["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

        fields = GPTConfig.__dataclass_fields__
        cfg = {}
        if os.path.exists(config_path):
            cfg = {k: v for k, v in json.load(open(config_path)).items() if k in fields}
        cfg["vocab_size"] = len(self.chars)
        self.config = GPTConfig(**cfg)

        self.model = GPT(self.config)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.best_val_loss = ckpt.get("val_loss")
        self.best_iter = ckpt.get("iter")

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]       # characters outside the vocabulary are skipped

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    @torch.inference_mode()
    def generate(self, prompt="\n", max_new_tokens=300, temperature=0.8, top_k=200, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.encode(prompt) or self.encode("\n")
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        return self.decode(y[0].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    g = Generator(a.ckpt)
    t0 = time.time()
    print(g.generate(a.prompt, a.tokens, a.temperature, a.top_k, a.seed))
    print(f"\n[{a.tokens} chars in {time.time() - t0:.1f}s on {g.device}]")


if __name__ == "__main__":
    main()
