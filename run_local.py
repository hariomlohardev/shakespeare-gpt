"""Run the model locally on CPU and generate an example.

    python run_local.py                              # default example
    python run_local.py "JULIET:" 400 0.8            # prompt, characters, temperature
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from inference import Generator            # uses src/model.py + model/best_ckpt.pt

prompt = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
length = int(sys.argv[2]) if len(sys.argv) > 2 else 300
temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

torch.set_num_threads(os.cpu_count() or 1)
print("Loading model on CPU...")
gen = Generator(os.path.join(HERE, "model", "best_ckpt.pt"), device="cpu")
print(f"Loaded {gen.model.num_params() / 1e6:.2f}M parameters (best val loss {gen.best_val_loss:.4f})\n")

t0 = time.time()
text = gen.generate(prompt, max_new_tokens=length, temperature=temperature, seed=42)
print("=" * 60)
print(text)
print("=" * 60)
print(f"\n{length} characters in {time.time() - t0:.1f}s  |  temperature {temperature}  |  seed 42")
