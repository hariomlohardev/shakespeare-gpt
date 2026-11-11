---
language: en
tags:
  - text-generation
  - pytorch
  - gpt
  - character-level
  - shakespeare
datasets:
  - tiny_shakespeare
---

# Shakespeare GPT (10.75M)

A small decoder-only transformer trained **character by character** on Tiny Shakespeare.
It runs **entirely locally**: in your browser via GitHub Pages, or in Python on your CPU.

| | |
|---|---|
| Parameters | 10.75M (6 layers, 6 heads, 384-dim, context 256) |
| Vocabulary | 65 characters |
| Best validation loss | **1.4642** (step 2000) |
| Training | 5,000 steps, best-validation checkpoint kept |

Trained with a custom optimizer: ✨ details coming soon, surprise! ✨

## Use it

```python
# pip install torch
# files in this repo: model/best_ckpt.pt, model/config.json, src/model.py, src/inference.py
from inference import Generator
g = Generator("model/best_ckpt.pt")
print(g.generate("ROMEO:", max_new_tokens=300, temperature=0.8))
```

Or open the browser demo (`docs/index.html`, also published via GitHub Pages) —
no server, no uploads, everything runs on your device.

## Limitations

It imitates the *style* of Shakespeare, not the meaning: expect plausible-looking nonsense.
Trained on ~1 MB of text, so it overfits quickly after step ~2000.
English only, characters outside its 65-symbol vocabulary are ignored.
