"""Re-draw the figures in assets/ from the two loss-history CSVs.

    python scripts/make_plots.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = os.path.join(ROOT, "assets")
ours = np.loadtxt(os.path.join(A, "loss_history.csv"), delimiter=",", skiprows=1)
base = np.loadtxt(os.path.join(A, "baseline_adamw_history.csv"), delimiter=",", skiprows=1)

C_BASE, C_OURS = "#2563eb", "#e5484d"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def best(h):
    i = int(np.argmin(h[:, 2]))
    return h[i, 0], h[i, 2]


# ---------- figure 1: train | val, baseline vs ours ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=140)
for ax, col, title in zip(axes, (1, 2), ("Training loss", "Validation loss")):
    for h, c, name in ((base, C_BASE, "AdamW (baseline)"), (ours, C_OURS, "Ours")):
        m = h[:, 0] >= 250                                # step 0 is identical (4.29) and would squash the y-axis
        ax.plot(h[m, 0], h[m, col], color=c, lw=2.8, marker="o", ms=5, label=name, zorder=3)
        ax.annotate(f"{h[-1, col]:.3f}", (h[-1, 0], h[-1, col]), xytext=(8, 0), textcoords="offset points",
                    color=c, fontweight="bold", va="center")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("training step"); ax.set_ylabel("loss"); ax.set_xlim(0, 5400); ax.grid(alpha=0.25)

axes[0].legend(frameon=False, fontsize=12, loc="upper right")
ax = axes[1]
(bs, bv), (os_, ov) = best(base), best(ours)
ax.scatter(bs, bv, marker="*", s=420, color=C_BASE, edgecolor="k", zorder=5)
ax.scatter(os_, ov, marker="*", s=420, color=C_OURS, edgecolor="k", zorder=5)
mm = base[:, 0] >= 250
ax.fill_between(base[mm, 0], ours[mm, 2], base[mm, 2], where=base[mm, 2] > ours[mm, 2], color="#999", alpha=0.18, zorder=1)
ax.set_ylim(1.44, 1.78)
ax.text(4100, 1.462, "shaded = ours overfits less", color="#555", fontsize=10.5, ha="center")
ax.text(0.40, 0.97, f"\u2605 best val loss\nOurs     {ov:.4f} (step {int(os_)})\nAdamW  {bv:.4f} (step {int(bs)})",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc"))
ax.legend(frameon=False, fontsize=12, loc="upper right")
fig.suptitle("nanoGPT on Tiny Shakespeare: AdamW vs ours", fontsize=19, fontweight="bold", y=1.02)
fig.text(0.5, -0.02, "10.75M-parameter character-level GPT, same seed and schedule. Both peak near step 2000, then overfit; "
         "the best-checkpoint gap is only 0.005 (single seed).", ha="center", fontsize=11, color="#555")
plt.tight_layout(); plt.savefig(os.path.join(A, "compare_train_val.png"), bbox_inches="tight", facecolor="white"); plt.close()

# ---------- figure 2: our own run ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
ax.plot(ours[:, 0], ours[:, 1], color="#10b981", lw=2.6, label="train")
ax.plot(ours[:, 0], ours[:, 2], color=C_OURS, lw=2.6, label="validation")
ax.axvline(os_, color="k", ls=":", lw=1.2)
ax.scatter(os_, ov, marker="*", s=300, color=C_OURS, edgecolor="k", zorder=5)
ax.annotate(f"best checkpoint\nval {ov:.4f} @ step {int(os_)}", (os_, ov), xytext=(30, 55), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=10.5)
ax.set_ylim(0.6, 2.4); ax.set_xlabel("training step"); ax.set_ylabel("loss"); ax.grid(alpha=0.25); ax.legend(frameon=False)
ax.set_title("Training curve", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(A, "loss_curve.png"), facecolor="white"); plt.close()
print("wrote assets/compare_train_val.png and assets/loss_curve.png")
