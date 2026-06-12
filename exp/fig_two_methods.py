# -*- coding: utf-8 -*-
"""Schematic contrasting the two proposed selectors: exact CFA pruning vs the
amortized causal predictor. Static diagram."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent.parent / "article3" / "images"
GOLD = "#2c6fb0"; GREEN = "#2ca02c"; INK = "#222222"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8.5, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, zorder=1, linestyle=ls))


def main():
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    # shared input
    box(ax, 1.5, 19, 16, 12, "Question $q$\n+ retrieved\nfragments\n$\\{c_1,\\dots,c_m\\}$",
        fc="#eef3fb", fs=8.5, bold=True)

    # ---- TOP lane: CFA pruning (exact) ----
    ax.text(20, 47.5, "CFA pruning  —  exact (measures causality)", fontsize=9.5,
            fontweight="bold", color=GOLD)
    box(ax, 22, 35, 18, 9, "Leave-one-out\nablations\n($m{+}1$ gen. of $G$)", ec=GOLD, fs=8)
    box(ax, 44, 35, 15, 9, "Causal scores\n$d_i$  per fragment", ec=GOLD, fs=8)
    box(ax, 63, 35, 15, 9, "Keep\nload-bearing", fc="#e9f1f9", ec=GOLD, fs=8.5, bold=True, tc=GOLD)
    box(ax, 82, 35.5, 16, 8, "Probe 11 · 159 tok\nAcc$_s$ 0.68", fc="#e9f1f9", ec=GOLD, fs=8, tc=GOLD)
    arrow(ax, 17.5, 27, 22, 39, color=GOLD)
    arrow(ax, 40, 39.5, 44, 39.5, color=GOLD)
    arrow(ax, 59, 39.5, 63, 39.5, color=GOLD)
    arrow(ax, 78, 39.5, 82, 39.5, color=GOLD, lw=1.2)

    # ---- BOTTOM lane: Amortized (cheap) ----
    ax.text(20, 16.5, "Amortized causal  —  cheap (predicts causality)", fontsize=9.5,
            fontweight="bold", color=GREEN)
    box(ax, 22, 4, 18, 9, "Cheap features\n(relevance, cross-enc,\nredundancy, length)", ec=GREEN, fs=7.6)
    box(ax, 44, 4, 15, 9, "Learned predictor\n(grad-boosted tree)", ec=GREEN, fs=8)
    box(ax, 63, 4, 15, 9, "Predicted\nload-bearing", fc="#eef7ee", ec=GREEN, fs=8.5, bold=True, tc=GREEN)
    box(ax, 82, 4.5, 16, 8, "Probe 0 · 90 tok\nAcc$_s$ 0.53", fc="#f4fbf4", ec=GREEN, fs=8, tc=GREEN)
    arrow(ax, 17.5, 23, 22, 8.5, color=GREEN)
    arrow(ax, 40, 8.5, 44, 8.5, color=GREEN)
    arrow(ax, 59, 8.5, 63, 8.5, color=GREEN)
    arrow(ax, 78, 8.5, 82, 8.5, color=GREEN, lw=1.2)

    # amortization link: CFA causal labels -> train predictor (dashed)
    arrow(ax, 51.5, 35, 51.5, 13, color="#666666", lw=1.4, ls=(0, (4, 3)))
    ax.text(53, 24, "train once on\nCFA labels\n(amortize)", fontsize=7.6, color="#555555",
            ha="left", va="center", style="italic")

    # bottom takeaway
    ax.text(50, 0.5, "Same goal — keep causally load-bearing context — exactly (CFA, costs generations) "
            "or near-free (amortized, approximate).", ha="center", fontsize=8.2, style="italic", color=INK)

    fig.tight_layout()
    fig.savefig(FIG / "two_methods.pdf"); fig.savefig(FIG / "two_methods.png", dpi=170)
    print("saved two_methods to", FIG)


if __name__ == "__main__":
    main()
