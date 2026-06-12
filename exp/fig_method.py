# -*- coding: utf-8 -*-
"""Method-overview schematic for CFA (Figure 1). Static diagram, no data."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent.parent / "article3" / "images"
FIG.mkdir(parents=True, exist_ok=True)

GOLD = "#2c6fb0"   # load-bearing / causal
GRAY = "#9e9e9e"   # distractor / balast
BLUE = "#1f77b4"
INK = "#222222"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=9, bold=False, tc=INK, round=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={round}",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    # ---- panel titles ----
    titles = [("1. Retrieved context", 12), ("2. Black-box leave-one-out", 38),
              ("3. Answer shift → causal score", 66), ("4. Use", 90)]
    for t, x in titles:
        ax.text(x, 53.5, t, ha="center", fontsize=9.5, fontweight="bold", color=INK)

    # ---- Panel 1: question + fragments ----
    box(ax, 2, 45, 20, 5.5, "Question  q", fc="#eef3fb", fs=9, bold=True)
    frag_y = [37, 30, 23, 16]
    labels = ["c₁ (distractor)", "c₂  load-bearing", "c₃ (distractor)", "c₄ (distractor)"]
    fcs = [GRAY, GOLD, GRAY, GRAY]
    for y, lab, fc in zip(frag_y, labels, fcs):
        box(ax, 2, y, 20, 5.5, lab, fc=fc, fs=8, tc="white", bold=(fc == GOLD))
    ax.text(12, 11, "fragments  $\\mathcal{D}=\\{c_1,\\dots,c_m\\}$", ha="center", fontsize=8, style="italic")

    # ---- Panel 2: generator + ablations ----
    box(ax, 30, 40, 18, 10, "Frozen generator\n$G(q,\\,\\mathcal{C})$\n(black-box: API/local)",
        fc="#f4f4f4", fs=8.5, bold=True)
    box(ax, 30, 31, 18, 5, "full answer  $a_{\\mathrm{full}}$", fc="#eef3fb", fs=8.5)
    arrow(ax, 22, 33, 30, 44)            # fragments -> G
    arrow(ax, 22, 47.5, 30, 46)          # question -> G
    arrow(ax, 39, 40, 39, 36)            # G -> a_full
    # leave-one-out rows
    ax.text(39, 27.5, "remove each $c_i$, regenerate:", ha="center", fontsize=8, style="italic")
    box(ax, 28, 19.5, 22, 5, "$-c_1$:  $a_{-1}=a_{\\mathrm{full}}$   ✓ no change", fc="white", ec=GRAY, fs=8, tc=GRAY)
    box(ax, 28, 13, 22, 5, "$-c_2$:  $a_{-2}\\neq a_{\\mathrm{full}}$   ✗ flips", fc="#e9f1f9", ec=GOLD, fs=8, tc=GOLD, bold=True)
    arrow(ax, 39, 31, 39, 24.5, color=INK, lw=1.2)

    # ---- Panel 3: shift formula + causal scores ----
    box(ax, 56, 41, 26, 9,
        "shift  $d_i = w_1\\,\\mathbb{1}[\\mathrm{flip}] + w_2\\,\\Delta F_1 + w_3\\,\\Delta\\mathrm{sem}$",
        fc="#f4f4f4", fs=8.2)
    arrow(ax, 50, 21, 56, 41)            # ablations -> shift
    # causal score bars
    bx, by, bw = 58, 14, 4.5
    scores = [0.08, 0.74, 0.05, 0.10]
    cols = [GRAY, GOLD, GRAY, GRAY]
    for i, (s, c) in enumerate(zip(scores, cols)):
        ax.add_patch(plt.Rectangle((bx + i * 6, by), bw, s * 22, color=c, zorder=3))
        ax.text(bx + i * 6 + bw / 2, by - 1.8, f"c{i+1}", ha="center", fontsize=8)
    ax.text(56, by + 17, "causal score per fragment", fontsize=8, style="italic")
    ax.plot([bx - 1.5, bx - 1.5], [by, by + 17], color=INK, lw=0.8)
    arrow(ax, 69, 41, 69, 31.5, color=INK, lw=1.2)

    # ---- Panel 4: outputs ----
    box(ax, 84, 33, 15, 9, "Faithfulness\ndiagnostics\n(%NoC, parametric)", fc="#eef7ee", fs=8, bold=True)
    box(ax, 84, 20, 15, 9, "Causal pruning\nkeep $\\{c_2\\}$\n≈acc, 4.1× fewer\ntokens", fc="#e9f1f9", ec=GOLD, fs=8, bold=True, tc=GOLD)
    arrow(ax, 82, 24, 84, 37, color=INK, lw=1.2)
    arrow(ax, 82, 22, 84, 24.5, color=GOLD, lw=1.2)

    fig.tight_layout()
    fig.savefig(FIG / "method_overview.pdf"); fig.savefig(FIG / "method_overview.png", dpi=170)
    print("saved method_overview to", FIG)


if __name__ == "__main__":
    main()
