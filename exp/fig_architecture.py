# -*- coding: utf-8 -*-
"""Clean architecture schematic of the PROPOSED method (no results/metrics).
Two ALTERNATIVE inference paths (not parallel): exact CFA (with ablations) and
the amortized predictor (no ablations, trained offline on CFA scores).
Colors encode the two PATHS only (not importance)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent.parent / "article3" / "images"
PATHA = "#2c6fb0"   # Path A (exact CFA) — blue
PATHB = "#1e8449"   # Path B (amortized) — green
NEUT = "#555555"    # neutral (input / outputs)
INK = "#2b2b2b"
TINT_A = "#e9f1f9"; TINT_B = "#eaf5ee"; TINT_NEUT = "#f1f1f1"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8.6, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arr(ax, x1, y1, x2, y2, color=INK, lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, zorder=1, linestyle=ls))


def main():
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")

    # shared input (neutral)
    box(ax, 1.5, 19, 13, 12, "Query $q$\n+ fragments\n$\\{c_1,\\dots,c_m\\}$",
        fc=TINT_NEUT, ec=NEUT, fs=8.8, bold=True)

    # ===== Path A: exact CFA (with ablations) — blue =====
    ax.text(17.5, 45, "Path A — exact CFA  (runs ablations)", color=PATHA, fontsize=9.5, fontweight="bold")
    box(ax, 17.5, 33, 15, 8.5, "Frozen $G$:\nleave-one-out\nablations", fc="white", ec=PATHA, fs=8.1, bold=True)
    box(ax, 35.5, 33, 14, 8.5, "Answer shift\n$d_i$", fc="white", ec=PATHA, fs=8.4)
    box(ax, 52.5, 33, 14, 8.5, "Causal score\n(measured)", fc=TINT_A, ec=PATHA, fs=8.4, bold=True)
    arr(ax, 14.5, 27, 17.5, 37, color=PATHA)
    arr(ax, 32.5, 37.2, 35.5, 37.2, color=PATHA)
    arr(ax, 49.5, 37.2, 52.5, 37.2, color=PATHA)

    # ===== Path B: amortized (no ablations) — green =====
    ax.text(17.5, 3.0, "Path B — amortized predictor  (no ablations)", color=PATHB, fontsize=9.5, fontweight="bold")
    box(ax, 17.5, 6, 15, 8.5, "Generation-free\nfeatures", fc="white", ec=PATHB, fs=8.1, bold=True)
    box(ax, 35.5, 6, 14, 8.5, "Learned predictor\n(GBT)", fc="white", ec=PATHB, fs=8.4)
    box(ax, 52.5, 6, 14, 8.5, "Causal score\n(predicted)", fc=TINT_B, ec=PATHB, fs=8.4, bold=True)
    arr(ax, 14.5, 23, 17.5, 10.3, color=PATHB)
    arr(ax, 32.5, 10.3, 35.5, 10.3, color=PATHB)
    arr(ax, 49.5, 10.3, 52.5, 10.3, color=PATHB)

    # offline training link (dashed)
    arr(ax, 56, 33, 44, 14.5, color="#888888", lw=1.4, ls=(0, (4, 3)))
    ax.text(46.5, 24, "offline:\ntrain once on\nCFA scores", fontsize=7.4, color="#666666",
            ha="center", va="center", style="italic")

    # "use ONE at inference"
    ax.annotate("", xy=(59.5, 33), xytext=(59.5, 14.5),
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=1.0))
    ax.text(60.6, 23.8, "use ONE\nat inference", fontsize=7.6, color="#555555", ha="left", va="center", fontweight="bold")

    # ===== two uses (neutral) =====
    box(ax, 80, 31, 18, 8.5, "Faithfulness\ndiagnostics", fc=TINT_NEUT, ec=NEUT, fs=8.6, bold=True)
    box(ax, 80, 8, 18, 8.5, "Causal context\nselection (pruning)", fc=TINT_NEUT, ec=NEUT, fs=8.6, bold=True)
    arr(ax, 66.5, 37, 80, 35.2, color=PATHA, lw=1.4)        # exact -> diagnostics (A only)
    arr(ax, 66.5, 35, 80, 14, color=PATHA, lw=1.3)          # exact -> selection
    arr(ax, 66.5, 10.3, 80, 12.5, color=PATHB, lw=1.3)      # amortized -> selection
    ax.text(89, 4.2, "selection: Path A or B;   diagnostics: Path A only", fontsize=7.6,
            color="#555555", ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(FIG / "architecture.pdf"); fig.savefig(FIG / "architecture.png", dpi=170)
    print("saved architecture to", FIG)


if __name__ == "__main__":
    main()
