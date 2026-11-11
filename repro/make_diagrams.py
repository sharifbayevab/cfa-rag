"""Generate the two vector overview figures used by the manuscript."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images"

BLUE = "#1769aa"
BLUE_FILL = "#e8f2fb"
GREEN = "#238636"
GREEN_FILL = "#eaf6ec"
ORANGE = "#c96b18"
ORANGE_FILL = "#fff2e5"
GREY = "#5f6368"
GREY_FILL = "#f1f3f4"
DARK = "#202124"


def box(ax, xy, wh, text, edge=GREY, face="white", fontsize=11, weight="normal", lw=1.8):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=DARK, weight=weight, linespacing=1.25)
    return patch


def arrow(ax, start, end, color=GREY, lw=2.0, style="-|>", connection="arc3"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                 mutation_scale=14, linewidth=lw,
                                 color=color, connectionstyle=connection))


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def method_overview():
    fig, ax = plt.subplots(figsize=(16, 6.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    titles = [
        (0.03, "1  Retrieved candidate pool"),
        (0.28, "2  Black-box deletion test"),
        (0.58, "3  Operational attribution"),
        (0.81, "4  Uses"),
    ]
    for x, title in titles:
        ax.text(x, 0.96, title, fontsize=14, weight="bold", color=DARK)

    box(ax, (0.035, 0.73), (0.19, 0.12), "Question $q$", BLUE, BLUE_FILL, 12, "bold")
    passages = [
        (0.035, 0.58, "$c_1$  high relevance", ORANGE, ORANGE_FILL),
        (0.035, 0.43, "$c_2$  answer-sensitive", BLUE, BLUE_FILL),
        (0.035, 0.28, "$c_3$  retrieved passage", GREY, GREY_FILL),
        (0.035, 0.13, "$c_4$  retrieved passage", GREY, GREY_FILL),
    ]
    for x, y, text, edge, face in passages:
        box(ax, (x, y), (0.19, 0.10), text, edge, face, 10.5,
            "bold" if "sensitive" in text else "normal")
    ax.text(0.13, 0.055, "Relevance and answer dependence\nneed not select the same passage",
            ha="center", va="center", fontsize=10.5, color=ORANGE, weight="bold")

    box(ax, (0.29, 0.69), (0.21, 0.15),
        "Frozen generator\n$\\hat y_D = G(q,D)$", GREY, GREY_FILL, 12, "bold")
    box(ax, (0.285, 0.43), (0.22, 0.13),
        "Remove $c_1,c_3,c_4$ separately\nnormalized answer unchanged",
        GREY, "white", 10.5)
    box(ax, (0.285, 0.21), (0.22, 0.13),
        "Remove $c_2$\nnormalized answer changes",
        BLUE, BLUE_FILL, 11, "bold")
    arrow(ax, (0.225, 0.77), (0.29, 0.77), DARK)
    arrow(ax, (0.395, 0.69), (0.395, 0.56), GREY)
    arrow(ax, (0.395, 0.43), (0.395, 0.34), BLUE)
    ax.text(0.395, 0.61, "leave one out and regenerate", ha="center",
            fontsize=9.5, color=GREY, style="italic")

    box(ax, (0.56, 0.52), (0.21, 0.31), "", BLUE, BLUE_FILL, 11.5, "bold")
    ax.text(0.665, 0.795, "Deletion-sensitivity profile", ha="center",
            fontsize=11.5, weight="bold", color=DARK)
    ax.text(0.665, 0.755, "$z_i=\\mathbf{1}[N(\\hat y_D)\\ne N(\\hat y_{-i})]$",
            ha="center", fontsize=10.5, color=DARK)
    xs = [0.595, 0.64, 0.685, 0.73]
    heights = [0.035, 0.145, 0.028, 0.04]
    colors = ["#9aa0a6", BLUE, "#9aa0a6", "#9aa0a6"]
    ax.plot([0.58, 0.75], [0.58, 0.58], color=GREY, lw=1.2)
    ax.bar(xs, heights, width=0.026, bottom=0.58, color=colors, edgecolor="white", zorder=3)
    for i, (x, h) in enumerate(zip(xs, heights), 1):
        ax.text(x, 0.555, f"$c_{i}$", ha="center", va="top", fontsize=9.5)
        ax.text(x, 0.59 + h, "1" if i == 2 else "0", ha="center", va="bottom",
                fontsize=9, color=BLUE if i == 2 else GREY, weight="bold")
    box(ax, (0.575, 0.27), (0.18, 0.17),
        "Deletion-sensitive set\n$\\mathcal{S}=\\{c_2\\}$",
        GREEN, GREEN_FILL, 12, "bold")
    arrow(ax, (0.505, 0.49), (0.56, 0.68), BLUE)
    arrow(ax, (0.665, 0.52), (0.665, 0.44), GREEN)
    ax.text(0.665, 0.15, "A change signal, not a claim\nabout desirability or reasoning",
            ha="center", fontsize=10, color=GREY, style="italic")

    box(ax, (0.81, 0.62), (0.16, 0.18),
        "Context-dependence\ndiagnostics\nPar, NoC, overlap",
        GREY, GREY_FILL, 11, "bold")
    box(ax, (0.81, 0.28), (0.16, 0.18),
        "Context pruning\nkeep $\\mathcal{S}$ and\ngenerate final answer",
        GREEN, GREEN_FILL, 11, "bold")
    arrow(ax, (0.755, 0.70), (0.81, 0.70), GREY)
    arrow(ax, (0.755, 0.36), (0.81, 0.37), GREEN)
    ax.text(0.89, 0.15, "Exact CFA: $m+1$ reader calls\nTransferred CFA: probes of other readers",
            ha="center", fontsize=10.5, color=BLUE, weight="bold")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    save(fig, "cfa_method_overview")


def system_architecture():
    fig, ax = plt.subplots(figsize=(16, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.96, "Offline measurement and learning", fontsize=14,
            weight="bold", color=BLUE)
    ax.text(0.61, 0.96, "Fresh-reader evaluation", fontsize=14,
            weight="bold", color=GREEN)
    ax.plot([0.575, 0.575], [0.05, 0.94], color="#c8cdd2", lw=1.5, ls="--")

    box(ax, (0.035, 0.67), (0.13, 0.15),
        "Question $q$\n+ passage pool $D$", GREY, GREY_FILL, 11.5, "bold")
    box(ax, (0.215, 0.67), (0.15, 0.15),
        "Exact CFA\nfull + leave-one-out\n$m+1$ reader calls",
        BLUE, BLUE_FILL, 11, "bold")
    box(ax, (0.415, 0.67), (0.13, 0.15),
        "Deletion labels\n$z_i$ for each passage",
        BLUE, BLUE_FILL, 11.5, "bold")
    arrow(ax, (0.165, 0.745), (0.215, 0.745), BLUE)
    arrow(ax, (0.365, 0.745), (0.415, 0.745), BLUE)

    box(ax, (0.215, 0.39), (0.15, 0.15),
        "Nine generation-free\nfeatures\nrelevance, rank,\nlength, redundancy",
        GREEN, GREEN_FILL, 10.5, "bold")
    box(ax, (0.415, 0.39), (0.13, 0.15),
        "Question-grouped\nOOF training\n(no question leakage)",
        GREEN, GREEN_FILL, 10.5, "bold")
    arrow(ax, (0.10, 0.67), (0.26, 0.54), GREEN, connection="arc3,rad=-0.13")
    arrow(ax, (0.365, 0.465), (0.415, 0.465), GREEN)
    arrow(ax, (0.48, 0.67), (0.48, 0.54), GREY, style="-|>")
    ax.text(0.493, 0.595, "training target", fontsize=9, color=GREY, style="italic")

    box(ax, (0.215, 0.12), (0.15, 0.13),
        "Diagnostics and\nexact pruning",
        BLUE, "white", 11, "bold")
    arrow(ax, (0.48, 0.67), (0.34, 0.25), BLUE, connection="arc3,rad=0.12")
    ax.text(0.07, 0.10,
            "Primary grid: 450 questions,\n5 generators, 3 datasets",
            fontsize=10.5, color=GREY)

    box(ax, (0.61, 0.68), (0.12, 0.14),
        "New question\n+ passage pool", GREY, GREY_FILL, 11, "bold")
    box(ax, (0.77, 0.68), (0.10, 0.14),
        "Frozen OOF\nselector", GREEN, GREEN_FILL, 11, "bold")
    box(ax, (0.90, 0.68), (0.075, 0.14),
        "Rank /\nselect $k$", GREEN, GREEN_FILL, 11, "bold")
    arrow(ax, (0.73, 0.75), (0.77, 0.75), GREEN)
    arrow(ax, (0.87, 0.75), (0.90, 0.75), GREEN)
    ax.plot([0.545, 0.59, 0.59, 0.82], [0.465, 0.465, 0.88, 0.88],
            color=GREEN, lw=1.8, ls="--")
    arrow(ax, (0.82, 0.88), (0.82, 0.82), GREEN, lw=1.8)
    ax.text(0.66, 0.895, "freeze after offline training", fontsize=9.5,
            color=GREEN, style="italic")

    box(ax, (0.625, 0.31), (0.35, 0.29), "", GREEN, "#f7fbf7", 11, "bold")
    ax.text(0.80, 0.565, "Fixed four-passage cover-EM (%)", ha="center",
            fontsize=11.5, weight="bold", color=DARK)
    methods = ["Dense", "CE+L", "OOF", "Cross", "CC"]
    values = [35.8, 37.3, 40.9, 41.3, 44.7]
    bar_x = [0.67, 0.735, 0.80, 0.865, 0.93]
    bar_h = [0.085, 0.105, 0.155, 0.161, 0.205]
    bar_colors = ["#9aa0a6", "#7f8c8d", GREEN, BLUE, ORANGE]
    ax.plot([0.65, 0.955], [0.365, 0.365], color=GREY, lw=1.1)
    ax.bar(bar_x, bar_h, width=0.035, bottom=0.365, color=bar_colors,
           edgecolor="white", zorder=3)
    for x, h, value, method, color in zip(bar_x, bar_h, values, methods, bar_colors):
        ax.text(x, 0.372 + h, f"{value:.1f}", ha="center", va="bottom",
                fontsize=9, weight="bold", color=color)
        ax.text(x, 0.345, method, ha="center", va="top", fontsize=8.5,
                weight="bold" if method == "OOF" else "normal")
    arrow(ax, (0.938, 0.68), (0.88, 0.60), GREEN, connection="arc3,rad=-0.12")

    box(ax, (0.65, 0.19), (0.30, 0.075),
        "OOF: 78.1% gold recall (best)  |  0 reader probes",
        GREEN, GREEN_FILL, 10.5, "bold")
    box(ax, (0.65, 0.075), (0.30, 0.075),
        "ContextCite: 33 probes  |  OOF cover contrast CI includes zero",
        ORANGE, ORANGE_FILL, 10, "bold")
    ax.plot([0.67, 0.60, 0.60], [0.68, 0.63, 0.112], color=ORANGE, lw=1.8)
    arrow(ax, (0.60, 0.112), (0.65, 0.112), ORANGE, lw=1.8)
    ax.text(0.61, 0.025,
            "All selectors use the same frozen Qwen3-4B reader and four-passage budget.",
            fontsize=10.5, color=GREY, style="italic")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    save(fig, "cfa_system_architecture")


if __name__ == "__main__":
    method_overview()
    system_architecture()
    print("Wrote cfa_method_overview and cfa_system_architecture as PDF and PNG")
