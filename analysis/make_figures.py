"""Figures 1 and 2 of the paper, drawn from results/run_level_results.json.

Usage: python analysis/make_figures.py results/run_level_results.json results/figures
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES = json.load(open(sys.argv[1]))
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

models = [  # name, color, marker, linestyle, fill
    ("Llama-3.1-8B", BLUE, "s", (0, (4, 2)), "none"),
    ("Gemma-2-2B", AQUA, "^", "-", AQUA),
    ("Llama-3.2-3B", BLUE, "o", "-", BLUE),
    ("Qwen2.5-3B", ORANGE, "D", "-", ORANGE),
]
# chrF++ gain = fine-tuned three-seed mean minus zero-shot run, under each zero-shot prompt (Table 8)
G = RES["gains_over_zero_shot"]
gain = {m: (G[m]["chrf_gain_alpaca"], G[m]["chrf_gain_bare"]) for m in G}

fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300)
for name, col, mk, ls, fc in models:
    a, b = gain[name]
    ax.plot([0, 1], [a, b], color=col, lw=1.6, ls=ls, zorder=2)
    ax.plot([0, 1], [a, b], ls="none", marker=mk, ms=6.5, mfc=fc if fc != "none" else "white", mec=col, mew=1.4, zorder=3)
# direct labels, nudged to avoid collisions
left = {"Llama-3.1-8B": 0.0, "Gemma-2-2B": 0.0, "Llama-3.2-3B": 0.0, "Qwen2.5-3B": 0.0}
right = {"Qwen2.5-3B": 0.0, "Gemma-2-2B": 0.2, "Llama-3.2-3B": -0.02, "Llama-3.1-8B": -0.26}
for name, col, mk, ls, fc in models:
    a, b = gain[name]
    ax.text(-0.04, a + left[name], f"{name}  +{a:.2f}", ha="right", va="center", color=TEXT, fontsize=8)
    ax.text(1.04, b + right[name], f"+{b:.2f}  {name}", ha="left", va="center", color=TEXT, fontsize=8)
ax.set_xlim(-0.95, 1.95)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Alpaca zero-shot prompt", "Bare zero-shot prompt"], color=TEXT)
ax.set_ylabel("chrF++ gain over zero-shot baseline")
ax.set_ylim(16.3, 21.9)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0)
fig.tight_layout()
fig.savefig(OUT / "fig1_gains.png", dpi=300, facecolor="white")
plt.close(fig)

# Figure 2: fine-tuned reference likelihood (BPB) against chrF++ and BLEU, mean +/- seed SD (Tables 7 and 10)
S = RES["fine_tuned_summary"]
pts = {m: (S[m]["bpb"]["mean"], S[m]["bpb"]["sd"], S[m]["chrf"]["mean"], S[m]["chrf"]["sd"], S[m]["bleu"]["mean"], S[m]["bleu"]["sd"]) for m in S}
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), dpi=300, sharex=True)
panels = [(axes[0], 2, "chrF++, fine-tuned", (27.9, 32.4), "a"), (axes[1], 4, "BLEU, fine-tuned", (4.3, 6.9), "b")]
offs = {  # per panel: name -> (dx, dy, ha)
    "a": {"Qwen2.5-3B": (0.005, 0.30, "left"), "Llama-3.2-3B": (-0.005, -0.62, "right"), "Gemma-2-2B": (-0.005, -0.50, "right"),
          "Llama-3.1-8B": (0.005, 0.40, "left")},
    "b": {"Qwen2.5-3B": (0.005, 0.17, "left"), "Llama-3.2-3B": (-0.005, -0.20, "right"), "Gemma-2-2B": (-0.005, -0.17, "right"),
          "Llama-3.1-8B": (0.005, 0.14, "left")},
}
for ax, k, ylab, ylim, tag in panels:
    for name, col, mk, ls, fc in models:
        v = pts[name]
        x, xs, y, ys = v[0], v[1], v[k], v[k + 1]
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="none", ecolor=col, elinewidth=1.2, capsize=0, zorder=2)
        ax.plot(x, y, marker=mk, ms=6.5, mfc=fc if fc != "none" else "white", mec=col, mew=1.4, ls="none", zorder=3)
        dx, dy, ha = offs[tag][name]
        ax.text(x + dx, y + dy, name, ha=ha, va="center", color=TEXT, fontsize=7.5)
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.70, 0.87)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.03, f"({tag})", transform=ax.transAxes, ha="left", va="bottom", color=TEXT, fontsize=8.5, fontweight="bold")
fig.supxlabel("Reference bits per byte, fine-tuned (lower = higher likelihood of the reference)", fontsize=8.5, color=TEXT, y=0.035)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(OUT / "fig2_bpb_generation.png", dpi=300, facecolor="white")
plt.close(fig)
print(f"wrote {OUT / 'fig1_gains.png'} and {OUT / 'fig2_bpb_generation.png'}")
