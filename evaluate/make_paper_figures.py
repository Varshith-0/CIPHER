#!/usr/bin/env python3
"""
CIPHER — NeurIPS-quality figure generation.

Generates all 8 paper figures from embedded data tables.
No external CSV files required; all results are hard-coded from
the final reported numbers in the paper.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore")

# ── Output dir ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# ── NeurIPS design system ─────────────────────────────────────────────────────
# Palette: muted, print-friendly classic tones
C_BLUE   = "#2F4B7C"   # ERP / primary model
C_GREEN  = "#3B7A57"   # DDA / positive
C_ORANGE = "#B5651D"   # acoustic / confound / negative
C_PURPLE = "#6E4B7E"   # CIPHER highlight
C_GREY   = "#7A7A7A"   # chance / neutral
C_TEAL   = "#2C7F7B"   # secondary green
C_AMBER  = "#B08A1E"   # warning / ablation

PALETTE_2 = [C_BLUE, C_GREEN]
PALETTE_3 = [C_BLUE, C_GREEN, C_ORANGE]
PALETTE_MODELS = [C_GREY, C_GREY, C_GREY, C_GREY, C_BLUE, C_BLUE, C_PURPLE]

def setup() -> None:
    mpl.rcParams.update({
        # Font
        "font.family":          "serif",
        "font.serif":           ["STIX Two Text", "Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "font.size":            9.5,
        "axes.titlesize":       10.5,
        "axes.titleweight":     "bold",
        "axes.labelsize":       9.5,
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "legend.fontsize":      8.5,
        "legend.title_fontsize":9.0,
        # Lines
        "axes.linewidth":       0.8,
        "xtick.major.width":    0.7,
        "ytick.major.width":    0.7,
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "lines.linewidth":      1.5,
        # Grid — subtle y-only
        "axes.grid":            False,
        "grid.color":           "#E5E5E5",
        "grid.linewidth":       0.45,
        "grid.alpha":           1.0,
        "axes.axisbelow":       True,
        # Background — transparent for publication overlays
        "axes.facecolor":       "none",
        "figure.facecolor":     "none",
        "savefig.facecolor":    "none",
        # Spines
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.edgecolor":       "#444444",
        # Resolution
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
    })

def save(name: str) -> None:
    out_svg = (OUT / name).with_suffix(".svg")
    plt.savefig(out_svg, format="svg", transparent=True)
    plt.close("all")
    print(f"  saved {out_svg.name}")


def _classic_axis(ax: plt.Axes, y_grid: bool = True) -> None:
    if y_grid:
        ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.45)
    ax.xaxis.grid(False)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — WER comparison (primary result)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_wer() -> None:
    """Grouped bar chart of WER by feature type and word type with error bars."""
    conditions = ["ERP\nReal words", "ERP\nPseudowords", "DDA\nReal words", "DDA\nPseudowords"]
    wer_mean   = [0.671, 0.780, 0.688, 0.772]
    wer_std    = [0.080, 0.029, 0.096, 0.050]
    chance     = 1 - 1/11   # ≈ 0.909

    colors = [C_BLUE, C_BLUE, C_GREEN, C_GREEN]
    alphas = [1.0, 0.55, 1.0, 0.55]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    x = np.arange(len(conditions))
    bars = []
    for i, (m, s, c, a) in enumerate(zip(wer_mean, wer_std, colors, alphas)):
        b = ax.bar(x[i], m, width=0.58, color=c, alpha=a,
                   yerr=s, capsize=4, error_kw={"linewidth": 1.2, "capthick": 1.2},
                   zorder=3)
        bars.append(b)

    # Chance line
    ax.axhline(chance, color=C_GREY, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(3.65, chance + 0.008, "Chance\n(0.909)", fontsize=7.5,
            color=C_GREY, va="bottom", ha="right")

    # Value labels on bars
    for xi, (m, s) in enumerate(zip(wer_mean, wer_std)):
        ax.text(xi, m + s + 0.015, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8.5)
    ax.set_ylabel("Word Error Rate (WER)  ↓", fontsize=9.5)
    ax.set_ylim(0.55, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    _classic_axis(ax)

    # Legend for feature type
    leg_patches = [
        mpatches.Patch(color=C_BLUE,  label="ERP  (0.5–40 Hz)"),
        mpatches.Patch(color=C_GREEN, label="DDA  (broadband 2000 Hz)"),
        mpatches.Patch(color="#888",  alpha=0.45, label="Pseudowords"),
    ]
    ax.legend(handles=leg_patches, fontsize=8.0, loc="upper left",
              frameon=True, edgecolor="#cccccc", framealpha=0.92)

    fig.tight_layout()
    save("fig1_wer_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Baseline comparison heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_baselines() -> None:
    """Side-by-side heatmaps ERP | DDA for Phoneme / Manner / Place."""
    import matplotlib.colors as mcolors

    models  = ["Chance", "LR", "LDA", "ShallowConvNet", "EEGNet", "EEG-Conformer", "CIPHER"]
    tasks   = ["Phoneme\nIdentity", "Manner", "Place"]

    # [model × task]  rows: Chance LR LDA Shallow EEGNet EEG-Conf CIPHER
    erp = np.array([
        [0.091, 0.500, 0.500],
        [0.089, 0.590, 0.507],
        [0.132, 0.769, 0.519],
        [0.170, 0.857, 0.573],
        [0.174, 0.857, 0.571],
        [0.167, 0.857, 0.571],
        [0.155, 0.852, 0.573],
    ])
    dda = np.array([
        [0.091, 0.500, 0.500],
        [0.139, 0.711, 0.513],
        [0.153, 0.800, 0.518],
        [0.160, 0.851, 0.567],
        [0.176, 0.855, 0.573],
        [0.171, 0.857, 0.571],
        [0.166, 0.860, 0.574],
    ])

    cmap = mpl.colormaps["Blues"]
    norm_ph  = mcolors.Normalize(vmin=0.07, vmax=0.20)   # phoneme col
    norm_bin = mcolors.Normalize(vmin=0.48, vmax=0.90)   # binary cols

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    titles = ["ERP  (0.5–40 Hz)", "DDA  (broadband 2000 Hz)"]

    for ax, data, title in zip(axes, [erp, dda], titles):
        nR, nC = data.shape
        # Cell colours per column
        for r in range(nR):
            for c in range(nC):
                v = data[r, c]
                norm = norm_ph if c == 0 else norm_bin
                fc = cmap(norm(v))
                rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1,
                                     facecolor=fc, edgecolor="white", linewidth=0.6)
                ax.add_patch(rect)
                # Text contrast
                brightness = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
                tc = "white" if brightness < 0.5 else "#111111"
                fw = "bold" if models[r] == "CIPHER" else "normal"
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=8.5, color=tc, fontweight=fw)

        # CIPHER highlight border
        cipher_row = models.index("CIPHER")
        for c in range(nC):
            rect = plt.Rectangle([c - 0.5, cipher_row - 0.5], 1, 1,
                                  fill=False, edgecolor=C_PURPLE, linewidth=2.0)
            ax.add_patch(rect)

        ax.set_xlim(-0.5, nC - 0.5)
        ax.set_ylim(-0.5, nR - 0.5)
        ax.set_xticks(range(nC))
        ax.set_xticklabels(tasks, fontsize=9.0)
        ax.set_yticks(range(nR))
        ax.set_yticklabels(models if ax == axes[0] else [], fontsize=9.0)
        ax.grid(False)
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.invert_yaxis()

    # Shared colorbar on a dedicated far-right axis (never overlays cells).
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm_bin)
    sm.set_array([])
    fig.subplots_adjust(right=0.88, wspace=0.10)
    cax = fig.add_axes([0.905, 0.16, 0.013, 0.68])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Accuracy", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)
    save("fig2_baseline_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Controls: EEG-only vs acoustic-only
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_controls() -> None:
    """Grouped bar chart with acoustic-only vs EEG controls side by side."""
    tasks     = ["Phoneme Identity (11-class)", "Manner (2-class)", "Place (2-class)"]
    erp_acc   = [0.104, 0.633, 0.518]
    dda_acc   = [0.129, 0.708, 0.525]
    acou_acc  = [1.000, 1.000, 1.000]
    chance    = [0.091, 0.500, 0.500]
    erp_ci    = [(0.088, 0.122), (0.567, 0.695), (None, None)]
    dda_ci    = [(0.114, 0.144), (0.636, 0.766), (None, None)]

    x   = np.arange(len(tasks))
    w   = 0.22
    fig, ax = plt.subplots(figsize=(8.5, 4.4))

    def _yerr(vals, cis):
        lo = [v - ci[0] if ci[0] else 0 for v, ci in zip(vals, cis)]
        hi = [ci[1] - v if ci[1] else 0 for v, ci in zip(vals, cis)]
        return [lo, hi]

    ax.bar(x - w,     erp_acc,  w, color=C_BLUE,   label="EEG  (ERP)",   zorder=3,
           yerr=_yerr(erp_acc, erp_ci),
           capsize=3.5, error_kw={"linewidth": 1.1})
    ax.bar(x,         dda_acc,  w, color=C_GREEN,  label="EEG  (DDA)",   zorder=3,
           yerr=_yerr(dda_acc, dda_ci),
           capsize=3.5, error_kw={"linewidth": 1.1})
    ax.bar(x + w,     acou_acc, w, color=C_ORANGE, label="Acoustic-only", zorder=3)

    # Chance markers
    for xi, ch in zip(x, chance):
        ax.plot([xi - 1.5*w, xi + 1.5*w], [ch, ch], color=C_GREY,
                linewidth=1.2, linestyle=":", zorder=4)

    # Value annotations  — only acoustic (to call out the 1.000)
    for xi in x:
        ax.text(xi + w, 1.000 + 0.012, "1.000", ha="center", va="bottom",
                fontsize=7.5, color=C_ORANGE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9.0)
    ax.set_ylabel("LOSO Accuracy", fontsize=9.5)
    ax.set_ylim(0.0, 1.14)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    _classic_axis(ax)

    # Chance legend element
    chance_line = mpl.lines.Line2D([], [], color=C_GREY, linestyle=":",
                                   linewidth=1.2, label="Chance level")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [chance_line], labels + ["Chance level"],
              fontsize=8.5, loc="upper right", frameon=True,
              edgecolor="#cccccc", framealpha=0.92)

    fig.tight_layout()
    save("fig3_control_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Time-window masking + block permutation
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_time_perm() -> None:
    tasks    = ["Phoneme\nIdentity", "Manner", "Place"]
    deltas   = [-0.0003, -0.0251, -0.0051]

    perm_tasks   = ["Phoneme\nIdentity", "Manner", "Place"]
    erp_p        = [0.706, 0.706, 0.706]
    dda_p        = [0.529, 0.529, 0.882]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    # ── Panel A: time-window masking delta ──────────────────────────────────
    ax = axes[0]
    colors_tw = [C_AMBER if d < -0.01 else C_GREY for d in deltas]
    bars = ax.bar(tasks, deltas, color=colors_tw, width=0.52, zorder=3,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.9, zorder=2)
    for bar, d in zip(bars, deltas):
        ypos = d - 0.003 if d < 0 else d + 0.001
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{d:+.4f}", ha="center", va="top" if d < 0 else "bottom",
                fontsize=8.0, fontweight="bold")
    ax.set_ylabel("Δ Accuracy  (masked − base)", fontsize=9.5)
    ax.set_ylim(-0.045, 0.008)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    _classic_axis(ax)
    tw_handles = [
        mpatches.Patch(color=C_AMBER, label="Notable drop (< -0.01)"),
        mpatches.Patch(color=C_GREY, label="Near-zero change"),
    ]
    ax.legend(handles=tw_handles, fontsize=8.0, loc="lower left",
              frameon=True, edgecolor="#cccccc", framealpha=0.92)

    # ── Panel B: permutation p-values ────────────────────────────────────────
    ax = axes[1]
    x  = np.arange(len(perm_tasks))
    w  = 0.3
    ax.bar(x - w/2, erp_p, w, color=C_BLUE,  label="ERP", zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, dda_p, w, color=C_GREEN, label="DDA", zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color=C_ORANGE, linewidth=1.4, linestyle="--", zorder=4,
               label="α = 0.05")
    ax.axhline(1.00, color="#cccccc", linewidth=0.6, zorder=1)
    for xi, (ep, dp) in enumerate(zip(erp_p, dda_p)):
        ax.text(xi - w/2, ep + 0.02, f"{ep:.3f}", ha="center",
                fontsize=7.5, color=C_BLUE)
        ax.text(xi + w/2, dp + 0.02, f"{dp:.3f}", ha="center",
                fontsize=7.5, color=C_GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(perm_tasks, fontsize=9.0)
    ax.set_ylabel("Empirical p-value", fontsize=9.5)
    ax.set_ylim(0.0, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(fontsize=8.5, loc="upper right", frameon=True,
              edgecolor="#cccccc", framealpha=0.92)
    _classic_axis(ax)

    fig.tight_layout()
    save("fig4_time_window_permutation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Ablation heatmap (multi-seed, 8-8 split)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_ablation_heatmap() -> None:
    import matplotlib.colors as mcolors

    variants  = ["Full\nCIPHER", "No SE", "No stoch.\ndepth",
                 "No attn\npooling", "No multi-\nscale"]
    cells     = ["ERP\nPhoneme", "ERP\nManner", "ERP\nPlace",
                 "DDA\nPhoneme", "DDA\nManner", "DDA\nPlace"]

    # [variant × cell]
    data = np.array([
        # ERP-PID  ERP-MAN  ERP-PLC  DDA-PID  DDA-MAN  DDA-PLC
        [0.132,   0.774,   0.551,   0.125,   0.704,   0.571],  # Full
        [0.132,   0.705,   0.549,   0.122,   0.704,   0.529],  # No SE
        [0.124,   0.808,   0.517,   0.125,   0.716,   0.537],  # No stoch depth
        [0.150,   0.822,   0.548,   0.137,   0.716,   0.572],  # No attn pool
        [0.124,   0.806,   0.563,   0.137,   0.852,   0.572],  # No multi-scale
    ])

    # Diverging colormap centered on full-CIPHER row
    full_row = data[0]
    delta = data - full_row[np.newaxis, :]
    vabs = max(abs(delta[1:].min()), abs(delta[1:].max()), 0.04)
    cmap = mpl.colormaps["RdYlGn"]
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    nR, nC = data.shape
    for r in range(nR):
        for c in range(nC):
            fc = cmap(norm(delta[r, c])) if r > 0 else "#DDEEFF"
            rect = plt.Rectangle([c - 0.5, r - 0.5], 1, 1,
                                  facecolor=fc, edgecolor="white", linewidth=0.8)
            ax.add_patch(rect)
            fc_rgb = mpl.colors.to_rgb(fc)
            brightness = 0.299*fc_rgb[0] + 0.587*fc_rgb[1] + 0.114*fc_rgb[2]
            tc = "white" if brightness < 0.48 else "#111111"
            fw = "bold" if r == 0 else "normal"
            # Show acc for row 0, delta for others
            label = f"{data[r,c]:.3f}" if r == 0 else f"{delta[r,c]:+.3f}"
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=8.5, color=tc, fontweight=fw)

    # Full-CIPHER row border
    for c in range(nC):
        rect = plt.Rectangle([c - 0.5, -0.5], 1, 1,
                              fill=False, edgecolor=C_PURPLE, linewidth=2.0)
        ax.add_patch(rect)

    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(-0.5, nR - 0.5)
    ax.set_xticks(range(nC))
    ax.set_xticklabels(cells, fontsize=9.0)
    ax.set_yticks(range(nR))
    ax.set_yticklabels(variants, fontsize=9.0)
    ax.tick_params(left=False, bottom=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()

    # Colorbar (delta scale only)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02, aspect=18)
    cbar.set_label("Δ vs Full CIPHER  (other rows)", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)

    # Column group labels
    ax.text(1.0/6.0, -0.04, "ERP", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=C_BLUE,
            transform=ax.transAxes, clip_on=False)
    ax.text(4.0/6.0, -0.04, "DDA", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=C_GREEN,
            transform=ax.transAxes, clip_on=False)

    fig.tight_layout()
    save("fig5_ablation_multiseed_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Ablation deltas lollipop chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_ablation_deltas() -> None:
    variants = ["No SE", "No stoch.\ndepth", "No attn\npooling", "No multi-\nscale"]
    tasks    = ["PID", "MAN", "PLC"]

    erp_delta = {
        "No SE":           [-0.000, -0.069, -0.002],
        "No stoch.\ndepth":[-0.008,  0.034, -0.034],
        "No attn\npooling":[+0.018,  0.048, -0.003],
        "No multi-\nscale":[-0.008,  0.032,  0.012],
    }
    dda_delta = {
        "No SE":           [-0.003, 0.000, -0.042],
        "No stoch.\ndepth":[ 0.000, 0.012, -0.034],
        "No attn\npooling":[+0.012, 0.012, +0.001],
        "No multi-\nscale":[+0.012, 0.148,  0.001],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    y = np.arange(len(variants))

    for ax, delta_dict, ft, fc in zip(axes, [erp_delta, dda_delta], ["ERP", "DDA"], [C_BLUE, C_GREEN]):
        markers = ["o", "s", "^"]
        task_colors = ["#555555", "#888888", "#BBBBBB"]
        for ti, (task, mc, tc) in enumerate(zip(tasks, markers, task_colors)):
            dx = [delta_dict[v][ti] for v in variants]
            ax.scatter(dx, y + (ti - 1)*0.18, marker=mc, color=fc,
                       alpha=0.65 + ti*0.12, s=55, zorder=4,
                       label=f"Task: {task}")
            for xi, yi_off in zip(dx, y + (ti - 1)*0.18):
                ax.plot([0, xi], [yi_off, yi_off], color=fc,
                        alpha=0.35, linewidth=1.0, zorder=3)

        ax.axvline(0, color="black", linewidth=0.9, zorder=5)
        ax.axvline(-0.01, color=C_ORANGE, linewidth=0.8, linestyle=":",
                   alpha=0.6, zorder=2)
        ax.axvline(+0.01, color=C_ORANGE, linewidth=0.8, linestyle=":",
                   alpha=0.6, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(variants if ax == axes[0] else [], fontsize=9.0)
        ax.set_xlabel("Δ Accuracy vs Full CIPHER", fontsize=9.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
        # Reference band
        ax.axvspan(-0.01, 0.01, alpha=0.08, color=C_GREY, zorder=1)
        _classic_axis(ax)

        if ax == axes[0]:
            ax.legend(fontsize=8.0, loc="lower right", frameon=True,
                      edgecolor="#cccccc", framealpha=0.92)

    fig.tight_layout()
    save("fig6_ablation_deltas.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — TMS ANOVA p-values
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_tms() -> None:
    labels  = ["ERP\nBilabial", "ERP\nAlveolar", "DDA\nBilabial", "DDA\nAlveolar"]
    pvals   = [0.530, 0.114, 0.068, 0.626]
    fstats  = [0.791, 3.458, 7.496, 0.516]
    colors  = [C_BLUE, C_BLUE, C_GREEN, C_GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # Panel A: p-values
    ax = axes[0]
    x = np.arange(len(labels))
    bar_colors = [C_ORANGE if p < 0.10 else c for p, c in zip(pvals, colors)]
    bars = ax.bar(x, pvals, color=bar_colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axhline(0.05, color="#CC0000", linewidth=1.4, linestyle="--",
               label="α = 0.05", zorder=4)
    ax.axhline(0.10, color=C_ORANGE, linewidth=1.1, linestyle=":",
               label="α = 0.10", zorder=4)
    for bar, p in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.015,
                f"p = {p:.3f}", ha="center", va="bottom",
                fontsize=8.0, fontweight="bold" if p < 0.10 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.0)
    ax.set_ylabel("ANOVA p-value", fontsize=9.5)
    ax.set_ylim(0, 0.85)
    handles = [
        mpatches.Patch(color=C_BLUE, label="ERP"),
        mpatches.Patch(color=C_GREEN, label="DDA"),
        mpatches.Patch(color=C_ORANGE, label="p < 0.10 highlight"),
        mpl.lines.Line2D([], [], color="#CC0000", linestyle="--", linewidth=1.4, label="α = 0.05"),
        mpl.lines.Line2D([], [], color=C_ORANGE, linestyle=":", linewidth=1.1, label="α = 0.10"),
    ]
    ax.legend(handles=handles, fontsize=8.2, loc="upper right", frameon=True,
              edgecolor="#cccccc", framealpha=0.92)
    _classic_axis(ax)

    # Panel B: F-statistics
    ax = axes[1]
    bars2 = ax.bar(x, fstats, color=bar_colors, width=0.55, zorder=3,
                   edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, f in zip(bars2, fstats):
        ax.text(bar.get_x() + bar.get_width()/2, f + 0.1,
                f"F={f:.2f}", ha="center", va="bottom", fontsize=8.0,
                fontweight="bold" if f > 5 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.0)
    ax.set_ylabel("F-statistic", fontsize=9.5)
    ax.set_ylim(0, 10.5)
    _classic_axis(ax)

    fig.tight_layout()
    save("fig7_tms_anova.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Lexicality effect (non-significant)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_lexicality() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # Panel A: Mean accuracy (per-item)
    ax = axes[0]
    features = ["ERP", "DDA"]
    real_acc  = [0.190, 0.079]
    pseudo_acc= [0.233, 0.075]
    chance    = [0.091, 0.091]
    x = np.arange(len(features))
    w = 0.32
    ax.bar(x - w/2, real_acc,   w, color=C_BLUE,   label="Real words",   zorder=3)
    ax.bar(x + w/2, pseudo_acc, w, color=C_ORANGE,  label="Pseudowords",  zorder=3,
           alpha=0.80)
    for xi, ch in zip(x, chance):
        ax.plot([xi - 0.55, xi + 0.55], [ch, ch], color=C_GREY,
                linewidth=1.1, linestyle=":", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9.5)
    ax.set_ylabel("Mean per-item accuracy", fontsize=9.5)
    ax.set_ylim(0.0, 0.32)
    _classic_axis(ax)
    # p-value annotation
    for xi, (t, p) in enumerate(zip([-1.783, 0.187], [0.095, 0.855])):
        ax.text(xi, 0.29, f"t = {t:.2f}\np = {p:.3f}", ha="center", va="top",
                fontsize=7.5, color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    # Panel B: WER (sequence-level, from LOSO)
    ax = axes[1]
    wer_real   = [0.671, 0.688]
    wer_pseudo = [0.780, 0.772]
    wer_std_r  = [0.080, 0.096]
    wer_std_p  = [0.029, 0.050]
    ax.bar(x - w/2, wer_real,   w, color=C_BLUE,  label="Real words",
           yerr=wer_std_r, capsize=4, error_kw={"linewidth":1.1}, zorder=3)
    ax.bar(x + w/2, wer_pseudo, w, color=C_ORANGE, label="Pseudowords",
           yerr=wer_std_p, capsize=4, error_kw={"linewidth":1.1}, zorder=3, alpha=0.80)
    ax.axhline(0.909, color=C_GREY, linewidth=1.1, linestyle="--", label="Chance WER")
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9.5)
    ax.set_ylabel("WER  (LOSO mean ± std)  ↓", fontsize=9.5)
    ax.set_ylim(0.55, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(fontsize=8.5, frameon=True, edgecolor="#cccccc")
    _classic_axis(ax)

    fig.tight_layout()
    save("fig8_lexicality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    setup()
    print("Generating NeurIPS-quality figures for CIPHER...")
    fig1_wer()
    fig2_baselines()
    fig3_controls()
    fig4_time_perm()
    fig5_ablation_heatmap()
    fig6_ablation_deltas()
    fig7_tms()
    fig8_lexicality()
    print(f"\nAll 8 figures saved to: {OUT}")