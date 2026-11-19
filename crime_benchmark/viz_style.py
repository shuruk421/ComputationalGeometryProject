"""Shared colours and matplotlib styling for every crime_benchmark figure.

Palette = slots 1-3 of the dataviz reference palette (validated all-pairs for
colour-vision deficiency on the light surface):
  incremental = blue, decremental = orange (same in every figure),
  consenting points = aqua (sub-3:1 contrast, so it always ships with a legend).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INC = "#2a78d6"
DEC = "#eb6834"
ALGO_COLOR = {"incremental": INC, "decremental": DEC}
ALGO_LABEL = {"incremental": "Incremental", "decremental": "Decremental"}
ALGO_MARKER = {"incremental": "o", "decremental": "s"}

CONSENT = "#1baf7a"
NONCONSENT = "#898781"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def apply():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
