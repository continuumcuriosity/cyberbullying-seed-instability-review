#!/usr/bin/env python3
"""Regenerate the manuscript's two-panel cross-domain results figure."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "figures" / "fig2_tradeoff.pdf"
ARMS = ["soft_uw1", "soft_off", "main", "no_contrastive"]
MARK = {"soft_uw1": "s", "soft_off": "o", "main": "^", "no_contrastive": "D"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

with (DATA / "runs12.csv").open(newline="") as stream:
    runs = list(csv.DictReader(stream))
for row in runs:
    row["test_f1"] = float(row["test_f1"])
    row["recall"] = float(row["recall"])

with (DATA / "votes12.csv").open(newline="") as stream:
    votes = list(csv.DictReader(stream))[1:]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.5))
for arm in ARMS:
    points = [row for row in runs if row["arm"] == arm]
    ax.plot([row["test_f1"] for row in points],
            [row["recall"] for row in points], MARK[arm], ms=4.2,
            mfc="white", mec="black", mew=0.9, ls="none", zorder=3,
            label=arm)
xs = np.array([row["test_f1"] for row in runs])
ys = np.array([row["recall"] for row in runs])
slope, intercept = np.polyfit(xs, ys, 1)
grid = np.linspace(xs.min(), xs.max(), 50)
ax.plot(grid, intercept + slope * grid, "--", lw=0.8, color="0.6", zorder=2)
ax.set_xlabel("In-distribution test macro-$F_1$")
ax.set_ylabel("Cyberbullying recall")
ax.set_title("(a) run-level association", fontsize=7.5, pad=4)
ax.text(0.03, 0.04,
        "$\\rho=-0.739$ (raw; $p=0.006$)\n"
        "$\\rho=-0.431$ (partial; $p=0.186$)",
        transform=ax.transAxes, fontsize=6.2, va="bottom", color="0.3")
ax.legend(loc="upper right", frameon=False, fontsize=6.2, numpoints=1,
          handletextpad=0.25, borderpad=0.1, labelspacing=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(lw=0.3, color="0.93", zorder=0)

ks = [int(row["k"]) for row in votes]
harmful = [int(row["n_harmful"]) for row in votes]
benign = [int(row["n_benign"]) for row in votes]
bx.bar(ks, harmful, width=0.78, color="0.35", ec="black", lw=0.5,
       label="harmful", zorder=3)
bx.bar(ks, benign, width=0.78, bottom=harmful, color="0.85", ec="black",
       lw=0.5, label="benign", zorder=3)
bx.set_xlabel("Number of the 12 runs flagging the comment")
bx.set_ylabel("Comments")
bx.set_title("(b) item-level disagreement", fontsize=7.5, pad=4)
bx.set_xticks(range(1, 13))
bx.legend(loc="upper center", frameon=False, fontsize=6.2,
          handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
bx.spines[["top", "right"]].set_visible(False)
bx.grid(axis="y", lw=0.3, color="0.93", zorder=0)

fig.tight_layout(pad=0.4, w_pad=2.2)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"Updated {OUT}")
