"""Regenerate the ClimbMix d12 validation-BPB trajectory figure.

The figure combines the original 2,520-step matched comparison with the
stabilized g=4 continuation to 5,040 steps. The extension is intentionally
shown with a different line style because the MLP baseline has not been
continued to the same horizon and the g=4 continuation reset the LR schedule.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent

MLP = [
    (250, 1.107120474222),
    (500, 1.006246294517),
    (750, 0.971314325089),
    (1000, 0.946986960359),
    (1250, 0.922800919706),
    (1500, 0.902853844373),
    (1750, 0.885606251068),
    (2000, 0.869643676194),
    (2250, 0.856451158043),
    (2500, 0.846898069695),
    (2520, 0.846393914830),
]

GRKAN_G4 = [
    (250, 1.184027),
    (500, 1.074947614287),
    (750, 1.031688),
    (1000, 0.994434466596),
    (1250, 0.966246),
    (1500, 0.977889592270),
    (1750, 0.934710),
    (2000, 0.916257038466),
    (2250, 0.901259),
    (2500, 0.890883456311),
    (2520, 0.890394441231),
]

GRKAN_G8 = [
    (500, 1.030523756184),
    (750, 1.053212),
    (1000, 1.031318594936),
    (1250, 0.989453),
    (1500, 0.964506953347),
    (1750, 0.944583),
    (2000, 0.927689381834),
    (2250, 0.913563),
    (2500, 0.903534960042),
    (2520, 0.902987947937),
]

GRKAN_G4_STABILIZED = [
    (2520, 0.890394441231),
    (2600, 0.890394441231),
    (2700, 0.890394441231),
    (2800, 0.889500786262),
    (2900, 0.889500786262),
    (3000, 0.882262571156),
    (3250, 0.924662),
    (3500, 0.911001),
    (3750, 0.901764),
    (4000, 0.891866610705),
    (4250, 0.882625),
    (4500, 0.874845),
    (4750, 0.867438),
    (5000, 0.862113845522),
    (5040, 0.861594492641),
]


def plot_series(ax, data, *, label, color, marker, linestyle, linewidth=1.7):
    x, y = zip(*data)
    ax.plot(
        x,
        y,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=4.5,
        markeredgewidth=0.7,
    )


def main() -> None:
    # Okabe-Ito colorblind-safe palette.
    blue = "#0072B2"
    orange = "#D55E00"
    purple = "#CC79A7"
    gray = "#666666"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.7, 3.45), constrained_layout=True)

    ax.axvspan(2520, 5040, color="#F3F3F3", alpha=0.75, zorder=0)
    plot_series(
        ax,
        MLP,
        label="SwiGLU MLP",
        color=blue,
        marker="o",
        linestyle="-",
        linewidth=1.9,
    )
    plot_series(
        ax,
        GRKAN_G4,
        label="GR-KAN g=4",
        color=orange,
        marker="s",
        linestyle="-",
    )
    plot_series(
        ax,
        GRKAN_G8,
        label="GR-KAN g=8",
        color=purple,
        marker="^",
        linestyle="-",
    )
    plot_series(
        ax,
        GRKAN_G4_STABILIZED,
        label="GR-KAN g=4 stabilized continuation",
        color=orange,
        marker="D",
        linestyle="--",
        linewidth=1.65,
    )

    mlp_final = MLP[-1][1]
    ax.hlines(
        mlp_final,
        xmin=2520,
        xmax=5040,
        colors=blue,
        linestyles=":",
        linewidth=1.4,
        label="MLP step-2520 reference",
    )

    ax.annotate(
        "0.8616",
        xy=(5040, 0.861594492641),
        xytext=(4580, 0.842),
        arrowprops={"arrowstyle": "->", "color": orange, "lw": 0.8},
        color=orange,
        fontsize=7.5,
        ha="left",
    )
    ax.annotate(
        "MLP 2520: 0.8464",
        xy=(2520, mlp_final),
        xytext=(2770, 0.832),
        arrowprops={"arrowstyle": "->", "color": blue, "lw": 0.8},
        color=blue,
        fontsize=7.5,
        ha="left",
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_xlim(200, 5200)
    ax.set_ylim(0.82, 1.20)
    ax.set_xticks([500, 1000, 1500, 2000, 2520, 3000, 4000, 5040])
    ax.set_yticks([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=1)

    fig.savefig(OUT_DIR / "scale_bpb_trajectories.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "scale_bpb_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
