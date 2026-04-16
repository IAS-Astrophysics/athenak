#!/usr/bin/env python3
"""
Plot the STS diffusion benchmark package.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "doc" / "data" / "sts_diffusion"
FIGURE_DIR = REPO_ROOT / "doc" / "figures" / "sts_diffusion"

SUMMARY_CSV = DATA_DIR / "summary.csv"
PROFILE_CSV = DATA_DIR / "profile_samples.csv"

CASE_META = {
    "hydro_viscosity": {
        "label": "Hydro viscosity",
        "ylabel": r"$v_y$",
    },
    "hydro_conduction": {
        "label": "Hydro conduction",
        "ylabel": r"$\Delta e_{\rm int}$",
    },
    "mhd_resistivity": {
        "label": "MHD resistivity",
        "ylabel": r"$B_y$",
    },
}

METHOD_STYLE = {
    "analytic": {"color": "black", "linestyle": "-", "linewidth": 1.4, "label": "analytic"},
    "explicit": {"color": "#d55e00", "marker": "o", "linestyle": "-", "linewidth": 1.2, "label": "explicit"},
    "sts": {"color": "#0072b2", "marker": "s", "linestyle": "-", "linewidth": 1.2, "label": "STS"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def plot_profiles(profile_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for ax, case_name in zip(axes, CASE_META):
        meta = CASE_META[case_name]
        case_rows = [
            row
            for row in profile_rows
            if row["case"] == case_name and int(row["resolution"]) == 256
        ]
        case_rows.sort(key=lambda row: float(row["x1"]))
        x = np.array([float(row["x1"]) for row in case_rows if row["method"] == "explicit"])
        analytic = np.array(
            [float(row["analytic_value"]) for row in case_rows if row["method"] == "explicit"]
        )
        explicit = np.array(
            [float(row["plotted_value"]) for row in case_rows if row["method"] == "explicit"]
        )
        sts = np.array([float(row["plotted_value"]) for row in case_rows if row["method"] == "sts"])

        ax.plot(x, analytic, **METHOD_STYLE["analytic"])
        ax.plot(x, explicit, **METHOD_STYLE["explicit"])
        ax.plot(x, sts, **METHOD_STYLE["sts"])
        ax.set_title(meta["label"])
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(meta["ylabel"])
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "sts_diffusion_profiles.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "sts_diffusion_profiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_cost(summary_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    for ax, case_name in zip(axes, CASE_META):
        meta = CASE_META[case_name]
        for method in ("explicit", "sts"):
            rows = [
                row for row in summary_rows if row["case"] == case_name and row["method"] == method
            ]
            rows.sort(key=lambda row: int(row["resolution"]))
            wall = np.array([float(row["wall_seconds_median"]) for row in rows])
            err = np.array([float(row["rms_l1_median"]) for row in rows])
            labels = [int(row["resolution"]) for row in rows]
            ax.plot(wall, err, **METHOD_STYLE[method])
            for xval, yval, res in zip(wall, err, labels):
                ax.annotate(str(res), (xval, yval), textcoords="offset points", xytext=(4, 4), fontsize=8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(meta["label"])
        ax.set_xlabel("median wall time [s]")
        ax.set_ylabel("L1 RMS error")
        ax.grid(alpha=0.25, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "sts_diffusion_accuracy_cost.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "sts_diffusion_accuracy_cost.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary_rows = read_csv(SUMMARY_CSV)
    profile_rows = read_csv(PROFILE_CSV)
    plot_profiles(profile_rows)
    plot_accuracy_cost(summary_rows)


if __name__ == "__main__":
    main()
