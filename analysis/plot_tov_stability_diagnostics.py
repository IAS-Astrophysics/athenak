#!/usr/bin/env python3
"""Plot TOV Minkowski stability histories from run_tov_stability.py outputs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RADIUS = 1.15235


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def read_hst(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    header = next(line for line in lines if line.startswith("#  [1]="))
    names = [token.split("]=", 1)[1] for token in header.split() if "]=" in token]
    data = np.loadtxt(path, comments="#", ndmin=2)
    return names, data


def read_case(case_dir: Path) -> dict:
    user_file = next(path for path in case_dir.glob("*.user.hst")
                     if not path.name.endswith(".z4c.user.hst"))
    z4c_file = next(case_dir.glob("*.z4c.user.hst"))
    mhd_file = next(case_dir.glob("*.mhd.hst"))
    user_names, user = read_hst(user_file)
    z4c_names, z4c = read_hst(z4c_file)
    mhd_names, mhd = read_hst(mhd_file)
    uidx = {name: i for i, name in enumerate(user_names)}
    zidx = {name: i for i, name in enumerate(z4c_names)}
    midx = {name: i for i, name in enumerate(mhd_names)}

    text = (case_dir / "input.athinput").read_text()
    nx = int(re.search(r"nx1\s*=\s*([0-9]+)", text).group(1))
    x1min = float(re.search(r"x1min\s*=\s*([-+0-9.eE]+)", text).group(1))
    x1max = float(re.search(r"x1max\s*=\s*([-+0-9.eE]+)", text).group(1))
    boost = float(re.search(r"star_boost_y\s*=\s*([-+0-9.eE]+)", text).group(1))
    half_width = 0.5*(x1max - x1min)
    cells_across = 2.0*RADIUS/(2.0*half_width/nx)

    rho = user[:, uidx["rho-max"]]
    time = user[:, uidx["time"]]
    rho_norm = rho/rho[0]
    finite_constraints = np.isfinite(z4c[:, zidx["C-norm2"]])
    first_bad = float("nan")
    if not np.all(finite_constraints):
        first_bad = float(z4c[np.argmax(~finite_constraints), zidx["time"]])
    below_half = np.where(rho_norm < 0.5)[0]
    t_half = float(time[below_half[0]]) if below_half.size else float("nan")
    below_floor = np.where(rho_norm < 1.0e-4)[0]
    t_floor = float(time[below_floor[0]]) if below_floor.size else float("nan")

    mass = mhd[:, midx["mass"]]
    return {
        "name": case_dir.name,
        "kind": "boosted" if boost != 0.0 else "unboosted",
        "nx": nx,
        "half_width": half_width,
        "boost": boost,
        "cells_across": cells_across,
        "time": time,
        "rho_norm": rho_norm,
        "mass_time": mhd[:, midx["time"]],
        "mass_norm": mass/mass[0] if mass[0] != 0.0 else np.full_like(mass, np.nan),
        "z4c_time": z4c[:, zidx["time"]],
        "constraint": z4c[:, zidx["C-norm2"]],
        "t_half": t_half,
        "t_floor": t_floor,
        "first_bad": first_bad,
    }


def plot_density(cases: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, ["unboosted", "boosted"]):
        for case in sorted([c for c in cases if c["kind"] == kind],
                           key=lambda x: (x["half_width"], x["nx"])):
            label = f"L={case['half_width']:g}, N={case['nx']}, {case['cells_across']:.1f} cells/diam"
            ax.plot(case["time"], case["rho_norm"], marker=".", ms=3, lw=1.2, label=label)
        ax.axhline(0.9, color="0.65", ls="--", lw=0.8)
        ax.axhline(0.5, color="0.65", ls=":", lw=0.8)
        ax.set_title(kind)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\rho_{\max}/\rho_{\max}(0)$")
    fig.suptitle("Central density proxy")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_max_normalized.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, ["unboosted", "boosted"]):
        for case in sorted([c for c in cases if c["kind"] == kind],
                           key=lambda x: (x["half_width"], x["nx"])):
            label = f"L={case['half_width']:g}, N={case['nx']}"
            ax.semilogy(case["time"], np.maximum(case["rho_norm"], 1.0e-12),
                        marker=".", ms=3, lw=1.2, label=label)
        ax.axhline(1.0e-4, color="0.65", ls="--", lw=0.8)
        ax.set_title(kind)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\rho_{\max}/\rho_{\max}(0)$")
    fig.suptitle("Central density proxy, log scale")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_max_normalized_log.png", dpi=180)
    plt.close(fig)


def plot_constraints(cases: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, kind in zip(axes, ["unboosted", "boosted"]):
        for case in sorted([c for c in cases if c["kind"] == kind],
                           key=lambda x: (x["half_width"], x["nx"])):
            label = f"L={case['half_width']:g}, N={case['nx']}"
            y = np.where(np.isfinite(case["constraint"]), case["constraint"], np.nan)
            ax.semilogy(case["z4c_time"], y, marker=".", ms=3, lw=1.2, label=label)
        ax.set_title(kind)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Z4c C-norm2")
    fig.suptitle("Constraint growth before failure")
    fig.tight_layout()
    fig.savefig(out_dir / "z4c_constraint_norm.png", dpi=180)
    plt.close(fig)


def write_csv(cases: list[dict], out_dir: Path) -> None:
    lines = ["case,boost,L,nx,cells_across_star,t_rho_below_0.5,t_rho_below_1e-4,t_first_nan_constraint"]
    for case in sorted(cases, key=lambda c: (c["boost"], c["half_width"], c["nx"])):
        lines.append(
            f"{case['name']},{case['boost']:.6g},{case['half_width']:.6g},{case['nx']},"
            f"{case['cells_across']:.6g},{case['t_half']:.6g},{case['t_floor']:.6g},"
            f"{case['first_bad']:.6g}"
        )
    (out_dir / "failure_times.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [read_case(case_dir) for case_dir in sorted(args.run_root.iterdir())
             if case_dir.is_dir() and any(
                 not path.name.endswith(".z4c.user.hst") for path in case_dir.glob("*.user.hst"))]
    plot_density(cases, out_dir)
    plot_constraints(cases, out_dir)
    write_csv(cases, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
