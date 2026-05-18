#!/usr/bin/env python3
"""Plot 200M TOV stability runs and mark boundary causal-contact times."""

from __future__ import annotations

import argparse
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


def parse_input(path: Path) -> dict:
    text = path.read_text()
    def get_float(name: str) -> float:
        return float(re.search(rf"^\s*{name}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE).group(1))
    def get_int(name: str) -> int:
        return int(re.search(rf"^\s*{name}\s*=\s*([0-9]+)", text, re.MULTILINE).group(1))
    nx = get_int("nx1")
    x1min = get_float("x1min")
    x1max = get_float("x1max")
    half_width = 0.5*(x1max - x1min)
    boost = get_float("star_boost_y")
    amr = "refinement = adaptive" in text
    dx_root = 2.0*half_width/nx
    finest_dx = 0.5*dx_root if amr else dx_root
    return {
        "nx": nx,
        "half_width": half_width,
        "boost": boost,
        "amr": amr,
        "dx_root": dx_root,
        "finest_dx": finest_dx,
        "cells_across": 2.0*RADIUS/finest_dx,
        "contact_surface": (half_width - RADIUS)/(1.0 + abs(boost)),
        "contact_center": half_width/(1.0 + abs(boost)),
        "surface_reaches_boundary": (half_width - RADIUS)/abs(boost) if abs(boost) > 0.0 else np.nan,
        "center_reaches_boundary": half_width/abs(boost) if abs(boost) > 0.0 else np.nan,
    }


def read_case(case_dir: Path) -> dict:
    info = parse_input(case_dir / "input.athinput")
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
    time = user[:, uidx["time"]]
    rho = user[:, uidx["rho-max"]]
    rho_norm = rho/rho[0]
    mass = mhd[:, midx["mass"]]
    finite_c = np.isfinite(z4c[:, zidx["C-norm2"]])
    first_nan = np.nan
    if (~finite_c).any():
        first_nan = float(z4c[np.argmax(~finite_c), zidx["time"]])
    below_half = np.where(rho_norm < 0.5)[0]
    below_90 = np.where(rho_norm < 0.9)[0]
    tail = max(3, len(rho)//3)
    tail_time = time[-tail:]
    tail_rho = rho[-tail:]
    tail_mean = np.nanmean(tail_rho)
    tail_std = np.nanstd(tail_rho)/tail_mean if tail_mean != 0.0 else np.nan
    tail_drift = np.nan
    if len(tail_time) >= 3 and tail_mean != 0.0:
        tail_drift = np.polyfit(tail_time, tail_rho, 1)[0]*(tail_time[-1] - tail_time[0])/tail_mean
    return {
        **info,
        "case": case_dir.name,
        "time": time,
        "rho_norm": rho_norm,
        "z4c_time": z4c[:, zidx["time"]],
        "constraint": z4c[:, zidx["C-norm2"]],
        "mass_time": mhd[:, midx["time"]],
        "mass_norm": mass/mass[0] if mass[0] != 0.0 else np.full_like(mass, np.nan),
        "final_time": float(time[-1]),
        "final_rho_norm": float(rho_norm[-1]),
        "tail_std": float(tail_std),
        "tail_drift": float(tail_drift),
        "mass_drift": float(mass[-1]/mass[0] - 1.0) if mass[0] != 0.0 else np.nan,
        "t_below_0p9": float(time[below_90[0]]) if below_90.size else np.nan,
        "t_below_0p5": float(time[below_half[0]]) if below_half.size else np.nan,
        "t_first_nan": first_nan,
    }


def label(case: dict) -> str:
    mode = "AMR" if case["amr"] else "uniform"
    boost = "boosted" if case["boost"] != 0.0 else "unboosted"
    return f"{boost}, L={case['half_width']:g}, {mode}, {case['cells_across']:.1f} cells/diam"


def add_contact_lines(ax, case: dict, color: str) -> None:
    ax.axvline(case["contact_surface"], color=color, ls=":", lw=0.9, alpha=0.75)
    ax.axvline(case["contact_center"], color=color, ls="--", lw=0.9, alpha=0.65)
    if np.isfinite(case["surface_reaches_boundary"]):
        ax.axvline(case["surface_reaches_boundary"], color=color, ls="-.", lw=0.9, alpha=0.75)


def plot_all(cases: list[dict], out_dir: Path) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, case in enumerate(cases):
        color = colors[idx % len(colors)]
        ax.plot(case["time"], case["rho_norm"], lw=1.5, color=color, label=label(case))
        add_contact_lines(ax, case, color)
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.axhline(0.9, color="0.5", lw=0.8, ls="--")
    ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("t [M]")
    ax.set_ylabel(r"$\rho_{\max}/\rho_{\max}(0)$")
    ax.set_title("Central density stability to 200M")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(0.99, 0.02,
            "dotted: surface-boundary causal contact; dashed: center-boundary contact; dash-dot: boosted surface reaches +y boundary",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_max_200m_with_boundary_contact.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, case in enumerate(cases):
        color = colors[idx % len(colors)]
        ax.semilogy(case["time"], np.maximum(case["rho_norm"], 1.0e-12),
                    lw=1.5, color=color, label=label(case))
        add_contact_lines(ax, case, color)
    ax.axhline(0.9, color="0.5", lw=0.8, ls="--")
    ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("t [M]")
    ax.set_ylabel(r"$\rho_{\max}/\rho_{\max}(0)$")
    ax.set_title("Central density stability to 200M, log scale")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_max_200m_log_with_boundary_contact.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, case in enumerate(cases):
        color = colors[idx % len(colors)]
        y = np.where(np.isfinite(case["constraint"]), case["constraint"], np.nan)
        ax.semilogy(case["z4c_time"], y, lw=1.5, color=color, label=label(case))
        add_contact_lines(ax, case, color)
    ax.set_xlabel("t [M]")
    ax.set_ylabel("Z4c C-norm2")
    ax.set_title("Constraint history to 200M")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "z4c_constraint_200m_with_boundary_contact.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, case in enumerate(cases):
        color = colors[idx % len(colors)]
        ax.plot(case["mass_time"], case["mass_norm"] - 1.0, lw=1.5, color=color, label=label(case))
        add_contact_lines(ax, case, color)
    ax.set_xlabel("t [M]")
    ax.set_ylabel(r"$M_b/M_b(0)-1$")
    ax.set_title("Baryonic mass drift to 200M")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mass_drift_200m_with_boundary_contact.png", dpi=180)
    plt.close(fig)


def write_csv(cases: list[dict], out_dir: Path) -> None:
    lines = [
        "case,boost,L,nx,amr,finest_dx,cells_across_star,contact_surface,contact_center,"
        "surface_reaches_boundary,center_reaches_boundary,"
        "final_t,final_rho_norm,tail_std,tail_drift,mass_drift,t_below_0.9,t_below_0.5,t_first_nan"
    ]
    for case in cases:
        lines.append(
            f"{case['case']},{case['boost']:.8g},{case['half_width']:.8g},{case['nx']},"
            f"{int(case['amr'])},{case['finest_dx']:.8g},{case['cells_across']:.8g},"
            f"{case['contact_surface']:.8g},{case['contact_center']:.8g},"
            f"{case['surface_reaches_boundary']:.8g},{case['center_reaches_boundary']:.8g},"
            f"{case['final_time']:.8g},{case['final_rho_norm']:.8g},"
            f"{case['tail_std']:.8g},{case['tail_drift']:.8g},{case['mass_drift']:.8g},"
            f"{case['t_below_0p9']:.8g},{case['t_below_0p5']:.8g},{case['t_first_nan']:.8g}"
        )
    (out_dir / "tov_200m_summary.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [read_case(path) for path in sorted(args.run_root.iterdir())
             if path.is_dir() and (path / "input.athinput").exists()]
    plot_all(cases, out_dir)
    write_csv(cases, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
