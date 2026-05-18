#!/usr/bin/env python3
"""Plot and summarize nested TOV damping-sweep outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RADIUS = 1.15235
CASE_ORDER = ["unboosted_L3_n48", "boosted_L3_n48", "unboosted_L5_n80", "boosted_L5_n80"]
CONFIG_ORDER = [
    "no_diss_no_damp",
    "diss_0p1_no_damp",
    "diss_0p5_no_damp",
    "diss_0p5_k1_0p02_k2_0",
    "diss_0p5_k1_0p02_k2_0p02",
]


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
        return float(re.search(rf"^{name}\s*=\s*([-+0-9.eE]+)", text, re.MULTILINE).group(1))
    def get_int(name: str) -> int:
        return int(re.search(rf"^{name}\s*=\s*([0-9]+)", text, re.MULTILINE).group(1))
    nx = get_int("nx1")
    half_width = 0.5*(get_float("x1max") - get_float("x1min"))
    dx = 2.0*half_width/nx
    return {
        "nx": nx,
        "half_width": half_width,
        "boost": get_float("star_boost_y"),
        "diss": get_float("diss"),
        "kappa1": get_float("damp_kappa1"),
        "kappa2": get_float("damp_kappa2"),
        "cells_across": 2.0*RADIUS/dx,
    }


def read_case(config_dir: Path, case_dir: Path) -> dict:
    input_info = parse_input(case_dir / "input.athinput")
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
    rho_half = np.where(rho_norm < 0.5)[0]
    rho_floor = np.where(rho_norm < 1.0e-4)[0]
    finite_constraint = np.isfinite(z4c[:, zidx["C-norm2"]])
    first_bad = np.where(~finite_constraint)[0]
    mass = mhd[:, midx["mass"]]
    tail = max(3, len(rho)//3)
    tail_time = time[-tail:]
    tail_rho = rho[-tail:]
    tail_mean = np.nanmean(tail_rho)
    tail_std = np.nanstd(tail_rho)/tail_mean if tail_mean != 0.0 else np.nan
    tail_drift = np.nan
    if len(tail_time) >= 3 and tail_mean != 0.0:
        tail_drift = np.polyfit(tail_time, tail_rho, 1)[0]*(tail_time[-1] - tail_time[0])/tail_mean

    return {
        **input_info,
        "config": config_dir.name,
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
        "t_half": float(time[rho_half[0]]) if rho_half.size else np.nan,
        "t_floor": float(time[rho_floor[0]]) if rho_floor.size else np.nan,
        "t_first_nan": float(z4c[first_bad[0], zidx["time"]]) if first_bad.size else np.nan,
        "settled": bool(
            time[-1] >= 19.6
            and np.isfinite(tail_std) and tail_std < 0.05
            and np.isfinite(tail_drift) and abs(tail_drift) < 0.05
            and np.isfinite(rho_norm[-1]) and 0.5 < rho_norm[-1] < 1.5
            and not first_bad.size
        ),
    }


def sorted_configs(records: list[dict]) -> list[str]:
    names = sorted({r["config"] for r in records})
    return sorted(names, key=lambda n: CONFIG_ORDER.index(n) if n in CONFIG_ORDER else len(CONFIG_ORDER))


def plot_rho(records: list[dict], out_dir: Path) -> None:
    configs = sorted_configs(records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, case in zip(axes.ravel(), CASE_ORDER):
        subset = [r for r in records if r["case"] == case]
        for config in configs:
            recs = [r for r in subset if r["config"] == config]
            if not recs:
                continue
            r = recs[0]
            ax.plot(r["time"], np.maximum(r["rho_norm"], 1.0e-12), lw=1.3, label=config)
        ax.axhline(0.9, color="0.65", ls="--", lw=0.8)
        ax.axhline(0.5, color="0.65", ls=":", lw=0.8)
        ax.set_title(case)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, which="both")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\rho_{\max}/\rho_{\max}(0)$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Central density proxy: damping/dissipation sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_max_damping_sweep.png", dpi=180)
    plt.close(fig)


def plot_constraints(records: list[dict], out_dir: Path) -> None:
    configs = sorted_configs(records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, case in zip(axes.ravel(), CASE_ORDER):
        subset = [r for r in records if r["case"] == case]
        for config in configs:
            recs = [r for r in subset if r["config"] == config]
            if not recs:
                continue
            r = recs[0]
            y = np.where(np.isfinite(r["constraint"]), r["constraint"], np.nan)
            ax.semilogy(r["z4c_time"], y, lw=1.3, label=config)
        ax.set_title(case)
        ax.grid(True, alpha=0.25, which="both")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("Z4c C-norm2")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Constraint history: damping/dissipation sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "z4c_constraint_damping_sweep.png", dpi=180)
    plt.close(fig)


def plot_failure_bars(records: list[dict], out_dir: Path) -> None:
    configs = sorted_configs(records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    x = np.arange(len(configs))
    for ax, case in zip(axes.ravel(), CASE_ORDER):
        vals = []
        for config in configs:
            recs = [r for r in records if r["case"] == case and r["config"] == config]
            vals.append(recs[0]["t_floor"] if recs and np.isfinite(recs[0]["t_floor"]) else recs[0]["final_time"])
        ax.bar(x, vals)
        ax.set_title(case)
        ax.set_xticks(x, configs, rotation=35, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"first $t$: $\rho_{\max}/\rho_0 < 10^{-4}$")
    fig.suptitle("Central-density collapse time")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_collapse_time_damping_sweep.png", dpi=180)
    plt.close(fig)


def write_csv(records: list[dict], out_dir: Path) -> None:
    lines = [
        "config,case,diss,kappa1,kappa2,boost,L,nx,cells_across_star,final_t,"
        "final_rho_norm,tail_std,tail_drift,mass_drift,t_rho_below_0.5,"
        "t_rho_below_1e-4,t_first_nan_constraint,settled"
    ]
    for r in sorted(records, key=lambda x: (sorted_configs(records).index(x["config"]), CASE_ORDER.index(x["case"]))):
        lines.append(
            f"{r['config']},{r['case']},{r['diss']:.8g},{r['kappa1']:.8g},{r['kappa2']:.8g},"
            f"{r['boost']:.8g},{r['half_width']:.8g},{r['nx']},{r['cells_across']:.8g},"
            f"{r['final_time']:.8g},{r['final_rho_norm']:.8g},{r['tail_std']:.8g},"
            f"{r['tail_drift']:.8g},{r['mass_drift']:.8g},{r['t_half']:.8g},"
            f"{r['t_floor']:.8g},{r['t_first_nan']:.8g},{int(r['settled'])}"
        )
    (out_dir / "damping_sweep_summary.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.run_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for config_dir in sorted(args.run_root.iterdir()):
        if not config_dir.is_dir():
            continue
        for case_dir in sorted(config_dir.iterdir()):
            if case_dir.is_dir() and (case_dir / "input.athinput").exists():
                records.append(read_case(config_dir, case_dir))
    plot_rho(records, out_dir)
    plot_constraints(records, out_dir)
    plot_failure_bars(records, out_dir)
    write_csv(records, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
