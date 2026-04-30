#!/usr/bin/env python3
"""Run and plot a compact Kerr photon-orbit beam test in CKS and ADM modes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_comparisons import read_binary_slice  # noqa: E402


SPIN = -0.5
ORBIT_R = 3.532088886237956
SOURCE_PHI = 0.0
SOLVERS = ("legacy", "dyn_cks", "dyn_adm")
SOLVER_LABELS = {
    "legacy": "legacy radiation",
    "dyn_cks": "dyn_radiation CKS",
    "dyn_adm": "dyn_radiation ADM",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cartesian_orbit_radius(spin: float = SPIN, orbit_r: float = ORBIT_R) -> float:
    return float(np.sqrt(orbit_r**2 + spin**2))


def horizon_cartesian_radius(spin: float = SPIN) -> float:
    r_h = 1.0 + np.sqrt(1.0 - spin**2)
    return float(np.sqrt(r_h**2 + spin**2))


def nangles(nlevel: int) -> int:
    return 10 * nlevel * nlevel + 2


def run_case(root: Path, run_dir: Path, solver: str, nlevel: int,
             tlim: float, nx: int) -> Path:
    exe = root / "build" / "src" / "athena"
    input_names = {
        "legacy": "rad_kerr_orbit_beam.athinput",
        "dyn_cks": "dynrad_kerr_orbit_beam.athinput",
        "dyn_adm": "dynrad_kerr_orbit_beam_adm.athinput",
    }
    input_file = root / "inputs" / "tests" / input_names[solver]
    basename = f"kerr_orbit_{solver}_n{nlevel}"
    block = "radiation" if solver == "legacy" else "dyn_radiation"
    cmd = [
        str(exe),
        "-i", str(input_file),
        "-d", str(run_dir),
        f"job/basename={basename}",
        f"mesh/nx1={nx}",
        f"mesh/nx2={nx}",
        f"meshblock/nx1={nx // 2}",
        f"meshblock/nx2={nx // 2}",
        f"time/tlim={tlim}",
        f"output1/dt={tlim}",
        f"{block}/nlevel={nlevel}",
    ]
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_btl", "self")
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, env=env)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"{basename}.out").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / f"{basename}.err").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{basename} failed with exit code {result.returncode}")
    outputs = sorted(run_dir.glob(f"{basename}.rad_xy.*.bin"))
    outputs += sorted((run_dir / "bin").glob(f"{basename}.rad_xy.*.bin"))
    if not outputs:
        raise FileNotFoundError(f"no binary rad_xy output for {basename}")
    return outputs[-1]


def decorate(ax: plt.Axes) -> None:
    r_orbit = cartesian_orbit_radius()
    r_h = horizon_cartesian_radius()
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(r_orbit * np.cos(theta), r_orbit * np.sin(theta),
            color="#8bc34a", lw=1.25, alpha=0.9, ls="--")
    ax.add_patch(plt.Circle((0.0, 0.0), r_h, facecolor="black",
                            edgecolor="black", lw=0.0, zorder=5))
    source_x = r_orbit * np.cos(SOURCE_PHI)
    source_y = r_orbit * np.sin(SOURCE_PHI)
    ax.add_patch(plt.Circle((source_x, source_y), 0.22, edgecolor="#21d4e8",
                            facecolor="none", lw=1.2, zorder=6))
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/dynrad_kerr_orbit_beam"))
    parser.add_argument("--fig-dir", type=Path,
                        default=Path("dyngr_radiation_paper/figures"))
    parser.add_argument("--nlevel", type=int, default=4)
    parser.add_argument("--tlim", type=float, default=5.0)
    parser.add_argument("--nx", type=int, default=72)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_dir = args.run_dir
    fig_dir = args.fig_dir if args.fig_dir.is_absolute() else root / args.fig_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fields: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for solver in SOLVERS:
        basename = f"kerr_orbit_{solver}_n{args.nlevel}"
        if args.skip_run:
            outputs = sorted(run_dir.glob(f"{basename}.rad_xy.*.bin"))
            outputs += sorted((run_dir / "bin").glob(f"{basename}.rad_xy.*.bin"))
            if not outputs:
                raise FileNotFoundError(f"missing output for {basename}")
            path = outputs[-1]
        else:
            path = run_case(root, run_dir, solver, args.nlevel, args.tlim, args.nx)
        data = read_binary_slice(path, ["r00"])
        fields[solver] = (data.x1v, data.x2v, np.asarray(data.variables["r00"], dtype=float))

    x, y, legacy = fields["legacy"]
    _, _, dyn_cks = fields["dyn_cks"]
    _, _, dyn_adm = fields["dyn_adm"]
    vmax = max(float(np.nanmax(legacy)), float(np.nanmax(dyn_cks)),
               float(np.nanmax(dyn_adm)))
    if vmax <= 0.0:
        raise RuntimeError("Kerr beam output is empty")
    legacy_n = legacy / vmax
    dyn_cks_n = dyn_cks / vmax
    dyn_adm_n = dyn_adm / vmax
    diff_cks = np.abs(dyn_cks - legacy) / vmax
    diff_adm = np.abs(dyn_adm - dyn_cks) / vmax
    linf_cks = float(np.nanmax(diff_cks))
    l1_cks = float(np.nanmean(np.abs(dyn_cks - legacy)) /
                   max(np.nanmean(np.abs(legacy)), 1.0e-300))
    linf_adm = float(np.nanmax(diff_adm))
    l1_adm = float(np.nanmean(np.abs(dyn_adm - dyn_cks)) /
                   max(np.nanmean(np.abs(dyn_cks)), 1.0e-300))
    print(f"nlevel={args.nlevel} Nang={nangles(args.nlevel)} "
          f"|dyn_cks-legacy|_inf/vmax={linf_cks:.6e} rel_L1={l1_cks:.6e} "
          f"|dyn_adm-dyn_cks|_inf/vmax={linf_adm:.6e} rel_L1={l1_adm:.6e}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2))
    fig.subplots_adjust(left=0.08, right=0.82, bottom=0.08, top=0.93,
                        wspace=0.16, hspace=0.24)
    axes = axes.ravel()
    panels = [
        ("legacy radiation", legacy_n, 0.0, 1.0, "magma"),
        ("dyn_radiation CKS", dyn_cks_n, 0.0, 1.0, "magma"),
        ("dyn_radiation ADM", dyn_adm_n, 0.0, 1.0, "magma"),
        (r"$|ADM-CKS|$", diff_adm, 0.0, max(linf_adm, 1.0e-12), "viridis"),
    ]
    ims = []
    for ax, (title, field, vmin, vmax_panel, cmap) in zip(axes, panels):
        im = ax.pcolormesh(x, y, field, shading="nearest", vmin=vmin,
                           vmax=vmax_panel, cmap=cmap)
        decorate(ax)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ims.append(im)
    axes[0].set_ylabel(r"$y$")
    axes[2].set_ylabel(r"$y$")
    axes[1].text(0.97, 0.94, rf"$N_{{ang}}={nangles(args.nlevel)}$",
                 transform=axes[1].transAxes, color="white", ha="right", va="top")
    cax_intensity = fig.add_axes([0.85, 0.18, 0.025, 0.68])
    fig.colorbar(ims[1], cax=cax_intensity,
                 label=r"$R^{tt}/\max(R^{tt})$")
    fig.colorbar(ims[3], ax=axes[3], orientation="horizontal",
                 shrink=0.86, pad=0.12, label=r"normalized difference")
    out = fig_dir / "kerr_orbit_beam_comparison.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
