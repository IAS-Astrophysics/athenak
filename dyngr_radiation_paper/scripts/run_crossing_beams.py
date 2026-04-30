#!/usr/bin/env python3
"""Run and plot the crossing-beams test in CKS and ADM geometry modes."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_comparisons import read_binary_slice  # noqa: E402


BEAM = {
    "amp": 1.0,
    "sigma": 0.055,
    "x0": 0.12,
    "y_lower": 0.15,
    "y_upper": 0.85,
    "x_cross": 0.75,
    "y_cross": 0.5,
}

SOLVERS = ("legacy", "dyn_cks", "dyn_adm")
SOLVER_LABELS = {
    "legacy": "legacy radiation",
    "dyn_cks": "dyn_radiation CKS",
    "dyn_adm": "dyn_radiation ADM",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def nangles(nlevel: int) -> int:
    return 10 * nlevel * nlevel + 2


def target_direction(y0: float) -> tuple[float, float]:
    dx = BEAM["x_cross"] - BEAM["x0"]
    dy = BEAM["y_cross"] - y0
    norm = float(np.hypot(dx, dy))
    return dx / norm, dy / norm


def beam_profile(x: np.ndarray, y: np.ndarray, y0: float) -> np.ndarray:
    qx, qy = target_direction(y0)
    dx = x - BEAM["x0"]
    dy = y - y0
    along = dx * qx + dy * qy
    perp = -dx * qy + dy * qx
    return BEAM["amp"] * np.exp(-0.5 * (perp / BEAM["sigma"]) ** 2) * (along >= 0.0)


def exact_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return beam_profile(xx, yy, BEAM["y_lower"]) + beam_profile(xx, yy, BEAM["y_upper"])


def run_case(root: Path, run_dir: Path, solver: str, nlevel: int, tlim: float) -> Path:
    exe = root / "build" / "src" / "athena"
    input_names = {
        "legacy": "rad_crossing_beams.athinput",
        "dyn_cks": "dynrad_crossing_beams.athinput",
        "dyn_adm": "dynrad_crossing_beams_adm.athinput",
    }
    input_file = root / "inputs" / "tests" / input_names[solver]
    basename = f"crossing_{solver}_n{nlevel}"
    block = "radiation" if solver == "legacy" else "dyn_radiation"
    cmd = [
        str(exe),
        "-i",
        str(input_file),
        "-d",
        str(run_dir),
        f"job/basename={basename}",
        f"{block}/nlevel={nlevel}",
        f"time/tlim={tlim}",
        f"output1/dt={tlim}",
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


def line_end(y0: float, x_end: float) -> float:
    qx, qy = target_direction(y0)
    return y0 + (x_end - BEAM["x0"]) * qy / qx


def decorate(ax: plt.Axes) -> None:
    x0 = BEAM["x0"]
    x_end = 1.6
    for y0 in (BEAM["y_lower"], BEAM["y_upper"]):
        ax.plot([x0, x_end], [y0, line_end(y0, x_end)], color="#8bc34a", lw=1.1, alpha=0.85)
        ax.add_patch(plt.Circle((x0, y0), 0.085, edgecolor="#21d4e8",
                                facecolor="none", lw=1.1))
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/dynrad_crossing_beams"))
    parser.add_argument("--fig-dir", type=Path,
                        default=Path("dyngr_radiation_paper/figures"))
    parser.add_argument("--nlevels", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--tlim", type=float, default=0.02)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_dir = args.run_dir
    fig_dir = args.fig_dir if args.fig_dir.is_absolute() else root / args.fig_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fields: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for nlevel in args.nlevels:
        for solver in SOLVERS:
            if args.skip_run:
                basename = f"crossing_{solver}_n{nlevel}"
                matches = sorted(run_dir.glob(f"{basename}.rad_xy.*.bin"))
                matches += sorted((run_dir / "bin").glob(f"{basename}.rad_xy.*.bin"))
                if not matches:
                    raise FileNotFoundError(f"missing output for {basename}")
                path = matches[-1]
            else:
                path = run_case(root, run_dir, solver, nlevel, args.tlim)
            data = read_binary_slice(path, ["r00"])
            field = np.asarray(data.variables["r00"], dtype=float)
            fields[(solver, nlevel)] = (data.x1v, data.x2v, field)

        x, y, legacy = fields[("legacy", nlevel)]
        _, _, dyn_cks = fields[("dyn_cks", nlevel)]
        _, _, dyn_adm = fields[("dyn_adm", nlevel)]
        exact = exact_field(x, y)
        denom = float(np.mean(np.abs(exact)))
        linf_denom = float(np.max(np.abs(exact)))
        row = {
            "nlevel": nlevel,
            "nangles": nangles(nlevel),
            "legacy_L1": float(np.mean(np.abs(legacy - exact)) / denom),
            "dyn_cks_L1": float(np.mean(np.abs(dyn_cks - exact)) / denom),
            "dyn_adm_L1": float(np.mean(np.abs(dyn_adm - exact)) / denom),
            "legacy_Linf": float(np.max(np.abs(legacy - exact)) / linf_denom),
            "dyn_cks_Linf": float(np.max(np.abs(dyn_cks - exact)) / linf_denom),
            "dyn_adm_Linf": float(np.max(np.abs(dyn_adm - exact)) / linf_denom),
            "dyn_cks_minus_legacy_Linf": float(np.max(np.abs(dyn_cks - legacy)) / linf_denom),
            "dyn_adm_minus_dyn_cks_Linf": float(np.max(np.abs(dyn_adm - dyn_cks)) / linf_denom),
        }
        rows.append(row)
        print(
            f"nlevel={nlevel} Nang={row['nangles']} "
            f"legacy_L1={row['legacy_L1']:.6e} "
            f"dyn_cks_L1={row['dyn_cks_L1']:.6e} "
            f"dyn_adm_L1={row['dyn_adm_L1']:.6e} "
            f"|ADM-CKS|_inf={row['dyn_adm_minus_dyn_cks_Linf']:.6e}",
            flush=True,
        )

    csv_path = run_dir / "crossing_beams_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(args.nlevels), 4, figsize=(13.8, 2.3 * len(args.nlevels)),
                             sharex=True, sharey=True, constrained_layout=True)
    if len(args.nlevels) == 1:
        axes = axes[None, :]
    vmax = 0.0
    for nlevel in args.nlevels:
      for solver in SOLVERS:
        vmax = max(vmax, float(np.nanmax(fields[(solver, nlevel)][2])))
      x, y, _ = fields[("legacy", nlevel)]
      vmax = max(vmax, float(np.nanmax(exact_field(x, y))))

    for row_idx, nlevel in enumerate(args.nlevels):
        x, y, legacy = fields[("legacy", nlevel)]
        _, _, dyn_cks = fields[("dyn_cks", nlevel)]
        _, _, dyn_adm = fields[("dyn_adm", nlevel)]
        exact = exact_field(x, y)
        for ax, title, field in zip(
            axes[row_idx],
            ("legacy radiation", "dyn_radiation CKS", "dyn_radiation ADM", "exact"),
            (legacy, dyn_cks, dyn_adm, exact),
        ):
            im = ax.pcolormesh(x, y, field, shading="nearest", vmin=0.0, vmax=vmax,
                               cmap="magma")
            decorate(ax)
            if row_idx == 0:
                ax.set_title(title)
            ax.text(0.97, 0.92, rf"$N_{{ang}}={nangles(nlevel)}$",
                    color="white", ha="right", va="top", transform=ax.transAxes)
            ax.set_aspect("equal")
        axes[row_idx, 0].set_ylabel(r"$y$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$x$")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label(r"$R^{tt}$")
    fig.savefig(fig_dir / "crossing_beams_comparison.png", dpi=190)
    plt.close(fig)

    nvals = np.array([row["nangles"] for row in rows], dtype=float)
    plt.figure(figsize=(5.6, 4.0))
    plt.loglog(nvals, [row["legacy_L1"] for row in rows], "o-", label="legacy radiation")
    plt.loglog(nvals, [row["dyn_cks_L1"] for row in rows], "s-", label="dyn_radiation CKS")
    plt.loglog(nvals, [row["dyn_adm_L1"] for row in rows], "^-", label="dyn_radiation ADM")
    plt.xlabel(r"$N_{ang}$")
    plt.ylabel(r"relative $L_1(R^{tt}-R^{tt}_{exact})$")
    plt.grid(True, which="both", ls=":", lw=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "crossing_beams_convergence.png", dpi=190)
    plt.close()

    print(f"wrote {csv_path}")
    print(f"wrote {fig_dir / 'crossing_beams_comparison.png'}")
    print(f"wrote {fig_dir / 'crossing_beams_convergence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
