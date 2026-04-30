#!/usr/bin/env python3
"""Run and plot the gas-radiation thermal equilibration test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_comparisons import read_binary_slice  # noqa: E402


RHO = 1.0
GAMMA = 5.0 / 3.0
GM1 = GAMMA - 1.0
ARAD = 1.0
TGAS0 = 2.0
TRAD0 = 1.0
KAPPA_A = 1.0
CV = RHO / GM1
UGAS0 = CV * TGAS0
URAD0 = ARAD * TRAD0**4
UTOT = UGAS0 + URAD0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def exact_relaxation(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate dE/dtau = aT^4 - E with conserved gas+radiation energy."""

    def rhs(tgas: float) -> float:
        erad = UTOT - CV * tgas
        return -(ARAD * tgas**4 - erad) / CV

    out = np.empty_like(tau, dtype=float)
    out[0] = TGAS0
    t_prev = float(tau[0])
    y = TGAS0
    for idx in range(1, len(tau)):
        t_next = float(tau[idx])
        nsub = max(1, int(np.ceil((t_next - t_prev) / 1.0e-4)))
        h = (t_next - t_prev) / nsub
        for _ in range(nsub):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * h * k1)
            k3 = rhs(y + 0.5 * h * k2)
            k4 = rhs(y + h * k3)
            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[idx] = y
        t_prev = t_next
    erad = UTOT - CV * out
    trad = np.maximum(erad / ARAD, 0.0) ** 0.25
    return out, trad, CV * out, erad


def equilibrium_temperature() -> float:
    lo, hi = 0.0, max(TGAS0, TRAD0, 1.0)
    while CV * hi + ARAD * hi**4 < UTOT:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if CV * mid + ARAD * mid**4 < UTOT:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cfl_for_steps(nsteps: int) -> float:
    return 0.04 / float(nsteps)


def run_case(root: Path, run_dir: Path, solver: str, nsteps: int) -> list[Path]:
    exe = root / "build" / "src" / "athena"
    input_file = root / "inputs" / "tests" / (
        "rad_equilibration.athinput" if solver == "legacy" else "dynrad_equilibration.athinput"
    )
    basename = f"equil_{solver}_{nsteps:03d}"
    for path in list(run_dir.glob(f"{basename}.relax.*.bin")):
        path.unlink()
    for path in list((run_dir / "bin").glob(f"{basename}.relax.*.bin")):
        path.unlink()
    cmd = [
        str(exe),
        "-i", str(input_file),
        "-d", str(run_dir),
        f"job/basename={basename}",
        f"time/nlim={nsteps}",
        "time/tlim=1.0",
        f"time/cfl_number={cfl_for_steps(nsteps):.16e}",
        f"output1/dt={1.0 / float(nsteps):.16e}",
    ]
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"{basename}.out").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / f"{basename}.err").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{basename} failed with exit code {result.returncode}")
    outputs = sorted(run_dir.glob(f"{basename}.relax.*.bin"))
    outputs += sorted((run_dir / "bin").glob(f"{basename}.relax.*.bin"))
    if not outputs:
        raise FileNotFoundError(f"no relaxation output for {basename}")
    return outputs


def read_series(paths: list[Path]) -> dict[str, np.ndarray]:
    by_time: dict[float, tuple[float, float, float]] = {}
    for path in paths:
        data = read_binary_slice(path, ["r00", "eint"])
        tau = KAPPA_A * RHO * data.time
        erad = float(np.mean(data.variables["r00"]))
        eint = float(np.mean(data.variables["eint"]))
        tgas = GM1 * eint / RHO
        trad = max(erad / ARAD, 0.0) ** 0.25
        by_time[round(tau, 14)] = (tgas, trad, eint, erad)
    times = np.array(sorted(by_time), dtype=float)
    vals = np.array([by_time[float(round(t, 14))] for t in times], dtype=float)
    return {
        "tau": times,
        "tgas": vals[:, 0],
        "trad": vals[:, 1],
        "ugas": vals[:, 2],
        "urad": vals[:, 3],
        "utot": vals[:, 2] + vals[:, 3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/dynrad_equilibration"))
    parser.add_argument("--fig-dir", type=Path,
                        default=Path("dyngr_radiation_paper/figures"))
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_dir = args.run_dir
    fig_dir = args.fig_dir if args.fig_dir.is_absolute() else root / args.fig_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    series: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for solver in ("legacy", "dyn"):
        for nsteps in args.steps:
            basename = f"equil_{solver}_{nsteps:03d}"
            if args.skip_run:
                outputs = sorted(run_dir.glob(f"{basename}.relax.*.bin"))
                outputs += sorted((run_dir / "bin").glob(f"{basename}.relax.*.bin"))
                if not outputs:
                    raise FileNotFoundError(f"missing output for {basename}")
            else:
                outputs = run_case(root, run_dir, solver, nsteps)
            series[(solver, nsteps)] = read_series(outputs)

    tau_exact = np.linspace(0.0, 1.0, 1001)
    tgas_exact, trad_exact, ugas_exact, urad_exact = exact_relaxation(tau_exact)
    teq = equilibrium_temperature()

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), sharex=True,
                             sharey="row", constrained_layout=True)
    colors = {"gas": "#1f77b4", "rad": "#e74c3c", "tot": "#8d5a72"}
    markers = {1: "o", 10: "x", 100: None}
    labels = {"legacy": "legacy radiation", "dyn": "dyn_radiation"}
    for col, solver in enumerate(("legacy", "dyn")):
        ax_t = axes[0, col]
        ax_u = axes[1, col]
        ax_t.plot(tau_exact, tgas_exact, ":", color="black", lw=1.5)
        ax_t.plot(tau_exact, trad_exact, ":", color="black", lw=1.5)
        ax_t.axhline(teq, color="black", ls="--", lw=1.2)
        ax_u.plot(tau_exact, ugas_exact, ":", color="black", lw=1.5)
        ax_u.plot(tau_exact, urad_exact, ":", color="black", lw=1.5)
        ax_u.axhline(UTOT, color="black", ls="--", lw=1.2)
        for nsteps in args.steps:
            data = series[(solver, nsteps)]
            marker = markers.get(nsteps)
            if marker is None:
                ax_t.plot(data["tau"], data["tgas"], color=colors["gas"], lw=2.0)
                ax_t.plot(data["tau"], data["trad"], color=colors["rad"], lw=2.0)
                ax_u.plot(data["tau"], data["ugas"], color=colors["gas"], lw=2.0)
                ax_u.plot(data["tau"], data["urad"], color=colors["rad"], lw=2.0)
                ax_u.plot(data["tau"], data["utot"], color=colors["tot"], lw=2.0)
            else:
                ax_t.plot(data["tau"], data["tgas"], marker, color=colors["gas"], ms=6,
                          mew=1.6, fillstyle="none", ls="None")
                ax_t.plot(data["tau"], data["trad"], marker, color=colors["rad"], ms=6,
                          mew=1.6, fillstyle="none", ls="None")
                ax_u.plot(data["tau"], data["ugas"], marker, color=colors["gas"], ms=6,
                          mew=1.6, fillstyle="none", ls="None")
                ax_u.plot(data["tau"], data["urad"], marker, color=colors["rad"], ms=6,
                          mew=1.6, fillstyle="none", ls="None")
                ax_u.plot(data["tau"], data["utot"], marker, color=colors["tot"], ms=6,
                          mew=1.6, fillstyle="none", ls="None")
        ax_t.set_title(labels[solver])
        ax_t.set_ylim(0.0, 2.5)
        ax_u.set_ylim(0.0, 4.4)
        ax_u.set_xlabel(r"$\bar{\alpha}^a t$")
        ax_t.grid(True, ls=":", lw=0.5, alpha=0.45)
        ax_u.grid(True, ls=":", lw=0.5, alpha=0.45)
    axes[0, 0].set_ylabel(r"$T$")
    axes[1, 0].set_ylabel(r"$u$")

    handles = [
        mlines.Line2D([], [], color=colors["gas"], lw=2.0, label="gas"),
        mlines.Line2D([], [], color=colors["rad"], lw=2.0, label="radiation"),
        mlines.Line2D([], [], color=colors["tot"], lw=2.0, label="total"),
        mlines.Line2D([], [], color="black", ls="--", lw=1.2, label="equilibrium"),
        mlines.Line2D([], [], color="black", ls=":", lw=1.5, label="exact"),
        mlines.Line2D([], [], color="black", marker="o", fillstyle="none", ls="None",
                      label="1 step"),
        mlines.Line2D([], [], color="black", marker="x", ls="None", label="10 steps"),
        mlines.Line2D([], [], color="black", lw=2.0, label="100 steps"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    out = fig_dir / "equilibration_comparison.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)

    dense = series[("dyn", max(args.steps))]
    tgas_ref, trad_ref, ugas_ref, urad_ref = exact_relaxation(dense["tau"])
    err_t = max(float(np.max(np.abs(dense["tgas"] - tgas_ref))),
                float(np.max(np.abs(dense["trad"] - trad_ref))))
    err_e = max(float(np.max(np.abs(dense["ugas"] - ugas_ref))),
                float(np.max(np.abs(dense["urad"] - urad_ref))))
    print(f"dyn {max(args.steps)}-step max |Delta T|={err_t:.6e} "
          f"max |Delta u|={err_e:.6e}")
    for solver in ("legacy", "dyn"):
        dense = series[(solver, max(args.steps))]
        print(f"{solver} final tau={dense['tau'][-1]:.6f} "
              f"Tgas={dense['tgas'][-1]:.6f} Trad={dense['trad'][-1]:.6f} "
              f"utot={dense['utot'][-1]:.6f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
