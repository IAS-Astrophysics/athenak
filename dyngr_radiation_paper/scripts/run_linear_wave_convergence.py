#!/usr/bin/env python3
"""Run and plot radiation linear-wave convergence for legacy, CKS, and ADM paths."""

from __future__ import annotations

import argparse
import csv
import math
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


GAMMA = 5.0 / 3.0
GM1 = GAMMA - 1.0
KPAR = 2.0 * math.pi
TLIM_DAMPING = 0.05

EIG = {
    "rho": 1.0,
    "pgas": 2.497687326549491e-01,
    "ux": 0.0,
    "erad": 7.493061979648474e-02,
    "fxrad": 0.0,
    "delta": 1.0e-4,
    "omega_real": 3.1488157526582414e+00,
    "omega_imag": -2.6190006385782953e-02,
    "drho_real": 8.3877889167048014e-01,
    "drho_imag": 0.0,
    "dpgas_real": 3.2084488925731219e-01,
    "dpgas_imag": -9.9134535607493107e-03,
    "dux_real": 4.2035369927276667e-01,
    "dux_imag": -3.4962560317943620e-03,
    "derad_real": 1.2904189937790903e-01,
    "derad_imag": 1.5203926879094193e-03,
    "dfxrad_real": 1.3260665610966586e-03,
    "dfxrad_imag": -6.7017329068802516e-03,
}

VARIABLES = ["dens", "velx", "eint", "r00", "r01"]
ADM_VARIABLES = ["dens", "velx", "press", "r00", "r01"]
COMPONENT_LABELS = {
    "dens": r"$\rho$",
    "velx": r"$u^x$",
    "pgas": r"$P_{\rm gas}$",
    "r00": r"$R^{tt}$",
    "r01": r"$R^{tx}$",
}

SOLVERS = ("legacy", "dyn_cks", "dyn_adm")
SOLVER_LABELS = {
    "legacy": "legacy radiation",
    "dyn_cks": "dyn_radiation CKS",
    "dyn_adm": "dyn_radiation ADM",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def final_time() -> float:
    return TLIM_DAMPING * math.log(2.0) / abs(EIG["omega_imag"])


def perturb(real: str, imag: str, cn: np.ndarray, sn: np.ndarray,
            damping: float) -> np.ndarray:
    return EIG["delta"] * damping * (EIG[real] * cn + EIG[imag] * sn)


def exact_solution(x: np.ndarray, time: float) -> dict[str, np.ndarray]:
    phase = EIG["omega_real"] * time - KPAR * x
    cn = np.cos(phase)
    sn = np.sin(phase)
    damping = math.exp(EIG["omega_imag"] * time)

    rho = EIG["rho"] + perturb("drho_real", "drho_imag", cn, sn, damping)
    pgas = EIG["pgas"] + perturb("dpgas_real", "dpgas_imag", cn, sn, damping)
    ux = EIG["ux"] + perturb("dux_real", "dux_imag", cn, sn, damping)
    erad_ff = EIG["erad"] + perturb("derad_real", "derad_imag", cn, sn, damping)
    fxrad_ff = EIG["fxrad"] + perturb("dfxrad_real", "dfxrad_imag", cn, sn, damping)

    u0 = np.sqrt(1.0 + ux * ux)
    boost11 = 1.0 + ux * ux / (1.0 + u0)
    prad_ff = erad_ff / 3.0
    r00 = u0 * u0 * erad_ff + 2.0 * u0 * ux * fxrad_ff + ux * ux * prad_ff
    r01 = (u0 * ux * erad_ff +
           (u0 * boost11 + ux * ux) * fxrad_ff +
           ux * boost11 * prad_ff)

    return {
        "dens": rho,
        "velx": ux,
        "pgas": pgas,
        "eint": pgas / GM1,
        "r00": r00,
        "r01": r01,
    }


def component_scale(exact: dict[str, np.ndarray], key: str) -> float:
    background = {
        "dens": EIG["rho"],
        "velx": EIG["ux"],
        "pgas": EIG["pgas"],
        "eint": EIG["pgas"] / GM1,
        "r00": EIG["erad"],
        "r01": EIG["fxrad"],
    }[key]
    return max(float(np.max(np.abs(exact[key] - background))), 1.0e-300)


def run_case(root: Path, run_dir: Path, solver: str, nx: int) -> Path:
    exe = root / "build" / "src" / "athena"
    input_names = {
        "legacy": "rad_lwave_convergence.athinput",
        "dyn_cks": "dynrad_lwave_convergence.athinput",
        "dyn_adm": "dynrad_lwave_adm_convergence.athinput",
    }
    input_file = root / "inputs" / "tests" / input_names[solver]
    basename = f"lwave_{solver}_{nx:04d}"
    for path in list(run_dir.glob(f"{basename}*")):
        if path.is_file():
            path.unlink()
    for path in list((run_dir / "bin").glob(f"{basename}*")):
        if path.is_file():
            path.unlink()
    mb_nx = min(max(8, nx // 2), nx)
    cmd = [
        str(exe),
        "-i", str(input_file),
        "-d", str(run_dir),
        f"job/basename={basename}",
        f"mesh/nx1={nx}",
        f"meshblock/nx1={mb_nx}",
        f"time/tlim={TLIM_DAMPING:.16e}",
        f"output1/dt={final_time():.16e}",
    ]
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_btl", "self")
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, env=env)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"{basename}.out").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / f"{basename}.err").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{basename} failed with exit code {result.returncode}")
    outputs = sorted(run_dir.glob(f"{basename}.lwave.*.bin"))
    outputs += sorted((run_dir / "bin").glob(f"{basename}.lwave.*.bin"))
    if not outputs:
        raise FileNotFoundError(f"no final binary output for {basename}")
    return outputs[-1]


def read_case(path: Path, solver: str) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    variables = ADM_VARIABLES if solver == "dyn_adm" else VARIABLES
    data = read_binary_slice(path, variables)
    values = {name: np.ravel(np.asarray(data.variables[name], dtype=float))
              for name in variables}
    if "press" in values:
        values["pgas"] = values["press"]
        values["eint"] = values["pgas"] / GM1
    else:
        values["pgas"] = GM1 * values["eint"]
    x = np.ravel(np.asarray(data.x1v, dtype=float))
    return data.time, x, values


def measure_errors(path: Path, solver: str) -> tuple[float, dict[str, float]]:
    time, x, values = read_case(path, solver)
    if x.size != values["dens"].size:
        x = np.resize(x, values["dens"].size)
    exact = exact_solution(x, time)
    errors: dict[str, float] = {}
    for key in ("dens", "velx", "pgas", "r00", "r01"):
        scale = component_scale(exact, key)
        errors[key] = float(np.mean(np.abs(values[key] - exact[key])) / scale)
    errors["rms"] = float(math.sqrt(sum(v * v for v in errors.values()) / len(errors)))
    errors["time"] = time
    return time, errors


def add_order_guides(ax: plt.Axes, nvals: np.ndarray, anchor: float) -> None:
    x1 = float(nvals[1])
    ref1 = anchor * (nvals / x1) ** -1
    ref2 = anchor * 0.55 * (nvals / x1) ** -2
    ax.loglog(nvals, ref1, "--", color="0.45", lw=1.0, label="1st-order")
    ax.loglog(nvals, ref2, ":", color="0.45", lw=1.2, label="2nd-order")


def convergence_rate(nvals: list[int], errors: list[float]) -> float:
    x = np.log(np.asarray(nvals[-3:], dtype=float))
    y = np.log(np.asarray(errors[-3:], dtype=float))
    return float(-np.polyfit(x, y, 1)[0])


def plot(rows: list[dict[str, float]], fig_dir: Path) -> Path:
    colors = {"legacy": "#6b78d6", "dyn_cks": "#e74c3c", "dyn_adm": "#2ca02c"}
    nvals = sorted({int(row["nx"]) for row in rows})

    fig = plt.figure(figsize=(11.2, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.16, 1.0])
    ax_main = fig.add_subplot(gs[:, 0])
    detail_axes = {
        "legacy": fig.add_subplot(gs[0, 1]),
        "dyn_cks": fig.add_subplot(gs[1, 1]),
        "dyn_adm": fig.add_subplot(gs[2, 1]),
    }

    by_solver = {
        solver: [row for row in rows if row["solver"] == solver]
        for solver in SOLVERS
    }
    markers = {"legacy": "o", "dyn_cks": "s", "dyn_adm": "^"}
    fillstyles = {"legacy": "none", "dyn_cks": "full", "dyn_adm": "none"}
    sizes = {"legacy": 7, "dyn_cks": 4.5, "dyn_adm": 6.0}
    for solver in SOLVERS:
        solver_rows = sorted(by_solver[solver], key=lambda row: row["nx"])
        xs = np.asarray([row["nx"] for row in solver_rows], dtype=float)
        ys = np.asarray([row["rms"] for row in solver_rows], dtype=float)
        ax_main.loglog(xs, ys, marker=markers[solver], ls="-", color=colors[solver],
                       lw=1.55, ms=sizes[solver], fillstyle=fillstyles[solver],
                       mew=1.35, label=SOLVER_LABELS[solver])
    anchor = by_solver["dyn_adm"][1]["rms"] if len(by_solver["dyn_adm"]) > 1 else by_solver["dyn_adm"][0]["rms"]
    add_order_guides(ax_main, np.asarray(nvals, dtype=float), anchor)
    ax_main.set_xlabel(r"$N_{\rm cell}$")
    ax_main.set_ylabel(r"normalized $L_1$ error")
    ax_main.set_title("radiation-fluid fast wave")
    ax_main.grid(True, which="both", ls=":", lw=0.55, alpha=0.7)
    ax_main.legend(frameon=True)

    component_colors = {
        "dens": "#1f77b4",
        "velx": "#2ca02c",
        "pgas": "#9467bd",
        "r00": "#ff7f0e",
        "r01": "#8c564b",
    }
    first_ax = detail_axes["legacy"]
    for ax in detail_axes.values():
        if ax is not first_ax:
            ax.sharex(first_ax)
            ax.sharey(first_ax)
    for solver, ax in detail_axes.items():
        solver_rows = sorted(by_solver[solver], key=lambda row: row["nx"])
        xs = np.asarray([row["nx"] for row in solver_rows], dtype=float)
        for key in ("dens", "velx", "pgas", "r00", "r01"):
            ax.loglog(xs, [row[key] for row in solver_rows], "o-",
                      color=component_colors[key], lw=1.35, ms=4,
                      label=COMPONENT_LABELS[key])
        ax.set_title(SOLVER_LABELS[solver])
        ax.grid(True, which="both", ls=":", lw=0.55, alpha=0.7)
        ax.set_ylabel(r"normalized $L_1$")
    detail_axes["dyn_adm"].set_xlabel(r"$N_{\rm cell}$")
    detail_axes["legacy"].legend(frameon=True, ncol=2, fontsize=8)
    detail_axes["dyn_adm"].legend(handles=[
        mlines.Line2D([], [], color="0.45", ls="--", lw=1.0, label="1st-order"),
        mlines.Line2D([], [], color="0.45", ls=":", lw=1.2, label="2nd-order"),
    ], frameon=True, loc="lower left", fontsize=8)
    add_order_guides(detail_axes["dyn_adm"], np.asarray(nvals, dtype=float),
                     by_solver["dyn_adm"][1]["dens"])

    out = fig_dir / "linear_wave_convergence.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/dynrad_linear_wave"))
    parser.add_argument("--fig-dir", type=Path,
                        default=Path("dyngr_radiation_paper/figures"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_dir = args.run_dir
    fig_dir = args.fig_dir if args.fig_dir.is_absolute() else root / args.fig_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    for solver in SOLVERS:
        for nx in args.resolutions:
            basename = f"lwave_{solver}_{nx:04d}"
            if args.skip_run:
                outputs = sorted(run_dir.glob(f"{basename}.lwave.*.bin"))
                outputs += sorted((run_dir / "bin").glob(f"{basename}.lwave.*.bin"))
                if not outputs:
                    raise FileNotFoundError(f"missing output for {basename}")
                path = outputs[-1]
            else:
                path = run_case(root, run_dir, solver, nx)
            time, errors = measure_errors(path, solver)
            row = {"solver": solver, "nx": float(nx), **errors}
            rows.append(row)
            print(f"{solver:6s} nx={nx:4d} t={time:.6e} "
                  f"rms={errors['rms']:.6e} r00={errors['r00']:.6e} "
                  f"r01={errors['r01']:.6e}", flush=True)

    csv_path = run_dir / "linear_wave_convergence_errors.csv"
    fieldnames = ["solver", "nx", "time", "rms", "dens", "velx", "pgas", "r00", "r01"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out = plot(rows, fig_dir)
    for solver in SOLVERS:
        solver_rows = sorted([row for row in rows if row["solver"] == solver],
                             key=lambda row: row["nx"])
        rates = {
            key: convergence_rate([int(row["nx"]) for row in solver_rows],
                                  [row[key] for row in solver_rows])
            for key in ("rms", "dens", "velx", "pgas", "r00", "r01")
        }
        print(f"{solver} last-three convergence rates: "
              + " ".join(f"{key}={value:.3f}" for key, value in rates.items()))
    print(f"wrote {csv_path}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
