#!/usr/bin/env python3
"""Run and plot ADM-specific dyn_radiation formal regression tests."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_comparisons import read_binary_slice  # noqa: E402


RESULT_RE = re.compile(r"ADM_FORMAL_TEST\s+(\S+)\s+(.*)")
PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_results(text: str) -> list[tuple[str, dict[str, float]]]:
    rows: list[tuple[str, dict[str, float]]] = []
    for line in text.splitlines():
        match = RESULT_RE.search(line)
        if match is None:
            continue
        values = {key: float(value) for key, value in PAIR_RE.findall(match.group(2))}
        rows.append((match.group(1), values))
    return rows


def run_athena(root: Path, run_dir: Path, case: str, input_file: str,
               *overrides: str) -> list[tuple[str, dict[str, float]]]:
    athena = root / "build" / "src" / "athena"
    case_dir = run_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    cmd = [
        str(athena),
        "-i",
        str(root / input_file),
        "-d",
        str(case_dir),
        f"job/basename={case}",
        *overrides,
    ]
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_btl", "self")
    env.setdefault("OMPI_MCA_btl_base_warn_component_unused", "0")
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, env=env)
    (run_dir / "logs" / f"{case}.out").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / f"{case}.err").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{case} failed with exit code {result.returncode}")
    rows = parse_results(result.stdout + "\n" + result.stderr)
    if not rows:
        raise RuntimeError(f"{case} did not print ADM_FORMAL_TEST diagnostics")
    return rows


def latest_bin(case_dir: Path, basename: str) -> Path:
    matches = sorted(case_dir.glob(f"{basename}.rad_xy.*.bin"))
    matches += sorted((case_dir / "bin").glob(f"{basename}.rad_xy.*.bin"))
    if not matches:
        raise FileNotFoundError(f"no rad_xy binary output for {basename}")
    return matches[-1]


def all_bins(case_dir: Path, basename: str) -> list[Path]:
    matches = sorted(case_dir.glob(f"{basename}.rad_xy.*.bin"))
    matches += sorted((case_dir / "bin").glob(f"{basename}.rad_xy.*.bin"))
    if not matches:
        raise FileNotFoundError(f"no rad_xy binary output for {basename}")
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/dynrad_adm_formal"))
    parser.add_argument("--fig-dir", type=Path,
                        default=Path("dyngr_radiation_paper/figures"))
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_dir = args.run_dir
    fig_dir = args.fig_dir if args.fig_dir.is_absolute() else root / args.fig_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, float | str]] = []
    if not args.skip_run:
        for tlim in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            label = f"t{tlim:.2f}"
            case = "adm_flrw_" + label.replace(".", "p")
            overrides = [
                f"time/tlim={tlim}",
                f"output1/dt={max(tlim, 1.0)}",
            ]
            if tlim == 0.0:
                overrides.append("time/nlim=0")
            for name, values in run_athena(
                root, run_dir, case,
                "inputs/tests/dynrad_flrw_redshift.athinput",
                *overrides,
            ):
                result_rows.append({"case": name, "label": label, **values})

        for name, values in run_athena(
            root, run_dir, "adm_lapse_gradient",
            "inputs/tests/dynrad_lapse_gradient.athinput",
        ):
            result_rows.append({"case": name, "label": "final", **values})

        for nx in (32, 64, 128):
            overrides = [
                f"mesh/nx1={nx}",
                f"meshblock/nx1={nx // 2}",
                "problem/momentum_abs_tolerance=1.0",
                "problem/momentum_rel_tolerance=1.0",
            ]
            for name, values in run_athena(
                root, run_dir, f"adm_momentum_n{nx}",
                "inputs/tests/dynrad_momentum_source.athinput",
                *overrides,
            ):
                result_rows.append({"case": name, "label": f"nx{nx}", **values})
    else:
        for path in sorted((run_dir / "logs").glob("*.out")):
            for name, values in parse_results(path.read_text(encoding="utf-8")):
                if path.stem.startswith("adm_flrw_t"):
                    label = path.stem.removeprefix("adm_flrw_").replace("p", ".")
                else:
                    label = path.stem.split("_n")[-1]
                if path.stem.endswith(("n32", "n64", "n128")):
                    label = "nx" + label
                elif not path.stem.startswith("adm_flrw_t"):
                    label = "final"
                result_rows.append({"case": name, "label": label, **values})

    csv_path = run_dir / "adm_formal_tests.csv"
    fieldnames = sorted({key for row in result_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    flrw = [row for row in result_rows if row["case"] == "flrw"]
    flrw.sort(key=lambda row: float(row["label"][1:]))
    flrw_times_arr = np.asarray([float(row["label"][1:]) for row in flrw])
    flrw_e_arr = np.asarray([float(row["mean_E"]) for row in flrw])
    flrw_exact = np.asarray([float(row["exact_E"]) for row in flrw])

    lapse_data = read_binary_slice(latest_bin(run_dir / "adm_lapse_gradient",
                                              "adm_lapse_gradient"), ["r00"])
    x = lapse_data.x1v
    alpha = 1.0 + 0.1*np.sin(2.0*np.pi*x)
    e_lapse = np.mean(np.asarray(lapse_data.variables["r00"]) * alpha[None, :]**2, axis=0)
    expected_lapse = 1.0 - 2.0*1.0*0.7*(0.1*2.0*np.pi*np.cos(2.0*np.pi*x))*lapse_data.time

    momentum = [
        row for row in result_rows
        if row["case"] == "momentum_source" and str(row["label"]).startswith("nx")
    ]
    momentum.sort(key=lambda row: int(str(row["label"])[2:]))
    nx_values = np.asarray([int(str(row["label"])[2:]) for row in momentum], dtype=float)
    abs_values = np.asarray([float(row["max_abs"]) for row in momentum])
    rel_values = np.asarray([float(row["max_rel"]) for row in momentum])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7), constrained_layout=True)

    ax = axes[0]
    ax.plot(flrw_times_arr, flrw_exact, "k-", lw=1.5, label=r"$a^{-4}$")
    ax.plot(flrw_times_arr, flrw_e_arr, "o", ms=4.5, label="dyn ADM")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$E$")
    ax.set_title("FLRW redshift")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, expected_lapse - 1.0, "k-", lw=1.5, label="linear source")
    ax.plot(x, e_lapse - 1.0, "o", ms=3.5, label="dyn ADM")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$E-E_0$")
    ax.set_title("static lapse gradient")
    ax.legend(frameon=False)

    ax = axes[2]
    ax.loglog(nx_values, abs_values, "o-", label=r"$L_\infty$ absolute")
    if len(nx_values) >= 2:
        guide_x = np.asarray([nx_values[0], nx_values[-1]])
        guide = abs_values[0]*(guide_x/guide_x[0])**-2
        ax.loglog(guide_x, guide, "k:", label=r"$N^{-2}$")
    ax.set_xlabel(r"$N_x$")
    ax.set_ylabel("closure residual")
    ax.set_title("momentum-source closure")
    ax.text(0.05, 0.06, rf"$\max$ relative at $N_x=128$: {rel_values[-1]:.2e}",
            transform=ax.transAxes, ha="left", va="bottom")
    ax.legend(frameon=False)

    fig_path = fig_dir / "adm_formal_tests.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"wrote {fig_path}")
    print(f"wrote {csv_path}")
    for row in result_rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
