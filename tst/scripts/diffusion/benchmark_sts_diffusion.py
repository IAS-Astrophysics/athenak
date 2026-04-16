#!/usr/bin/env python3
"""
Run a small CPU benchmark package for the exact STS diffusion problems.

Outputs:
  - doc/data/sts_diffusion/raw_runs.csv
  - doc/data/sts_diffusion/summary.csv
  - doc/data/sts_diffusion/profile_samples.csv
  - doc/data/sts_diffusion/environment.json
  - doc/data/sts_diffusion/setup_table.tex
  - doc/data/sts_diffusion/highres_table.tex
  - doc/data/sts_diffusion/results_findings.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VIS_PYTHON = REPO_ROOT / "vis" / "python"
if str(VIS_PYTHON) not in sys.path:
    sys.path.insert(0, str(VIS_PYTHON))

import athena_read  # noqa: E402


BUILD_DIR = REPO_ROOT / "tst" / "build"
BUILD_SRC = BUILD_DIR / "src"
ATHENA_BIN = BUILD_SRC / "athena"
INPUT_LINK = BUILD_SRC / "inputs"
BUILD_DEPENDENCIES = [
    REPO_ROOT / "src",
    REPO_ROOT / "inputs" / "tests",
    REPO_ROOT / "tst" / "scripts" / "diffusion",
    REPO_ROOT / "vis" / "python" / "plot_sts_diffusion_benchmark.py",
]

DATA_DIR = REPO_ROOT / "doc" / "data" / "sts_diffusion"
RAW_CSV = DATA_DIR / "raw_runs.csv"
SUMMARY_CSV = DATA_DIR / "summary.csv"
PROFILE_CSV = DATA_DIR / "profile_samples.csv"
ENV_JSON = DATA_DIR / "environment.json"
SETUP_TABLE_TEX = DATA_DIR / "setup_table.tex"
HIGHRES_TABLE_TEX = DATA_DIR / "highres_table.tex"
RESULTS_FINDINGS_TEX = DATA_DIR / "results_findings.tex"

RESOLUTIONS = [64, 128, 256]
WARMUP_RUNS = 1
MEASURED_RUNS = 5
PROFILE_RESOLUTION = 256
THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "KOKKOS_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]

DT_RE = re.compile(r"\bdt\s*=\s*([0-9.eE+-]+)")
FINAL_RE = re.compile(r"time=\s*([0-9.eE+-]+)\s+cycle=\s*(\d+)")
CPU_TIME_RE = re.compile(r"cpu time used\s*=\s*([0-9.eE+-]+)")
ZCPS_RE = re.compile(r"zone-cycles/cpu_second\s*=\s*([0-9.eE+-]+)")


@dataclass(frozen=True)
class CaseConfig:
    name: str
    label: str
    input_file: Path
    selector_flag: str
    profile_variable: str
    profile_key: str
    profile_ylabel: str
    pde_description: str
    eos_note: str
    coefficient_name: str
    coefficient_value: float
    sts_cap: float
    diffusivity: float
    amplitude: float = 1.0e-6
    density: float = 1.0
    gamma: float = 1.66667
    t0: float = 0.5
    tlim: float = 4.0
    x10: float = 0.0
    x1min: float = -5.0
    x1max: float = 5.0

    def method_flag(self, method: str) -> str:
        return f"{self.selector_flag}={'explicit' if method == 'explicit' else 'sts'}"

    @property
    def solution_time(self) -> float:
        return self.t0 + self.tlim

    @property
    def profile_background(self) -> float:
        if self.name == "hydro_conduction":
            p0 = 1.0 / self.gamma
            return p0 / (self.gamma - 1.0)
        return 0.0

    def gaussian_kernel(self, x1: float) -> float:
        return math.exp((x1 - self.x10) ** 2 / (-4.0 * self.diffusivity * self.solution_time)) / (
            math.sqrt(4.0 * math.pi * self.diffusivity * self.solution_time)
        )

    def analytic_profile(self, x1: float) -> float:
        return self.amplitude * self.gaussian_kernel(x1)


CASES = [
    CaseConfig(
        name="hydro_viscosity",
        label="Hydro viscosity",
        input_file=REPO_ROOT / "inputs" / "tests" / "viscosity.athinput",
        selector_flag="hydro/viscosity_integrator",
        profile_variable="hydro_w_vy",
        profile_key="vely",
        profile_ylabel="v_y",
        pde_description="viscous diffusion of transverse velocity",
        eos_note="Hydro ideal EOS",
        coefficient_name="nu",
        coefficient_value=1.0,
        sts_cap=-1.0,
        diffusivity=1.0,
    ),
    CaseConfig(
        name="hydro_conduction",
        label="Hydro conduction",
        input_file=REPO_ROOT / "inputs" / "tests" / "sts_conduction.athinput",
        selector_flag="hydro/conductivity_integrator",
        profile_variable="hydro_w_e",
        profile_key="eint",
        profile_ylabel="delta e_int",
        pde_description="thermal diffusion of internal-energy perturbation",
        eos_note="Hydro ideal EOS",
        coefficient_name="kappa",
        coefficient_value=1.0,
        sts_cap=4.0,
        diffusivity=1.0 * (1.66667 - 1.0) / 1.0,
    ),
    CaseConfig(
        name="mhd_resistivity",
        label="MHD resistivity",
        input_file=REPO_ROOT / "inputs" / "tests" / "sts_resistivity.athinput",
        selector_flag="mhd/ohmic_resistivity_integrator",
        profile_variable="mhd_bcc2",
        profile_key="bcc2",
        profile_ylabel="B_y",
        pde_description="Ohmic diffusion of transverse magnetic field",
        eos_note="MHD isothermal EOS",
        coefficient_name="eta",
        coefficient_value=1.0,
        sts_cap=8.0,
        diffusivity=1.0,
    ),
]


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed in {cwd}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    if capture_output:
        return result.stdout + result.stderr
    return ""


def fixed_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in THREAD_ENV_KEYS:
        env[key] = "1"
    return env


def cpu_model() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file():
                newest = max(newest, candidate.stat().st_mtime)
    return newest


def build_is_stale() -> bool:
    if not ATHENA_BIN.exists():
        return True
    build_time = ATHENA_BIN.stat().st_mtime
    return any(newest_mtime(path) > build_time for path in BUILD_DEPENDENCIES)


def ensure_build(rebuild: bool, build_threads: int) -> bool:
    rebuilt = False
    if rebuild and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    if not ATHENA_BIN.exists() or build_is_stale():
        run_command(["cmake", "-B", "tst/build"], cwd=REPO_ROOT)
        BUILD_SRC.mkdir(parents=True, exist_ok=True)
        run_command(["make", "-j", str(build_threads)], cwd=BUILD_SRC)
        rebuilt = True

    if INPUT_LINK.exists() or INPUT_LINK.is_symlink():
        if INPUT_LINK.is_symlink() and INPUT_LINK.resolve() == (REPO_ROOT / "inputs").resolve():
            return rebuilt
        if INPUT_LINK.is_dir() and not INPUT_LINK.is_symlink():
            raise RuntimeError(f"{INPUT_LINK} exists and is not the expected symlink")
        INPUT_LINK.unlink()
    INPUT_LINK.symlink_to(Path("../../inputs"))
    return rebuilt


def clear_output_tree() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in DATA_DIR.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()


def clear_benchmark_run_files() -> None:
    if BUILD_SRC.exists():
        for path in BUILD_SRC.glob("bench_*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
        tab_dir = BUILD_SRC / "tab"
        if tab_dir.exists():
            for path in tab_dir.glob("bench_*.tab"):
                if path.is_file() or path.is_symlink():
                    path.unlink()


def parse_output_metrics(output: str) -> dict[str, float]:
    dt_match = DT_RE.search(output)
    final_match = FINAL_RE.search(output)
    cpu_time_match = CPU_TIME_RE.search(output)
    zcps_match = ZCPS_RE.search(output)
    if dt_match is None or final_match is None or cpu_time_match is None or zcps_match is None:
        raise RuntimeError("Could not parse AthenaK stdout for benchmark metrics")
    return {
        "first_dt": float(dt_match.group(1)),
        "final_time": float(final_match.group(1)),
        "final_cycle": int(final_match.group(2)),
        "cpu_time_used": float(cpu_time_match.group(1)),
        "zonecycles_per_cpu_second": float(zcps_match.group(1)),
    }


def unique_basename(case: CaseConfig, method: str, resolution: int, run_kind: str, repeat: int) -> str:
    return f"bench_{case.name}_{method}_n{resolution}_{run_kind}{repeat}"


def base_flags(case: CaseConfig, method: str, resolution: int, basename: str) -> list[str]:
    flags = [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}",
        "mesh/nx2=1",
        "mesh/nx3=1",
        f"meshblock/nx1={resolution // 4}",
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        f"time/sts_integrator={'none' if method == 'explicit' else 'rkl2'}",
        case.method_flag(method),
    ]
    if method == "sts":
        flags.append(f"time/sts_max_dt_ratio={case.sts_cap:g}")
    else:
        flags.append("time/sts_max_dt_ratio=-1.0")
    return flags


def profile_flags(case: CaseConfig, method: str, resolution: int, basename: str) -> list[str]:
    return base_flags(case, method, resolution, basename)


def load_error_row(basename: str) -> dict[str, float]:
    data = athena_read.error_dat(str(BUILD_SRC / f"{basename}-errs.dat"))
    row = data[-1]
    return {
        "nx1": int(row[0]),
        "nx2": int(row[1]),
        "nx3": int(row[2]),
        "ncycle": int(row[3]),
        "rms_l1": float(row[4]),
        "linfty": float(row[5]),
    }


def load_profile_rows(case: CaseConfig, method: str, resolution: int, basename: str) -> list[dict[str, float | str | int]]:
    tab_files = sorted((BUILD_SRC / "tab").glob(f"{basename}*.tab"))
    if not tab_files:
        raise RuntimeError(f"No tab output found for profile capture basename {basename}")
    data = athena_read.tab(str(tab_files[-1]))
    if "x1v" not in data or case.profile_key not in data:
        raise RuntimeError(f"Tab output for {basename} is missing x1v/{case.profile_key}")

    rows: list[dict[str, float | str | int]] = []
    xvals = data["x1v"]
    raw_vals = data[case.profile_key]
    for x1, raw_value in zip(xvals, raw_vals):
        analytic = case.analytic_profile(float(x1))
        plotted = float(raw_value) - case.profile_background
        rows.append(
            {
                "case": case.name,
                "method": method,
                "resolution": resolution,
                "x1": float(x1),
                "raw_value": float(raw_value),
                "plotted_value": plotted,
                "analytic_value": analytic,
            }
        )
    return rows


def make_profile_input(case: CaseConfig) -> Path:
    text = case.input_file.read_text()
    output_blocks = [int(match.group(1)) for match in re.finditer(r"<output(\d+)>", text)]
    next_output = max(output_blocks) + 1 if output_blocks else 1
    block = "\n".join(
        [
            f"<output{next_output}>",
            "file_type   = tab",
            f"variable    = {case.profile_variable}",
            "data_format = %24.16e",
            f"dt          = {case.tlim:g}",
            "slice_x2    = 0.0",
            "slice_x3    = 0.0",
            "",
        ]
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{case.name}_profile.athinput",
        dir=str(BUILD_SRC),
        delete=False,
    )
    with handle:
        handle.write(text.rstrip() + "\n\n" + block)
    return Path(handle.name)


def remove_outputs(basename: str) -> None:
    for path in BUILD_SRC.glob(f"{basename}*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
    tab_dir = BUILD_SRC / "tab"
    if tab_dir.exists():
        for path in tab_dir.glob(f"{basename}*.tab"):
            if path.is_file() or path.is_symlink():
                path.unlink()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def fmt_decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def summarize_rows(raw_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in raw_rows:
        key = (row["case"], row["method"], int(row["resolution"]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for key in sorted(groups):
        grouped = groups[key]
        wall = [float(r["wall_seconds"]) for r in grouped]
        cpu = [float(r["cpu_time_used"]) for r in grouped]
        zcps = [float(r["zonecycles_per_cpu_second"]) for r in grouped]
        errs = [float(r["rms_l1"]) for r in grouped]
        linf = [float(r["linfty"]) for r in grouped]
        dt0 = [float(r["first_dt"]) for r in grouped]
        cycles = [int(r["final_cycle"]) for r in grouped]
        summary_rows.append(
            {
                "case": key[0],
                "method": key[1],
                "resolution": key[2],
                "samples": len(grouped),
                "wall_seconds_median": statistics.median(wall),
                "wall_seconds_min": min(wall),
                "wall_seconds_max": max(wall),
                "wall_seconds_std": statistics.stdev(wall) if len(wall) > 1 else 0.0,
                "cpu_time_used_median": statistics.median(cpu),
                "zonecycles_per_cpu_second_median": statistics.median(zcps),
                "rms_l1_median": statistics.median(errs),
                "linfty_median": statistics.median(linf),
                "first_dt_median": statistics.median(dt0),
                "final_cycle_median": statistics.median(cycles),
            }
        )
    return summary_rows


def assert_regression_sanity(summary_rows: list[dict]) -> None:
    thresholds = {
        ("hydro_viscosity", "sts"): {"pair": (64, 128), "max_error": 3.5e-11, "max_ratio": 0.60},
        ("hydro_conduction", "explicit"): {
            "pair": (128, 256),
            "max_error": 4.0e-12,
            "max_ratio": 0.30,
        },
        ("hydro_conduction", "sts"): {
            "pair": (128, 256),
            "max_error": 3.0e-12,
            "max_ratio": 0.30,
        },
        ("mhd_resistivity", "sts"): {"pair": (64, 128), "max_error": 1.2e-11, "max_ratio": 0.32},
    }
    for (case_name, method), config in thresholds.items():
        low_res, high_res = config["pair"]
        rows = {
            int(row["resolution"]): row
            for row in summary_rows
            if row["case"] == case_name and row["method"] == method
        }
        low = rows[low_res]["rms_l1_median"]
        high = rows[high_res]["rms_l1_median"]
        if high > config["max_error"]:
            raise RuntimeError(
                f"{case_name} {method} sanity check failed: {high:g} > {config['max_error']:g}"
            )
        if (high / low) > config["max_ratio"]:
            raise RuntimeError(
                f"{case_name} {method} convergence ratio failed: {high/low:g} > "
                f"{config['max_ratio']:g}"
            )


def profile_center_row(
    profile_rows: list[dict[str, float | str | int]], case_name: str, method: str
) -> dict[str, float | str | int]:
    rows = [
        row
        for row in profile_rows
        if row["case"] == case_name
        and row["method"] == method
        and int(row["resolution"]) == PROFILE_RESOLUTION
    ]
    if not rows:
        raise RuntimeError(
            f"Missing profile rows for {case_name} {method} at resolution {PROFILE_RESOLUTION}"
        )
    return min(rows, key=lambda row: abs(float(row["x1"])))


def assert_profile_sanity(profile_rows: list[dict[str, float | str | int]]) -> None:
    for method in ("explicit", "sts"):
        row = profile_center_row(profile_rows, "hydro_conduction", method)
        analytic = float(row["analytic_value"])
        plotted = float(row["plotted_value"])
        rel_error = abs(plotted - analytic) / max(abs(analytic), 1.0e-30)
        if rel_error > 1.0e-3:
            raise RuntimeError(
                "Hydro conduction profile-center sanity check failed for "
                f"{method}: relative error {rel_error:g} > 1e-3"
            )


def write_environment_metadata(rebuilt: bool, build_threads: int) -> None:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()
    git_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        ).strip()
    )
    if git_dirty:
        git_commit = f"{git_commit}-dirty"

    metadata = {
        "git_commit": git_commit,
        "git_worktree_dirty": git_dirty,
        "hostname": platform.node(),
        "cpu_model": cpu_model(),
        "python_version": sys.version.split()[0],
        "benchmark_mode": "CPU release build via tst/build",
        "build_rebuilt": rebuilt,
        "build_threads": build_threads,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "resolutions": RESOLUTIONS,
        "thread_env": {key: "1" for key in THREAD_ENV_KEYS},
    }
    ENV_JSON.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_setup_table() -> None:
    coeff_tex = {
        "nu": r"\nu",
        "kappa": r"\kappa",
        "eta": r"\eta",
    }
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Case & PDE & EOS & Coefficient & STS cap & Resolutions \\",
        r"\midrule",
    ]
    for case in CASES:
        cap = "none" if case.sts_cap < 0.0 else fmt_decimal(case.sts_cap, 0)
        coeff = coeff_tex.get(case.coefficient_name, case.coefficient_name)
        lines.append(
            f"{case.label} & {case.pde_description} & {case.eos_note} & "
            f"${coeff}={case.coefficient_value:g}$ & {cap} & 64, 128, 256 \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    SETUP_TABLE_TEX.write_text("\n".join(lines) + "\n")


def write_highres_table(summary_rows: list[dict]) -> None:
    rows_by_key = {
        (row["case"], row["method"], int(row["resolution"])): row for row in summary_rows
    }
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Case & explicit RMS & STS RMS & STS/explicit & explicit $t$ [s] & STS $t$ [s] & speedup & cycles (exp/STS) \\",
        r"\midrule",
    ]
    for case in CASES:
        explicit = rows_by_key[(case.name, "explicit", PROFILE_RESOLUTION)]
        sts = rows_by_key[(case.name, "sts", PROFILE_RESOLUTION)]
        error_ratio = sts["rms_l1_median"] / explicit["rms_l1_median"]
        speedup = explicit["wall_seconds_median"] / sts["wall_seconds_median"]
        cycles = f"{int(explicit['final_cycle_median'])}/{int(sts['final_cycle_median'])}"
        lines.append(
            f"{case.label} & {fmt_float(explicit['rms_l1_median'])} & "
            f"{fmt_float(sts['rms_l1_median'])} & {fmt_decimal(error_ratio)} & "
            f"{fmt_decimal(explicit['wall_seconds_median'])} & "
            f"{fmt_decimal(sts['wall_seconds_median'])} & {fmt_decimal(speedup)} & "
            f"{cycles} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    HIGHRES_TABLE_TEX.write_text("\n".join(lines) + "\n")


def write_results_findings(summary_rows: list[dict]) -> None:
    rows_by_key = {
        (row["case"], row["method"], int(row["resolution"])): row for row in summary_rows
    }
    lines = [r"\begin{itemize}"]
    for case in CASES:
        explicit = rows_by_key[(case.name, "explicit", PROFILE_RESOLUTION)]
        sts = rows_by_key[(case.name, "sts", PROFILE_RESOLUTION)]
        error_ratio = sts["rms_l1_median"] / explicit["rms_l1_median"]
        speedup = explicit["wall_seconds_median"] / sts["wall_seconds_median"]
        if error_ratio < 0.95:
            accuracy_phrase = "and a smaller RMS error than"
        elif error_ratio > 1.05:
            accuracy_phrase = "while keeping the RMS error in the same small-error regime as"
        else:
            accuracy_phrase = "with essentially the same RMS error as"
        lines.append(
            r"\item "
            + f"{case.label}: at $N_x={PROFILE_RESOLUTION}$, STS is "
            + f"{fmt_decimal(speedup)}x faster than the explicit run, {accuracy_phrase} "
            + f"explicit ({fmt_float(sts['rms_l1_median'])} vs. "
            + f"{fmt_float(explicit['rms_l1_median'])}, ratio {fmt_decimal(error_ratio)})."
        )
    lines.append(r"\end{itemize}")
    RESULTS_FINDINGS_TEX.write_text("\n".join(lines) + "\n")


def benchmark_case(case: CaseConfig, method: str, resolution: int, env: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    total_runs = WARMUP_RUNS + MEASURED_RUNS
    for repeat in range(total_runs):
        run_kind = "warmup" if repeat < WARMUP_RUNS else "measure"
        basename = unique_basename(case, method, resolution, run_kind, repeat)
        flags = base_flags(case, method, resolution, basename)
        t0 = time.perf_counter()
        output = run_command(
            ["./athena", "-i", str(case.input_file)] + flags,
            cwd=BUILD_SRC,
            env=env,
            capture_output=True,
        )
        wall_seconds = time.perf_counter() - t0
        metrics = parse_output_metrics(output)
        errors = load_error_row(basename)

        if run_kind == "measure":
            rows.append(
                {
                    "case": case.name,
                    "method": method,
                    "resolution": resolution,
                    "repeat": repeat - WARMUP_RUNS + 1,
                    "wall_seconds": wall_seconds,
                    "cpu_time_used": metrics["cpu_time_used"],
                    "zonecycles_per_cpu_second": metrics["zonecycles_per_cpu_second"],
                    "final_cycle": metrics["final_cycle"],
                    "first_dt": metrics["first_dt"],
                    "rms_l1": errors["rms_l1"],
                    "linfty": errors["linfty"],
                }
            )

        remove_outputs(basename)
    return rows


def capture_profiles(env: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for case in CASES:
        temp_input = make_profile_input(case)
        try:
            for method in ("explicit", "sts"):
                basename = unique_basename(case, method, PROFILE_RESOLUTION, "profile", 0)
                flags = profile_flags(case, method, PROFILE_RESOLUTION, basename)
                run_command(
                    ["./athena", "-i", str(temp_input)] + flags,
                    cwd=BUILD_SRC,
                    env=env,
                    capture_output=True,
                )
                rows.extend(load_profile_rows(case, method, PROFILE_RESOLUTION, basename))
                remove_outputs(basename)
        finally:
            temp_input.unlink(missing_ok=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark STS diffusion accuracy and cost.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clean and rebuild the CPU benchmark binary before running.",
    )
    parser.add_argument(
        "--build-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel build threads.",
    )
    args = parser.parse_args()

    clear_output_tree()
    rebuilt = ensure_build(rebuild=args.rebuild, build_threads=args.build_threads)
    clear_benchmark_run_files()
    write_environment_metadata(rebuilt=rebuilt, build_threads=args.build_threads)
    write_setup_table()

    env = fixed_thread_env()
    raw_rows: list[dict] = []
    for case in CASES:
        for method in ("explicit", "sts"):
            for resolution in RESOLUTIONS:
                raw_rows.extend(benchmark_case(case, method, resolution, env))

    summary_rows = summarize_rows(raw_rows)
    assert_regression_sanity(summary_rows)
    profile_rows = capture_profiles(env)
    assert_profile_sanity(profile_rows)

    write_csv(
        RAW_CSV,
        [
            "case",
            "method",
            "resolution",
            "repeat",
            "wall_seconds",
            "cpu_time_used",
            "zonecycles_per_cpu_second",
            "final_cycle",
            "first_dt",
            "rms_l1",
            "linfty",
        ],
        raw_rows,
    )
    write_csv(
        SUMMARY_CSV,
        [
            "case",
            "method",
            "resolution",
            "samples",
            "wall_seconds_median",
            "wall_seconds_min",
            "wall_seconds_max",
            "wall_seconds_std",
            "cpu_time_used_median",
            "zonecycles_per_cpu_second_median",
            "rms_l1_median",
            "linfty_median",
            "first_dt_median",
            "final_cycle_median",
        ],
        summary_rows,
    )
    write_csv(
        PROFILE_CSV,
        [
            "case",
            "method",
            "resolution",
            "x1",
            "raw_value",
            "plotted_value",
            "analytic_value",
        ],
        profile_rows,
    )

    write_highres_table(summary_rows)
    write_results_findings(summary_rows)

    print(f"Wrote raw benchmark data to {RAW_CSV}")
    print(f"Wrote summary benchmark data to {SUMMARY_CSV}")
    print(f"Wrote profile samples to {PROFILE_CSV}")


if __name__ == "__main__":
    main()
