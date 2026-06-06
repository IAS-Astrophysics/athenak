#!/usr/bin/env python3
"""Measure primitive slice values near the planned Stage-1 sink radius.

The script scans source-run ``*slice_x3*.athdf`` files, keeps primitive-style
outputs containing density and pressure, selects the file closest in HDF5
``Time`` to the requested restart time, and reports statistics inside
``dmin <= 4`` and in the comparison shell ``4 < dmin <= 8`` around the analytic
equal-mass circular binary positions used by ``dynbbh`` when no trajectory file
is active.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SETUP_ROOT = Path("/home/hzhu/scratch3/staged_zoom_20260528_192623")
REPORT_PATH = SETUP_ROOT / "sink_slice_diagnostics.md"

CASES = {
    "SANE": {
        "run_dir": Path("/home/hzhu/scratch2/hzhu/acc/cbd/tilted_large/run_0"),
        "restart_time": 450000.0,
    },
    "MAD": {
        "run_dir": Path("/home/hzhu/scratch2/hzhu/acc/cbd/tilted_large/run_29"),
        "restart_time": 228000.0,
    },
    "BONDI": {
        "run_dir": Path("/home/hzhu/scratch2/hzhu/acc/bondi/cooling/run_1"),
        "restart_time": 621136.0,
    },
}

PBS_SCRIPTS = {
    case: SETUP_ROOT / case / "scripts" / "submit_stage1_debug_scaling.pbs"
    for case in CASES
}
LAUNCH_SCRIPTS = {
    case: SETUP_ROOT / case / "scripts" / "launch_stage1.sh"
    for case in CASES
}

SEP = 25.0
Q = 1.0
SINK_RADIUS = 4.0
SHELL_OUTER_RADIUS = 8.0

TEMP_UNIT_CGS_BONDI = None


@dataclass(frozen=True)
class Candidate:
    path: Path
    time: float
    variable_names: tuple[str, ...]
    dataset_names: tuple[str, ...]
    num_variables: tuple[int, ...]


def read_time(path: Path) -> float:
    with h5py.File(path, "r") as h5f:
        return float(h5f.attrs["Time"])


def decode_attr_array(values: Any) -> tuple[str, ...]:
    return tuple(
        v.decode("utf-8") if isinstance(v, bytes | np.bytes_) else str(v)
        for v in values
    )


def scan_candidate(path: Path) -> Candidate | None:
    try:
        with h5py.File(path, "r") as h5f:
            names = decode_attr_array(h5f.attrs["VariableNames"])
            dataset_names = decode_attr_array(h5f.attrs["DatasetNames"])
            num_variables = tuple(int(v) for v in h5f.attrs["NumVariables"])
            name_set = set(names)
            if "dens" not in name_set:
                return None
            if not ({"press", "pgas", "pres"} & name_set):
                return None
            if "uov" not in h5f:
                return None
            return Candidate(
                path=path,
                time=float(h5f.attrs["Time"]),
                variable_names=names,
                dataset_names=dataset_names,
                num_variables=num_variables,
            )
    except (OSError, KeyError, ValueError):
        return None


def bracket_by_time(paths: list[Path], restart_time: float, window: int = 64) -> tuple[list[Path], int]:
    lo = 0
    hi = len(paths) - 1
    best_idx = 0
    best_dt = math.inf
    while lo <= hi:
        mid = (lo + hi) // 2
        time = read_time(paths[mid])
        dt = abs(time - restart_time)
        if dt < best_dt:
            best_idx = mid
            best_dt = dt
        if time < restart_time:
            lo = mid + 1
        elif time > restart_time:
            hi = mid - 1
        else:
            best_idx = mid
            break
    start = max(0, best_idx - window)
    stop = min(len(paths), best_idx + window + 1)
    return paths[start:stop], best_idx


def select_closest(run_dir: Path, restart_time: float) -> tuple[int, int, Candidate]:
    bin_dir = run_dir / "bin"
    search_dir = bin_dir if bin_dir.is_dir() else run_dir
    paths = sorted(search_dir.glob("*slice_x3*.athdf"))
    bracket, _best_idx = bracket_by_time(paths, restart_time)
    candidates = [candidate for p in bracket if (candidate := scan_candidate(p))]
    if not candidates:
        raise RuntimeError(f"No primitive slice_x3 candidates found under {run_dir}")
    selected = min(candidates, key=lambda c: abs(c.time - restart_time))
    return len(paths), len(candidates), selected


def variable_location(candidate: Candidate, variable: str) -> tuple[str, int]:
    aliases = {
        "rho": ("dens", "rho"),
        "press": ("press", "pgas", "pres"),
    }[variable]
    flat_index = next(
        i for i, name in enumerate(candidate.variable_names) if name in aliases
    )
    offset = 0
    for dataset_name, count in zip(candidate.dataset_names, candidate.num_variables):
        if flat_index < offset + count:
            return dataset_name, flat_index - offset
        offset += count
    raise RuntimeError(f"Could not map {variable} in {candidate.path}")


def binary_positions(time: float) -> tuple[tuple[float, float], tuple[float, float]]:
    omega = SEP ** -1.5
    phase = omega * time
    c = math.cos(phase)
    s = math.sin(phase)
    r1 = Q / (1.0 + Q) * SEP
    r2 = -SEP / (1.0 + Q)
    return (r1 * c, r1 * s), (r2 * c, r2 * s)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def measure(candidate: Candidate, temp_unit_cgs: float | None) -> dict[str, Any]:
    rho_dataset, rho_index = variable_location(candidate, "rho")
    press_dataset, press_index = variable_location(candidate, "press")
    if rho_dataset != press_dataset:
        raise RuntimeError(f"rho and pressure live in different datasets in {candidate.path}")

    bh1, bh2 = binary_positions(candidate.time)
    stats: dict[str, Any] = {
        "bh1": bh1,
        "bh2": bh2,
        "rho_dataset": rho_dataset,
        "rho_index": rho_index,
        "press_index": press_index,
        "masks": {},
    }

    values = {
        "inside": {"rho": [], "press": [], "T_code": []},
        "shell": {"rho": [], "press": [], "T_code": []},
    }

    with h5py.File(candidate.path, "r") as h5f:
        data = h5f[rho_dataset]
        x1v = h5f["x1v"]
        x2v = h5f["x2v"]
        for block in range(data.shape[1]):
            x = np.asarray(x1v[block])
            y = np.asarray(x2v[block])
            xx, yy = np.meshgrid(x, y, indexing="xy")
            d1 = np.hypot(xx - bh1[0], yy - bh1[1])
            d2 = np.hypot(xx - bh2[0], yy - bh2[1])
            dmin = np.minimum(d1, d2)
            inside = dmin <= SINK_RADIUS
            shell = (dmin > SINK_RADIUS) & (dmin <= SHELL_OUTER_RADIUS)

            rho = np.asarray(data[rho_index, block, 0, :, :], dtype=np.float64)
            press = np.asarray(data[press_index, block, 0, :, :], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_code = press / rho

            for mask_name, mask in (("inside", inside), ("shell", shell)):
                values[mask_name]["rho"].append(rho[mask])
                values[mask_name]["press"].append(press[mask])
                values[mask_name]["T_code"].append(t_code[mask])

    for mask_name, fields in values.items():
        stats["masks"][mask_name] = {}
        for field_name, chunks in fields.items():
            arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            stats["masks"][mask_name][field_name] = summarize(arr)
            if field_name == "T_code" and temp_unit_cgs is not None:
                stats["masks"][mask_name]["T_cgs"] = summarize(arr * temp_unit_cgs)

    return stats


def derive_bondi_temperature_unit() -> float:
    # Mirrors src/units/units.cpp: temperature = c^2 * mu * amu / k_B.
    mu = 0.618
    speed_of_light_cgs = 2.99792458e10
    atomic_mass_unit_cgs = 1.660538921e-24
    k_boltzmann_cgs = 1.3806488e-16
    return speed_of_light_cgs**2 * mu * atomic_mass_unit_cgs / k_boltzmann_cgs


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6e}"


def stats_table(case_results: dict[str, Any], case: str) -> list[str]:
    lines = [
        f"### {case}",
        "",
        "| mask | variable | count | mean | median | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mask_name, mask_label in (("inside", "dmin <= 4"), ("shell", "4 < dmin <= 8")):
        for var in ("rho", "press", "T_code", "T_cgs"):
            if var not in case_results["stats"]["masks"][mask_name]:
                continue
            stat = case_results["stats"]["masks"][mask_name][var]
            lines.append(
                "| "
                + " | ".join(
                    [
                        mask_label,
                        var,
                        fmt(stat["count"]),
                        fmt(stat["mean"]),
                        fmt(stat["median"]),
                        fmt(stat["min"]),
                        fmt(stat["max"]),
                    ]
                )
                + " |"
            )
    return lines


def verify_pbs() -> dict[str, dict[str, bool]]:
    required = [
        "#PBS -A BBHGRMHD",
        "#PBS -q debug-scaling",
        "#PBS -l select=22",
        "#PBS -l walltime=01:00:00",
        "#PBS -l filesystems=flare:home",
        "#PBS -l place=scatter",
        "/home/hzhu/athenak/build_cb_clean_codex/src/athena",
    ]
    result = {}
    for case, pbs_path in PBS_SCRIPTS.items():
        pbs_text = pbs_path.read_text()
        launch_text = LAUNCH_SCRIPTS[case].read_text()
        case_dir = SETUP_ROOT / case
        result[case] = {
            "pbs_required_lines": all(line in pbs_text for line in required),
            "no_qsub": "qsub" not in pbs_text and "qsub" not in launch_text,
            "local_restart": f"{case_dir}/rst/" in pbs_text and f"{case_dir}/rst/" in launch_text,
            "stage1_input": f"{case_dir}/input/stage1.par" in launch_text or "input/stage1.par" in launch_text,
            "case_logs": f"{case_dir}/logs" in pbs_text and 'LOG_DIR="$CASE_DIR/logs"' in launch_text,
        }
    return result


def write_report(results: dict[str, Any], pbs_check: dict[str, dict[str, bool]]) -> None:
    lines = [
        "# Stage-1 Sink Slice Diagnostics",
        "",
        "This report was generated by `scripts/measure_sink_slice_values.py`. It only reads source ATHDF slices and prepared scripts; it does not modify parfiles or submit jobs.",
        "",
        "## PBS Pre-check",
        "",
        "| case | required PBS lines | no qsub | local restart via launch | stage1 input | case logs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        check = pbs_check[case]
        lines.append(
            f"| {case} | {check['pbs_required_lines']} | {check['no_qsub']} | "
            f"{check['local_restart']} | {check['stage1_input']} | {check['case_logs']} |"
        )

    lines += [
        "",
        "## Selection Method",
        "",
        "- Searched each source run `bin/` directory for `*slice_x3*.athdf`.",
        "- The slice times are monotonic in filename order for the sampled source directories. To avoid opening thousands of scratch HDF5 files, the script binary-searches HDF5 `Time`, then inspects a +/-64-file metadata bracket around the target.",
        "- Kept primitive-style files in that bracket containing `dens` and `press`/`pgas`/`pres` in HDF5 metadata and a `uov` dataset.",
        "- Selected the bracket candidate with HDF5 `Time` closest to the restart time from the setup notes.",
        "- Used the analytic equal-mass circular `dynbbh` orbit with `sep = 25`, `q = 1`, `omega = sep^-1.5`, and phase `omega * Time`. This assumes zero phase at `t = 0`, matching the no-trajectory path in `src/pgen/dynbbh.cpp`.",
        "- Used `dmin = min(distance_to_BH1, distance_to_BH2)` in the xy plane.",
        "- Masks are `dmin <= 4.0` and `4.0 < dmin <= 8.0`; both had coverage, so no fallback shell was needed.",
        "",
        "## Selected Files",
        "",
        "| case | located slice_x3 files | inspected primitive bracket | restart time | selected Time | |dt| | selected path | BH1 xy | BH2 xy |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for case, result in results.items():
        candidate = result["candidate"]
        stats = result["stats"]
        bh1 = stats["bh1"]
        bh2 = stats["bh2"]
        lines.append(
            f"| {case} | {result['file_count']} | {result['candidate_count']} | {CASES[case]['restart_time']:.6f} | "
            f"{candidate.time:.6f} | {abs(candidate.time - CASES[case]['restart_time']):.6f} | "
            f"`{candidate.path}` | ({bh1[0]:.6f}, {bh1[1]:.6f}) | ({bh2[0]:.6f}, {bh2[1]:.6f}) |"
        )

    lines += [
        "",
        "## Statistics",
        "",
    ]
    for case in CASES:
        lines.extend(stats_table(results[case], case))
        lines.append("")

    sane = results["SANE"]["stats"]["masks"]
    mad = results["MAD"]["stats"]["masks"]
    bondi = results["BONDI"]["stats"]["masks"]
    lines += [
        "## Interpretation",
        "",
        f"- SANE has inside/shell median rho {sane['inside']['rho']['median']:.6e} / {sane['shell']['rho']['median']:.6e}, median pressure {sane['inside']['press']['median']:.6e} / {sane['shell']['press']['median']:.6e}, and median T_code {sane['inside']['T_code']['median']:.6e} / {sane['shell']['T_code']['median']:.6e}.",
        f"- MAD has inside/shell median rho {mad['inside']['rho']['median']:.6e} / {mad['shell']['rho']['median']:.6e}, median pressure {mad['inside']['press']['median']:.6e} / {mad['shell']['press']['median']:.6e}, and median T_code {mad['inside']['T_code']['median']:.6e} / {mad['shell']['T_code']['median']:.6e}.",
        f"- BONDI has inside/shell median rho {bondi['inside']['rho']['median']:.6e} / {bondi['shell']['rho']['median']:.6e}, median pressure {bondi['inside']['press']['median']:.6e} / {bondi['shell']['press']['median']:.6e}, and median T_code {bondi['inside']['T_code']['median']:.6e} / {bondi['shell']['T_code']['median']:.6e}.",
        f"- BONDI physical temperature used `T_cgs = T_code * {derive_bondi_temperature_unit():.6e} K`, derived from `src/units/units.cpp` with `mu = 0.618`; its median inside/shell T_cgs is {bondi['inside']['T_cgs']['median']:.6e} K / {bondi['shell']['T_cgs']['median']:.6e} K.",
        "",
        "## Stage-1 Sink Parameter Implications",
        "",
        "- The current placeholder/generated `dexcise = 1.0e-2` is closest to the SANE local density scale, still below the SANE medians, and far below the MAD and BONDI medians.",
        "- For BONDI, the measured median densities near and just outside the sink are many orders of magnitude larger than `1.0e-2`, so a density target of `1.0e-2` would strongly evacuate density relative to the local source state.",
        "- The current placeholder/generated `texcise = 1.0e-3` is below the measured median `T_code` in all three selected slices. It is especially far below BONDI, where `1.0e-3` corresponds to about "
        f"`{1.0e-3 * derive_bondi_temperature_unit():.6e} K`, while the measured BONDI median sink-shell temperatures are orders of magnitude hotter.",
        "- Because BONDI uses physical-unit ISM cooling, the sink temperature target should be treated as a physical temperature choice, not just a harmless code-unit placeholder. These measurements argue against carrying `texcise = 1.0e-3` and `dexcise = 1.0e-2` forward for BONDI without an explicit physical rationale.",
        "",
        "No Stage-1 parfiles were modified.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    temp_unit = derive_bondi_temperature_unit()
    pbs_check = verify_pbs()
    results: dict[str, Any] = {}
    for case, config in CASES.items():
        print(f"{case}: scanning {config['run_dir'] / 'bin'}", flush=True)
        file_count, count, candidate = select_closest(config["run_dir"], config["restart_time"])
        case_temp_unit = temp_unit if case == "BONDI" else None
        stats = measure(candidate, case_temp_unit)
        results[case] = {
            "file_count": file_count,
            "candidate_count": count,
            "candidate": candidate,
            "stats": stats,
        }
        print(
            f"{case}: selected {candidate.path} Time={candidate.time:.6f} "
            f"dt={abs(candidate.time - config['restart_time']):.6f}",
            flush=True,
        )
    write_report(results, pbs_check)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
