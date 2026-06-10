#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = [
    "time",
    "dt",
    "rho-max",
    "alpha-min",
    "alpha-max",
    "chi-min",
    "chi-max",
    "gbar-min",
    "gbar-max",
    "psi4-min",
    "psi4-max",
    "detg-min",
    "detg-max",
    "Kdd-max",
    "Add-max",
    "Theta-max",
    "Khat-max",
    "bad-metric",
    "beta-max",
    "B-max",
    "rho-mass",
    "rho-avg",
]

Z4C_LABELS = [
    "time",
    "dt",
    "C-norm2",
    "H-norm2",
    "M-norm2",
    "Z-norm2",
    "Mx-norm2",
    "My-norm2",
    "Mz-norm2",
    "Theta-norm",
    "Volume",
]

SUMMARY_FIELDS = [
    "rho-max",
    "detg-min",
    "psi4-min",
    "chi-min",
    "chi-max",
    "Kdd-max",
    "Add-max",
    "Theta-max",
    "Khat-max",
    "bad-metric",
    "rho-mass",
]

PLOT_FIELDS = ["detg-min", "psi4-min", "Khat-max", "bad-metric", "rho-max"]
Z4C_SUMMARY_FIELDS = ["C-norm2", "H-norm2", "M-norm2", "Z-norm2", "Theta-norm"]
Z4C_PLOT_FIELDS = ["C-norm2", "H-norm2", "M-norm2", "Theta-norm"]


def find_history(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("*.user.hst"))
    files = [path for path in files if ".z4c.user.hst" not in path.name]
    if not files:
        raise FileNotFoundError(f"no metric user history found in {run_dir}")
    return files[0]


def find_z4c_history(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("*.z4c.user.hst"))
    return files[0] if files else None


def read_history(path: Path) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < len(LABELS):
        raise ValueError(f"{path} has {data.shape[1]} columns, expected at least {len(LABELS)}")
    return {label: data[:, idx] for idx, label in enumerate(LABELS)}


def read_labeled_history(path: Path, labels: Sequence[str]) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < len(labels):
        raise ValueError(f"{path} has {data.shape[1]} columns, expected at least {len(labels)}")
    return {label: data[:, idx] for idx, label in enumerate(labels)}


def finite_rows(history: Dict[str, np.ndarray], fields: Sequence[str]) -> np.ndarray:
    mask = np.isfinite(history["time"])
    for field in fields:
        mask &= np.isfinite(history[field])
    return mask


def first_nonfinite_time(history: Dict[str, np.ndarray], fields: Sequence[str]) -> Optional[float]:
    mask = finite_rows(history, fields)
    bad = np.flatnonzero(~mask)
    if bad.size == 0:
        return None
    return float(history["time"][bad[0]])


def summarize(history: Dict[str, np.ndarray], fields: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for field in fields:
        finite = finite_rows(history, ["time", field])
        first_bad = first_nonfinite_time(history, [field])
        if not np.any(finite):
            rows.append({
                "field": field,
                "initial": "",
                "final": "",
                "min": "",
                "max": "",
                "abs_change": "",
                "growth_per_time": "",
                "first_nonfinite_time": "" if first_bad is None else first_bad,
            })
            continue
        values = history[field][finite]
        times = history["time"][finite]
        initial = float(values[0])
        final = float(values[-1])
        elapsed = float(times[-1] - times[0])
        rows.append({
            "field": field,
            "initial": initial,
            "final": final,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "abs_change": float(abs(final - initial)),
            "growth_per_time": "" if elapsed <= 0.0 else float(abs(final - initial) / elapsed),
            "first_nonfinite_time": "" if first_bad is None else first_bad,
        })
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def interp_at(history: Dict[str, np.ndarray], field: str, time: float) -> float:
    finite = finite_rows(history, ["time", field])
    times = history["time"][finite]
    values = history[field][finite]
    if times.size == 0:
        return math.nan
    if time <= times[0]:
        return float(values[0])
    if time >= times[-1]:
        return float(values[-1])
    return float(np.interp(time, times, values))


def defect(field: str, value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    if field in ("detg-min", "detg-max", "psi4-min", "psi4-max"):
        return abs(value - 1.0)
    return abs(value)


def compare_baseline(
    case_history: Dict[str, np.ndarray],
    baseline_history: Dict[str, np.ndarray],
    fields: Sequence[str],
) -> List[Dict[str, object]]:
    case_finite = finite_rows(case_history, fields)
    baseline_finite = finite_rows(baseline_history, fields)
    if not np.any(case_finite) or not np.any(baseline_finite):
        common_time = math.nan
    else:
        common_time = float(min(case_history["time"][case_finite][-1], baseline_history["time"][baseline_finite][-1]))
    rows: List[Dict[str, object]] = []
    for field in fields:
        case_value = interp_at(case_history, field, common_time) if math.isfinite(common_time) else math.nan
        baseline_value = interp_at(baseline_history, field, common_time) if math.isfinite(common_time) else math.nan
        case_defect = defect(field, case_value)
        baseline_defect = defect(field, baseline_value)
        rows.append({
            "field": field,
            "common_time": common_time,
            "case_value": case_value,
            "baseline_value": baseline_value,
            "case_minus_baseline": case_value - baseline_value,
            "case_defect": case_defect,
            "baseline_defect": baseline_defect,
            "defect_ratio_case_over_baseline": (
                math.nan if baseline_defect == 0.0 or not math.isfinite(baseline_defect)
                else case_defect / baseline_defect
            ),
        })
    return rows


def plot_case(history: Dict[str, np.ndarray], case: str, output_path: Path) -> None:
    fig, axes = plt.subplots(len(PLOT_FIELDS), 1, figsize=(8, 11), sharex=True)
    for axis, field in zip(axes, PLOT_FIELDS):
        finite = finite_rows(history, ["time", field])
        axis.plot(history["time"][finite], history[field][finite], marker="o", linewidth=1.4, markersize=3)
        axis.set_ylabel(field)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time")
    fig.suptitle(case)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_z4c_case(history: Dict[str, np.ndarray], case: str, output_path: Path) -> None:
    fig, axes = plt.subplots(len(Z4C_PLOT_FIELDS), 1, figsize=(8, 9), sharex=True)
    for axis, field in zip(axes, Z4C_PLOT_FIELDS):
        finite = finite_rows(history, ["time", field])
        axis.semilogy(history["time"][finite], history[field][finite], marker="o", linewidth=1.4, markersize=3)
        axis.set_ylabel(field)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"{case} Z4c constraints")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def process_case(
    case: str,
    run_root: Path,
    post_root: Path,
    baseline_history: Dict[str, np.ndarray],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    run_dir = run_root / case
    case_post = post_root / case
    history_path = find_history(run_dir)
    history = read_history(history_path)
    z4c_path = find_z4c_history(run_dir)
    z4c_history = read_labeled_history(z4c_path, Z4C_LABELS) if z4c_path is not None else None
    summary_rows = summarize(history, SUMMARY_FIELDS)
    comparison_rows = compare_baseline(history, baseline_history, ["detg-min", "psi4-min", "chi-max", "Kdd-max", "Khat-max", "rho-max"])

    write_csv(
        case_post / "metric_history_summary.csv",
        summary_rows,
        ["field", "initial", "final", "min", "max", "abs_change", "growth_per_time", "first_nonfinite_time"],
    )
    write_csv(
        case_post / "baseline_common_time_comparison.csv",
        comparison_rows,
        [
            "field",
            "common_time",
            "case_value",
            "baseline_value",
            "case_minus_baseline",
            "case_defect",
            "baseline_defect",
            "defect_ratio_case_over_baseline",
        ],
    )
    plot_case(history, case, case_post / "metric_history_trends.png")
    if z4c_history is not None:
        z4c_summary_rows = summarize(z4c_history, Z4C_SUMMARY_FIELDS)
        write_csv(
            case_post / "z4c_constraint_summary.csv",
            z4c_summary_rows,
            ["field", "initial", "final", "min", "max", "abs_change", "growth_per_time", "first_nonfinite_time"],
        )
        plot_z4c_case(z4c_history, case, case_post / "z4c_constraint_trends.png")

    finite = finite_rows(history, SUMMARY_FIELDS)
    last_finite_index = int(np.flatnonzero(finite)[-1]) if np.any(finite) else -1
    first_bad = first_nonfinite_time(history, SUMMARY_FIELDS)
    z4c_last_finite_index = -1
    z4c_first_bad: Optional[float] = None
    if z4c_history is not None:
        z4c_finite = finite_rows(z4c_history, Z4C_SUMMARY_FIELDS)
        z4c_last_finite_index = int(np.flatnonzero(z4c_finite)[-1]) if np.any(z4c_finite) else -1
        z4c_first_bad = first_nonfinite_time(z4c_history, Z4C_SUMMARY_FIELDS)
    case_row = {
        "case": case,
        "history_path": str(history_path),
        "final_finite_time": "" if last_finite_index < 0 else float(history["time"][last_finite_index]),
        "first_nonfinite_time": "" if first_bad is None else first_bad,
        "bad_metric_final_finite": "" if last_finite_index < 0 else float(history["bad-metric"][last_finite_index]),
        "detg_min_final_finite": "" if last_finite_index < 0 else float(history["detg-min"][last_finite_index]),
        "psi4_min_final_finite": "" if last_finite_index < 0 else float(history["psi4-min"][last_finite_index]),
        "Khat_max_final_finite": "" if last_finite_index < 0 else float(history["Khat-max"][last_finite_index]),
        "summary_path": str(case_post / "metric_history_summary.csv"),
        "comparison_path": str(case_post / "baseline_common_time_comparison.csv"),
        "plot_path": str(case_post / "metric_history_trends.png"),
        "z4c_history_path": "" if z4c_path is None else str(z4c_path),
        "z4c_final_finite_time": (
            "" if z4c_history is None or z4c_last_finite_index < 0
            else float(z4c_history["time"][z4c_last_finite_index])
        ),
        "z4c_first_nonfinite_time": "" if z4c_first_bad is None else z4c_first_bad,
        "C_norm2_final_finite": (
            "" if z4c_history is None or z4c_last_finite_index < 0
            else float(z4c_history["C-norm2"][z4c_last_finite_index])
        ),
        "H_norm2_final_finite": (
            "" if z4c_history is None or z4c_last_finite_index < 0
            else float(z4c_history["H-norm2"][z4c_last_finite_index])
        ),
        "M_norm2_final_finite": (
            "" if z4c_history is None or z4c_last_finite_index < 0
            else float(z4c_history["M-norm2"][z4c_last_finite_index])
        ),
        "Z_norm2_final_finite": (
            "" if z4c_history is None or z4c_last_finite_index < 0
            else float(z4c_history["Z-norm2"][z4c_last_finite_index])
        ),
        "Theta_norm_final_finite": (
            "" if z4c_history is None or z4c_last_finite_index < 0
            else float(z4c_history["Theta-norm"][z4c_last_finite_index])
        ),
        "z4c_summary_path": "" if z4c_history is None else str(case_post / "z4c_constraint_summary.csv"),
        "z4c_plot_path": "" if z4c_history is None else str(case_post / "z4c_constraint_trends.png"),
    }
    return case_row, comparison_rows


def plot_combined(case_rows: List[Dict[str, object]], run_root: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fields = ["detg-min", "psi4-min", "Khat-max", "bad-metric"]
    for axis, field in zip(axes.reshape(-1), fields):
        for row in case_rows:
            case = str(row["case"])
            history = read_history(find_history(run_root / case))
            finite = finite_rows(history, ["time", field])
            if np.any(finite):
                axis.plot(history["time"][finite], history[field][finite], label=case, linewidth=1.2)
        axis.set_ylabel(field)
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_combined_z4c(case_rows: List[Dict[str, object]], run_root: Path, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, field in zip(axes.reshape(-1), Z4C_PLOT_FIELDS):
        for row in case_rows:
            case = str(row["case"])
            z4c_path = find_z4c_history(run_root / case)
            if z4c_path is None:
                continue
            history = read_labeled_history(z4c_path, Z4C_LABELS)
            finite = finite_rows(history, ["time", field])
            if np.any(finite):
                axis.semilogy(history["time"][finite], history[field][finite], label=case, linewidth=1.2)
        axis.set_ylabel(field)
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce Gaussian-normal Z4c parameter scan histories.")
    parser.add_argument("--case", action="append", help="Case name. Can be repeated.")
    parser.add_argument("--run-root", type=Path, default=Path("/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs"))
    parser.add_argument("--post-root", type=Path, default=Path("/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post/gnc_parameter_scan"))
    parser.add_argument(
        "--baseline-case",
        default="minkowski_static_selfgrav_starfloor_smallbox_diag_2n",
        help="Reference fixed-gauge run to compare at common time.",
    )
    args = parser.parse_args()

    if not args.case:
        raise SystemExit("provide at least one --case")

    args.post_root.mkdir(parents=True, exist_ok=True)
    baseline_history = read_history(find_history(args.run_root / args.baseline_case))

    case_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    for case in args.case:
        case_row, rows = process_case(case, args.run_root, args.post_root, baseline_history)
        case_rows.append(case_row)
        comparison_rows.extend({"case": case, **row} for row in rows)

    if len(case_rows) > 1:
        write_csv(
            args.post_root / "gnc_parameter_scan_summary.csv",
            case_rows,
            [
                "case",
                "history_path",
                "final_finite_time",
                "first_nonfinite_time",
                "bad_metric_final_finite",
                "detg_min_final_finite",
                "psi4_min_final_finite",
                "Khat_max_final_finite",
                "summary_path",
                "comparison_path",
                "plot_path",
                "z4c_history_path",
                "z4c_final_finite_time",
                "z4c_first_nonfinite_time",
                "C_norm2_final_finite",
                "H_norm2_final_finite",
                "M_norm2_final_finite",
                "Z_norm2_final_finite",
                "Theta_norm_final_finite",
                "z4c_summary_path",
                "z4c_plot_path",
            ],
        )
        write_csv(
            args.post_root / "gnc_parameter_scan_common_time_comparison.csv",
            comparison_rows,
            [
                "case",
                "field",
                "common_time",
                "case_value",
                "baseline_value",
                "case_minus_baseline",
                "case_defect",
                "baseline_defect",
                "defect_ratio_case_over_baseline",
            ],
        )
        plot_combined(case_rows, args.run_root, args.post_root / "gnc_parameter_scan_metric_trends.png")
        plot_combined_z4c(case_rows, args.run_root, args.post_root / "gnc_parameter_scan_z4c_constraints.png")

    for row in case_rows:
        print(
            f"{row['case']}: final_finite_time={row['final_finite_time']} "
            f"first_nonfinite_time={row['first_nonfinite_time']} "
            f"detg_min={row['detg_min_final_finite']} Khat_max={row['Khat_max_final_finite']}"
        )


if __name__ == "__main__":
    main()
