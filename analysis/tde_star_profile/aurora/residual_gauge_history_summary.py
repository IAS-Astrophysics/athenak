#!/usr/bin/env python3
"""Summarize residual-gauge debug history files.

Run after `source ~/athenak_env` when using the Aurora analysis environment.
"""

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Set


LABEL_RE = re.compile(r"\[(\d+)\]=(\S+)")


def labels_from_header(path: Path) -> List[str]:
    labels = []  # type: List[str]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            for idx, label in LABEL_RE.findall(line):
                pos = int(idx) - 1
                while len(labels) <= pos:
                    labels.append(f"col{len(labels) + 1}")
                labels[pos] = label
    return labels


def load_rows(path: Path) -> List[List[float]]:
    rows = []  # type: List[List[float]]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(value) for value in line.split()])
    return rows


def row_is_finite(row: List[float]) -> bool:
    return all(math.isfinite(value) for value in row)


def summarize(path: Path, interesting: Set[str]) -> Dict[str, str]:
    labels = labels_from_header(path)
    rows = load_rows(path)
    finite = [row_is_finite(row) for row in rows]
    first_bad_idx = next((idx for idx, ok in enumerate(finite) if not ok), None)
    last_finite_idx = None
    for idx, ok in enumerate(finite):
      if ok:
        last_finite_idx = idx

    out = {
        "file": str(path),
        "rows": str(len(rows)),
        "last_finite_row": "" if last_finite_idx is None else str(last_finite_idx),
        "last_finite_time": "",
        "first_bad_row": "" if first_bad_idx is None else str(first_bad_idx),
        "first_bad_time": "",
    }
    if last_finite_idx is not None and rows and labels:
        out["last_finite_time"] = f"{rows[last_finite_idx][0]:.16e}"
    if first_bad_idx is not None and rows and labels:
        out["first_bad_time"] = f"{rows[first_bad_idx][0]:.16e}"

    if last_finite_idx is not None:
        last = rows[last_finite_idx]
        for label, value in zip(labels, last):
            if label in interesting:
                out[f"last_{label}"] = f"{value:.16e}"

    return out


def finite_series(labels: List[str], rows: List[List[float]], label: str):
    if label not in labels:
        return [], []
    index = labels.index(label)
    times = []
    values = []
    for row in rows:
        if not row_is_finite(row):
            break
        if len(row) <= index:
            continue
        time = row[0]
        value = row[index]
        if math.isfinite(time) and math.isfinite(value):
            times.append(time)
            values.append(value)
    return times, values


def plot_histories(run_dir: Path, plot_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_sets = {
        "metric_gauge_trends.png": [
            "alpha-min", "alpha-max", "chi-min", "detg-min",
            "Khat-max", "Theta-max", "alpha-res", "beta-res", "Gam-res",
        ],
        "lapse_source_trends.png": [
            "src-full", "src-bg", "src-res",
            "src-adapt", "aK-bg", "Khat-res",
        ],
        "z4c_constraint_trends.png": [
            "C-norm2", "H-norm2", "M-norm2", "Z-norm2", "Theta-norm",
        ],
        "mhd_integral_trends.png": [
            "rho-max", "mass", "1-mom", "2-mom", "3-mom", "tot-E",
        ],
    }

    for output_name, wanted in plot_sets.items():
        series = []
        for path in sorted(run_dir.glob("*.hst")):
            labels = labels_from_header(path)
            rows = load_rows(path)
            for label in wanted:
                times, values = finite_series(labels, rows, label)
                if times:
                    series.append((path.name, label, times, values))
        if not series:
            continue

        nplot = len(series)
        fig_height = max(4.0, 1.45 * nplot)
        fig, axes = plt.subplots(nplot, 1, figsize=(9, fig_height), sharex=True)
        if nplot == 1:
            axes = [axes]
        for axis, (filename, label, times, values) in zip(axes, series):
            positive = all(value > 0.0 for value in values)
            if positive and (max(values) / max(min(values), 1.0e-300) > 1.0e3):
                axis.semilogy(times, values, marker="o", linewidth=1.2, markersize=2.5)
            else:
                axis.plot(times, values, marker="o", linewidth=1.2, markersize=2.5)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            axis.set_title(filename, fontsize=9)
        axes[-1].set_xlabel("time")
        fig.tight_layout()
        fig.savefig(plot_dir / output_name, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument(
        "--interesting",
        default=(
            "rho-max,alpha-min,alpha-max,chi-min,detg-min,Kdd-max,Add-max,"
            "Theta-max,Khat-max,bad-metric,alpha-res,beta-res,B-res,Gam-res,"
            "src-full,src-bg,src-res,src-adapt,aK-bg,Khat-res,"
            "C-norm2,H-norm2,M-norm2,Theta-norm,mass,1-mom,2-mom,3-mom,tot-E"
        ),
    )
    args = parser.parse_args()

    interesting = {item.strip() for item in args.interesting.split(",") if item.strip()}
    paths = sorted(args.run_dir.glob("*.hst"))
    summaries = [summarize(path, interesting) for path in paths]
    keys = []  # type: List[str]
    for summary in summaries:
        for key in summary:
            if key not in keys:
                keys.append(key)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)
    if args.plot_dir:
        plot_histories(args.run_dir, args.plot_dir)

    for summary in summaries:
        print(summary["file"])
        for key in keys:
            if key != "file" and summary.get(key, ""):
                print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
