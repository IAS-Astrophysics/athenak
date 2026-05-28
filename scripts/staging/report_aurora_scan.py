#!/usr/bin/env python3
"""Summarize Aurora staging scan manifests."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CATEGORY_ORDER = [
    "restart",
    "restart_rank_local",
    "mhd_w_bcc",
    "slice",
    "torque_excluded",
    "am_excluded",
    "history",
    "parfile_input",
    "npz",
    "png",
    "other_athdf",
    "other_candidate",
]
TRANSFER_BUDGET_BYTES = 90 * 1000**4


def read_jsonl(path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def human_size(num):
    if num is None:
        return "unknown"
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1000.0 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1000.0
    return f"{value:.2f} PB"


def load_run_summary(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def write_report(output_dir):
    summaries = load_run_summary(output_dir / "run_summary.csv")

    counts = defaultdict(int)
    sizes = defaultdict(int)
    selected_counts = defaultdict(int)
    selected_sizes = defaultdict(int)
    for record in read_jsonl(output_dir / "classified_inventory.jsonl") or []:
        counts[record["category"]] += 1
        sizes[record["category"]] += record.get("size") or 0
    for record in read_jsonl(output_dir / "selected_transfer_manifest.jsonl") or []:
        selected_counts[record["category"]] += 1
        selected_sizes[record["category"]] += record.get("size") or 0

    total_selected = sum(selected_sizes.values())
    missing_period = [
        row["run_dir"] for row in summaries
        if not row.get("orbit_period") or row.get("orbit_confidence") in ("none", "low")
    ]
    restart_ambiguous = [
        row["run_dir"] for row in summaries
        if row.get("restart_grouping_ambiguous") == "True"
    ]
    rank_local = [
        row["run_dir"] for row in summaries
        if row.get("single_file_per_rank_detected") == "True"
    ]
    missing_par = [
        row["run_dir"] for row in summaries
        if row.get("has_parfile") == "False"
    ]
    missing_outputs = [
        (row["run_dir"], row.get("missing_expected_outputs", ""))
        for row in summaries
        if row.get("missing_expected_outputs")
    ]

    lines = []
    lines.append("# Aurora Staging Scan Report")
    lines.append("")
    lines.append(f"- Runs found: {len(summaries)}")
    lines.append(
        f"- Estimated selected transfer size: {human_size(total_selected)} "
        f"({total_selected} bytes)"
    )
    if total_selected > TRANSFER_BUDGET_BYTES:
        lines.append(
            f"- **WARNING:** selected size exceeds the 90 TB budget "
            f"({human_size(TRANSFER_BUDGET_BYTES)})."
        )
    else:
        lines.append(
            f"- Selected size is within the rough 90 TB budget "
            f"({human_size(TRANSFER_BUDGET_BYTES)})."
        )
    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Candidates | Candidate size | Selected | Selected size |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for category in CATEGORY_ORDER:
        lines.append(
            f"| {category} | {counts[category]} | {human_size(sizes[category])} | "
            f"{selected_counts[category]} | {human_size(selected_sizes[category])} |"
        )
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(
        "| Run | Selected size | Orbit period | Confidence | Rank-local restart | "
        "Latest restart | Latest mhd_w_bcc | Missing |"
    )
    lines.append("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
    for row in summaries:
        period = row.get("orbit_period") or ""
        try:
            period_text = f"{float(period):.8g}" if period else ""
        except ValueError:
            period_text = period
        selected_size = human_size(as_int(row.get("selected_total_bytes")))
        lines.append(
            f"| {row['run_dir']} | {selected_size} | "
            f"{period_text} | {row.get('orbit_confidence', '')} | "
            f"{row.get('single_file_per_rank_detected', '')} | "
            f"{Path(row.get('latest_restart_selected', '')).name} | "
            f"{Path(row.get('latest_mhd_w_bcc_selected', '')).name} | "
            f"{row.get('missing_expected_outputs', '')} |"
        )
    lines.append("")
    lines.append("## Included Analysis Products")
    lines.append("")
    lines.append(
        f"- NPZ selected: {selected_counts['npz']} files, "
        f"{human_size(selected_sizes['npz'])}"
    )
    lines.append(
        f"- PNG selected: {selected_counts['png']} files, "
        f"{human_size(selected_sizes['png'])}"
    )
    lines.append("")
    lines.append("## Excluded AthenaDF Outputs")
    lines.append("")
    lines.append(
        f"- Torque AthenaDF excluded: {counts['torque_excluded']} files, "
        f"{human_size(sizes['torque_excluded'])}"
    )
    lines.append(
        f"- Angular-momentum AthenaDF excluded: {counts['am_excluded']} files, "
        f"{human_size(sizes['am_excluded'])}"
    )
    lines.append("")
    lines.append("## Potential Problems")
    lines.append("")
    add_problem(
        lines,
        "Runs where orbital period could not be confidently inferred",
        missing_period,
    )
    add_problem(lines, "Runs with ambiguous restart grouping", restart_ambiguous)
    add_problem(lines, "Runs with rank-local restarts", rank_local)
    add_problem(lines, "Runs missing parfile.par", missing_par)
    if missing_outputs:
        lines.append("- Runs with missing expected outputs:")
        for run_dir, missing in missing_outputs:
            lines.append(f"  - `{run_dir}`: {missing}")
    else:
        lines.append("- Runs with missing expected outputs: none")
    lines.append("")
    lines.append("## Manifest Notes")
    lines.append("")
    lines.append(
        "- `selected_transfer_manifest.txt` is intended for later "
        "`rsync --files-from` use. See `manifest_metadata.json` for the "
        "source root assumption."
    )
    lines.append(
        "- This report is generated from manifests only; it performs no copy or "
        "simulation-data mutation."
    )

    report_path = output_dir / "scan_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


def add_problem(lines, title, items):
    if not items:
        lines.append(f"- {title}: none")
        return
    lines.append(f"- {title}:")
    for item in items:
        lines.append(f"  - `{item}`")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_report(args.output_dir)


if __name__ == "__main__":
    main()
