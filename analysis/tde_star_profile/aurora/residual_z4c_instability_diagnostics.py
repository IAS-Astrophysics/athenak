#!/usr/bin/env python3
"""Summarize residual-Z4c instability runs.

The script intentionally avoids plotting dependencies.  It reports first
nonfinite history times, last finite diagnostic values, a simple exponential
growth rate for H-norm2, and the peak location of |con_H| in dense bin slices.
"""

import argparse
import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LABEL_RE_PREFIX = "#  "


def labels_from_header(path: Path) -> List[str]:
    labels: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            if not line.startswith(LABEL_RE_PREFIX):
                continue
            for item in line[3:].split():
                if not item.startswith("[") or "]=" not in item:
                    continue
                left, right = item.split("]=", 1)
                try:
                    index = int(left[1:]) - 1
                except ValueError:
                    continue
                while len(labels) <= index:
                    labels.append(f"col{len(labels) + 1}")
                labels[index] = right
    return labels


def load_history(path: Path) -> Tuple[List[str], np.ndarray]:
    labels = labels_from_header(path)
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            values = []
            for item in line.split():
                try:
                    values.append(float(item))
                except ValueError:
                    values.append(math.nan)
            rows.append(values)
    data = np.array(rows, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if not labels and data.size:
        labels = [f"col{idx + 1}" for idx in range(data.shape[1])]
    return labels, data


def finite_mask(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.array([], dtype=bool)
    return np.all(np.isfinite(data), axis=1)


def summarize_history(path: Path, wanted: Sequence[str]) -> Dict[str, object]:
    labels, data = load_history(path)
    mask = finite_mask(data)
    first_bad = int(np.flatnonzero(~mask)[0]) if np.any(~mask) else None
    finite_indices = np.flatnonzero(mask)
    last_good = int(finite_indices[-1]) if finite_indices.size else None
    summary: Dict[str, object] = {
        "file": str(path),
        "rows": int(data.shape[0]) if data.size else 0,
        "first_bad_time": None if first_bad is None or data.size == 0 else float(data[first_bad, 0]),
        "last_good_time": None if last_good is None or data.size == 0 else float(data[last_good, 0]),
    }
    if last_good is not None:
        for label in wanted:
            if label in labels:
                index = labels.index(label)
                if index < data.shape[1]:
                    summary[f"last_{label}"] = float(data[last_good, index])
    return summary


def growth_rate(path: Path, field: str, start: float, end: float) -> Optional[float]:
    labels, data = load_history(path)
    if field not in labels or data.size == 0:
        return None
    time = data[:, 0]
    values = data[:, labels.index(field)]
    good = np.isfinite(time) & np.isfinite(values) & (values > 0.0)
    time = time[good]
    values = values[good]
    if time.size < 2:
        return None
    eps = 1.0e-10 * max(1.0, abs(start), abs(end), abs(float(time[-1])))
    if start < time[0] - eps or end > time[-1] + eps:
        return None
    start = max(start, float(time[0]))
    end = min(end, float(time[-1]))
    v0 = float(np.interp(start, time, values))
    v1 = float(np.interp(end, time, values))
    if v0 <= 0.0 or v1 <= 0.0 or end <= start:
        return None
    return (math.log(v1) - math.log(v0)) / (end - start)


def import_bin_convert(repo: Path):
    sys.modules.setdefault("h5py", types.SimpleNamespace())
    sys.path.insert(0, str(repo / "vis/python"))
    import bin_convert  # type: ignore

    return bin_convert


def cell_center_from_geometry(filedata: Dict[str, object], block: int,
                              k: int, j: int, i: int) -> Tuple[float, float, float, int]:
    geom = np.asarray(filedata["mb_geometry"])[block]
    index = np.asarray(filedata["mb_index"])[block]
    logical = np.asarray(filedata["mb_logical"])[block]
    x1min, x1max, x2min, x2max, x3min, x3max = [float(x) for x in geom]
    nx1_mb = int(filedata["nx1_mb"])
    nx2_mb = int(filedata["nx2_mb"])
    nx3_mb = int(filedata["nx3_mb"])
    dx1 = (x1max - x1min) / max(nx1_mb, 1)
    dx2 = (x2max - x2min) / max(nx2_mb, 1)
    dx3 = (x3max - x3min) / max(nx3_mb, 1)
    x = x1min + (int(index[0]) + i + 0.5) * dx1
    y = x2min + (int(index[2]) + j + 0.5) * dx2
    z = x3min + (int(index[4]) + k + 0.5) * dx3
    return x, y, z, int(logical[3])


def slice_peak(repo: Path, run_dir: Path, target_time: Optional[float],
               variable: str = "con_H") -> Optional[Dict[str, object]]:
    bin_dir = run_dir / "bin"
    if not bin_dir.is_dir():
        return None
    files = sorted(bin_dir.glob("*.bin"))
    files = [path for path in files if any(tag in path.name for tag in ("xy_con", "xz_con", "yz_con"))]
    if not files:
        return None
    bin_convert = import_bin_convert(repo)
    candidates: List[Tuple[float, Path, Dict[str, object]]] = []
    for path in files:
        try:
            data = bin_convert.read_binary(str(path))
        except Exception:
            continue
        time = float(data.get("time", math.nan))
        if math.isfinite(time):
            candidates.append((abs(time - target_time) if target_time is not None else -time, path, data))
    if not candidates:
        return None
    _, path, data = min(candidates, key=lambda item: item[0])
    var_names = list(data["var_names"])
    if variable not in var_names:
        return None
    best: Optional[Dict[str, object]] = None
    for block, array in enumerate(data["mb_data"][variable]):
        values = np.asarray(array)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        abs_values = np.abs(np.where(finite, values, np.nan))
        flat = int(np.nanargmax(abs_values))
        k, j, i = np.unravel_index(flat, values.shape)
        value = float(values[k, j, i])
        x, y, z, level = cell_center_from_geometry(data, block, k, j, i)
        current = {
            "slice_file": str(path),
            "slice_time": float(data["time"]),
            "variable": variable,
            "peak_abs": abs(value),
            "peak_value": value,
            "x": x,
            "y": y,
            "z": z,
            "level": level,
            "block": block,
        }
        if best is None or current["peak_abs"] > best["peak_abs"]:
            best = current
    return best


def classify_peak(peak: Dict[str, object], star_x: float, star_y: float, star_z: float,
                  star_radius: float, bh_radius: float,
                  domain: Tuple[float, float, float, float, float, float]) -> str:
    x = float(peak["x"])
    y = float(peak["y"])
    z = float(peak["z"])
    r_star = math.sqrt((x - star_x) ** 2 + (y - star_y) ** 2 + (z - star_z) ** 2)
    r_bh = math.sqrt(x * x + y * y + z * z)
    x1min, x1max, x2min, x2max, x3min, x3max = domain
    d_boundary = min(x - x1min, x1max - x, y - x2min, x2max - y, z - x3min, x3max - z)
    peak["r_star"] = r_star
    peak["r_bh"] = r_bh
    peak["d_boundary"] = d_boundary
    if d_boundary < 1.0:
        return "outer_boundary"
    if bh_radius > 0.0 and r_bh < bh_radius:
        return "bh_or_excision"
    if r_star <= 0.8 * star_radius:
        return "star_interior"
    if r_star <= 2.0 * star_radius:
        return "star_surface_or_near_star"
    return "other_or_amr_edge"


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/hzhu/athenak_tde"))
    parser.add_argument("--case", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--growth-start", type=float, default=2.5)
    parser.add_argument("--growth-end", type=float, default=3.1)
    parser.add_argument("--star-x", type=float, default=40.0)
    parser.add_argument("--star-y", type=float, default=0.0)
    parser.add_argument("--star-z", type=float, default=0.0)
    parser.add_argument("--star-radius", type=float, default=0.47106)
    parser.add_argument("--bh-class-radius", type=float, default=4.0)
    parser.add_argument("--domain", type=float, nargs=6, default=(-25.6, 96.0, -12.8, 12.8, -12.8, 12.8))
    args = parser.parse_args()
    if args.output_root is not None and args.case is not None:
        args.output_json = args.output_json or args.output_root / args.case / "instability_diagnostics.json"
        args.output_csv = args.output_csv or args.output_root / args.case / "history_summary.csv"

    wanted = [
        "rho-max", "alpha-min", "alpha-max", "chi-min", "chi-max", "detg-min",
        "detg-max", "bad-metric", "alpha-res", "beta-res", "B-res", "Gam-res",
        "C-norm2", "H-norm2", "M-norm2", "Theta-norm", "mass", "1-mom", "tot-E",
    ]
    histories = [summarize_history(path, wanted) for path in sorted(args.run_dir.glob("*.hst"))]
    z4c_hist = next((path for path in sorted(args.run_dir.glob("*.z4c.user.hst"))), None)
    h_growth = growth_rate(z4c_hist, "H-norm2", args.growth_start, args.growth_end) if z4c_hist else None
    last_good_time = None
    if histories:
        times = [item.get("last_good_time") for item in histories if item.get("last_good_time") is not None]
        if times:
            last_good_time = min(float(time) for time in times)
    peak = slice_peak(args.repo, args.run_dir, last_good_time)
    if peak is not None:
        peak["classification"] = classify_peak(
            peak, args.star_x, args.star_y, args.star_z, args.star_radius,
            args.bh_class_radius, tuple(args.domain)
        )

    report = {
        "run_dir": str(args.run_dir),
        "histories": histories,
        "H_norm2_growth_rate": h_growth,
        "growth_window": [args.growth_start, args.growth_end],
        "constraint_peak": peak,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, histories)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
