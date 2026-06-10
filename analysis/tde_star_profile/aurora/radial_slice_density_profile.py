#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_bin_convert():
    vis_dir = repo_root() / "vis" / "python"
    if str(vis_dir) not in sys.path:
        sys.path.insert(0, str(vis_dir))
    import bin_convert  # type: ignore

    return bin_convert


def unique_sorted(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def find_files(run_dir: Path, output_id: str) -> List[Path]:
    candidates: List[Path] = []
    for base in (run_dir / "bin", run_dir):
        candidates.extend(base.glob(f"*.{output_id}.*.bin"))
    if not candidates:
        candidates.extend(run_dir.rglob(f"*.{output_id}.*.bin"))
    return unique_sorted(candidates)


def output_number(path: Path) -> str:
    match = re.search(r"\.(\d{5})\.bin$", path.name)
    return match.group(1) if match else path.stem.rsplit(".", 1)[-1]


def as_2d(field: np.ndarray, quantity: str) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(field))
    if squeezed.ndim != 2:
        raise ValueError(f"Expected {quantity} to become 2D after squeeze, got {field.shape}")
    return squeezed


def second_axis(data: Dict[str, np.ndarray], field: np.ndarray) -> Tuple[np.ndarray, str]:
    n_second = field.shape[0]
    for name in ("x2v", "x3v"):
        coord = np.asarray(data[name])
        if coord.size == n_second:
            return coord, "y" if name == "x2v" else "z"
    raise ValueError(
        f"Could not match second coordinate to slice shape {field.shape}: "
        f"x2v={np.asarray(data['x2v']).shape}, x3v={np.asarray(data['x3v']).shape}"
    )


def read_density_slice(path: Path) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray, float, int]:
    bin_convert = load_bin_convert()
    data = bin_convert.read_binary_as_athdf(str(path), quantities=["dens"])
    dens = as_2d(data["dens"], "dens")
    x = np.asarray(data["x1v"], dtype=np.float64)
    second, axis_name = second_axis(data, dens)
    time = float(data.get("Time", np.nan))
    cycle = int(data.get("NumCycles", -1))
    return x, np.asarray(second, dtype=np.float64), axis_name, dens, time, cycle


def median_spacing(coord: np.ndarray) -> float:
    if coord.size < 2:
        return 1.0
    diffs = np.diff(coord)
    return float(np.median(np.abs(diffs[np.isfinite(diffs)])))


def profile_one(path: Path, output_id: str, r_edges: np.ndarray) -> Dict[str, object]:
    x, second, axis_name, dens, time, cycle = read_density_slice(path)
    finite_dens = np.nan_to_num(dens, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    peak_j, peak_i = np.unravel_index(int(np.argmax(finite_dens)), finite_dens.shape)
    peak_x = float(x[peak_i])
    peak_second = float(second[peak_j])
    peak_density = float(dens[peak_j, peak_i])

    r_max = float(r_edges[-1])
    x_mask = np.abs(x - peak_x) <= r_max
    second_mask = np.abs(second - peak_second) <= r_max
    x_sub = x[x_mask]
    second_sub = second[second_mask]
    dens_sub = dens[np.ix_(second_mask, x_mask)]
    dx2 = (x_sub - peak_x) ** 2
    nbins = len(r_edges) - 1
    sums = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)
    chunk_size = 64
    dr = float(r_edges[1] - r_edges[0])

    for start in range(0, second_sub.size, chunk_size):
        stop = min(start + chunk_size, second_sub.size)
        dy2 = (second_sub[start:stop] - peak_second)[:, None] ** 2
        radii = np.sqrt(dx2[None, :] + dy2)
        bins = np.floor(radii / dr).astype(np.int64)
        valid = bins < nbins
        values = dens_sub[start:stop, :]
        finite = np.isfinite(values) & valid
        if not np.any(finite):
            continue
        flat_bins = bins[finite]
        flat_values = values[finite].astype(np.float64, copy=False)
        sums += np.bincount(flat_bins, weights=flat_values, minlength=nbins)[:nbins]
        counts += np.bincount(flat_bins, minlength=nbins)[:nbins].astype(np.int64)

    means = np.full(nbins, np.nan, dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    dx = median_spacing(x)
    dsecond = median_spacing(second)
    r2 = (x_sub[None, :] - peak_x) ** 2 + (second_sub[:, None] - peak_second) ** 2
    inside = r2 <= r_max * r_max
    slice_integral = float(np.nansum(np.where(inside, dens_sub, 0.0)) * dx * dsecond)

    return {
        "path": str(path),
        "output_id": output_id,
        "plane": f"x{axis_name}",
        "axis_name": axis_name,
        "output_number": output_number(path),
        "time": time,
        "cycle": cycle,
        "peak_x": peak_x,
        "peak_second": peak_second,
        "peak_density": peak_density,
        "slice_integral_inside_rmax": slice_integral,
        "r_center": 0.5 * (r_edges[:-1] + r_edges[1:]),
        "rho_mean": means,
        "counts": counts,
    }


def write_profiles(path: Path, case: str, profiles: List[Dict[str, object]]) -> None:
    columns = [
        "case",
        "output_id",
        "plane",
        "output_number",
        "time",
        "cycle",
        "peak_x",
        "peak_second",
        "peak_density",
        "r_center",
        "rho_mean",
        "count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for profile in profiles:
            r_center = np.asarray(profile["r_center"])
            rho_mean = np.asarray(profile["rho_mean"])
            counts = np.asarray(profile["counts"])
            for radius, rho, count in zip(r_center, rho_mean, counts):
                writer.writerow(
                    {
                        "case": case,
                        "output_id": profile["output_id"],
                        "plane": profile["plane"],
                        "output_number": profile["output_number"],
                        "time": profile["time"],
                        "cycle": profile["cycle"],
                        "peak_x": profile["peak_x"],
                        "peak_second": profile["peak_second"],
                        "peak_density": profile["peak_density"],
                        "r_center": float(radius),
                        "rho_mean": float(rho),
                        "count": int(count),
                    }
                )


def write_summary(path: Path, case: str, profiles: List[Dict[str, object]]) -> None:
    columns = [
        "case",
        "output_id",
        "plane",
        "output_number",
        "time",
        "cycle",
        "peak_x",
        "peak_second",
        "peak_density",
        "slice_integral_inside_rmax",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for profile in profiles:
            writer.writerow({column: profile[column] if column != "case" else case for column in columns})


def plot_overlay(path: Path, title: str, profiles: List[Dict[str, object]], rho_floor: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    denom = max(len(profiles) - 1, 1)
    for idx, profile in enumerate(profiles):
        rho = np.clip(np.asarray(profile["rho_mean"], dtype=np.float64), rho_floor, None)
        ax.plot(
            profile["r_center"],
            rho,
            color=cmap(idx / denom),
            lw=1.4,
            label=f"t={float(profile['time']):.3g}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("r from density maximum on slice")
    ax.set_ylabel("slice annulus-mean density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(path: Path, profiles: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for profile in profiles:
        grouped.setdefault(str(profile["output_id"]), []).append(profile)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for output_id, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: float(item["time"]))
        times = np.asarray([profile["time"] for profile in group], dtype=np.float64)
        peak_density = np.asarray([profile["peak_density"] for profile in group], dtype=np.float64)
        peak_x = np.asarray([profile["peak_x"] for profile in group], dtype=np.float64)
        integral = np.asarray([profile["slice_integral_inside_rmax"] for profile in group], dtype=np.float64)
        axes[0].plot(times, peak_density, marker="o", label=output_id)
        axes[1].plot(times, peak_x, marker="o", label=output_id)
        axes[2].plot(times, integral, marker="o", label=output_id)
    axes[0].set_ylabel("peak density")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("peak x")
    axes[2].set_ylabel("slice integral")
    axes[2].set_xlabel("time")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute density profiles from AthenaK 2D XY/XZ binary slice outputs."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post"),
    )
    parser.add_argument("--ids", nargs="+", default=["xy_mhd", "xz_mhd"], help="Slice output ids to process.")
    parser.add_argument("--r-max", type=float, default=2.0)
    parser.add_argument("--dr", type=float, default=0.0125)
    parser.add_argument("--rho-floor", type=float, default=1.0e-16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edges = np.arange(0.0, args.r_max + 0.5 * args.dr, args.dr, dtype=np.float64)
    if len(edges) < 2:
        raise ValueError("r-max/dr must define at least one radial bin")

    profiles: List[Dict[str, object]] = []
    for output_id in args.ids:
        files = find_files(args.run_dir, output_id)
        if not files:
            print(f"No *.{output_id}.*.bin files found under {args.run_dir}", file=sys.stderr)
            continue
        for path in files:
            print(f"reading {path}")
            profiles.append(profile_one(path, output_id, edges))

    if not profiles:
        raise FileNotFoundError(f"No requested slice output files found under {args.run_dir}")
    profiles.sort(key=lambda item: (str(item["output_id"]), float(item["time"]), str(item["output_number"])))

    out_dir = args.output_root / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    write_profiles(out_dir / "slice_radial_density_profiles.csv", args.case, profiles)
    write_summary(out_dir / "slice_radial_density_summary.csv", args.case, profiles)

    for output_id in args.ids:
        group = [profile for profile in profiles if profile["output_id"] == output_id]
        if group:
            plot_overlay(
                out_dir / f"{output_id}_slice_radial_density_profile_overlay.png",
                f"{args.case} {output_id}",
                group,
                args.rho_floor,
            )
    plot_summary(out_dir / "slice_radial_density_summary.png", profiles)

    metadata = {
        "case": args.case,
        "run_dir": str(args.run_dir),
        "ids": args.ids,
        "r_max": args.r_max,
        "dr": args.dr,
        "center_mode": "density maximum per output",
        "note": "Profiles are annulus means on 2D slices, not 3D shell averages.",
        "n_profiles": len(profiles),
        "files": [profile["path"] for profile in profiles],
    }
    (out_dir / "slice_radial_density_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(out_dir / "slice_radial_density_summary.csv")
    print(out_dir / "xy_mhd_slice_radial_density_profile_overlay.png")
    print(out_dir / "xz_mhd_slice_radial_density_profile_overlay.png")


if __name__ == "__main__":
    main()
