#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import json
import time
from typing import List, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

DEBUG_LOG_PATH = Path("/home/hzhu/athenak_tde/.cursor/debug-c11e65.log")
DEBUG_SESSION_ID = "c11e65"


def agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass
    # endregion


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_bin_convert():
    vis_dir = repo_root() / "vis" / "python"
    if str(vis_dir) not in sys.path:
        sys.path.insert(0, str(vis_dir))
    import bin_convert  # type: ignore

    return bin_convert


def unique_sorted(paths: List[Path]) -> List[Path]:
    seen: Set[Path] = set()
    result: List[Path] = []
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def find_xy_files(input_dir: Path, output_id: str) -> List[Path]:
    patterns = [
        input_dir / "bin",
        input_dir,
        input_dir / "bin" / "rank_00000000",
        input_dir / "rank_00000000",
    ]
    files: List[Path] = []
    for base in patterns:
        files.extend(base.glob(f"*.{output_id}.*.bin"))
    if not files:
        files.extend(
            path
            for path in input_dir.rglob(f"*.{output_id}.*.bin")
            if "rank_" not in path.parent.name or path.parent.name == "rank_00000000"
        )
    return unique_sorted(files)


def uses_rank_dirs(path: Path) -> bool:
    return "rank_00000000" in path.parts


def read_slice(path: Path, quantities: List[str]) -> dict:
    bin_convert = load_bin_convert()
    reader = (
        bin_convert.read_all_ranks_binary_as_athdf
        if uses_rank_dirs(path)
        else bin_convert.read_binary_as_athdf
    )
    return reader(str(path), quantities=quantities)


def read_meshblock_bounds(path: Path) -> np.ndarray:
    bin_convert = load_bin_convert()
    reader = (
        bin_convert.read_all_ranks_binary
        if uses_rank_dirs(path)
        else bin_convert.read_binary
    )
    filedata = reader(str(path))
    geometry = np.asarray(filedata["mb_geometry"])
    if geometry.size == 0:
        return np.empty((0, 4))
    return geometry[:, [0, 1, 2, 3]]


def read_raw_filedata(path: Path) -> dict:
    bin_convert = load_bin_convert()
    reader = bin_convert.read_all_ranks_binary if uses_rank_dirs(path) else bin_convert.read_binary
    return reader(str(path))


def as_xy(data: dict, quantity: str) -> np.ndarray:
    arr = np.asarray(data[quantity])
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected {quantity} to be a 2D XY slice after squeeze, got {arr.shape}")
    return arr


def load_quantity(path: Path, quantity: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if quantity == "bmag":
        data = read_slice(path, ["dens", "bcc1", "bcc2", "bcc3"])
        field = np.sqrt(as_xy(data, "bcc1")**2 + as_xy(data, "bcc2")**2 + as_xy(data, "bcc3")**2)
    else:
        quantities = ["dens"] if quantity == "dens" else ["dens", quantity]
        data = read_slice(path, quantities)
        field = as_xy(data, quantity)
    dens = as_xy(data, "dens")
    return np.asarray(data["x1v"]), np.asarray(data["x2v"]), field, dens


def display_field(field: np.ndarray, use_log10: bool, floor: float) -> np.ndarray:
    if use_log10:
        return np.log10(np.clip(field, floor, None))
    return field


def display_limits(vmin: float, vmax: float, use_log10: bool) -> Tuple[float, float]:
    if use_log10:
        return np.log10(vmin), np.log10(vmax)
    return vmin, vmax


def symmetry_debug_metrics(path: Path, x: np.ndarray, y: np.ndarray, dens: np.ndarray, x_peak: float, y_peak: float) -> None:
    finite = np.isfinite(dens)
    x_window = np.abs(x - x_peak) <= 2.0
    y_window = np.abs(y) <= 2.0
    window = finite[np.ix_(y_window, x_window)]
    dens_window = dens[np.ix_(y_window, x_window)]
    y_sub = y[y_window]
    if dens_window.size == 0:
        return
    pair_diffs = []
    pair_scales = []
    max_pair = None
    x_sub = x[x_window]
    for j, yv in enumerate(y_sub):
        jm = int(np.argmin(np.abs(y_sub + yv)))
        if j >= jm:
            continue
        upper = dens_window[j, :]
        lower = dens_window[jm, :]
        valid = np.isfinite(upper) & np.isfinite(lower)
        if np.any(valid):
            diff = np.abs(upper[valid] - lower[valid])
            scale = np.maximum(np.abs(upper[valid]) + np.abs(lower[valid]), 1.0e-300)
            pair_diffs.append(float(np.max(diff)))
            pair_scales.append(float(np.max(diff / scale)))
            local = int(np.argmax(diff))
            valid_indices = np.flatnonzero(valid)
            ix = int(valid_indices[local])
            this_diff = float(diff[local])
            if max_pair is None or this_diff > max_pair["diff"]:
                max_pair = {
                    "diff": this_diff,
                    "relative": float(diff[local] / scale[local]),
                    "x": float(x_sub[ix]),
                    "y_a": float(y_sub[j]),
                    "y_b": float(y_sub[jm]),
                    "density_a": float(upper[ix]),
                    "density_b": float(lower[ix]),
                }
    near_x = int(np.argmin(np.abs(x - x_peak)))
    above_y = int(np.argmin(np.where(y > 0.0, y, np.inf)))
    below_y = int(np.argmin(np.where(y < 0.0, -y, np.inf)))
    raw = read_raw_filedata(path)
    geometry = np.asarray(raw["mb_geometry"])
    logical = np.asarray(raw["mb_logical"])
    crosses_y0 = int(np.count_nonzero((geometry[:, 2] <= 0.0) & (geometry[:, 3] >= 0.0))) if geometry.size else 0
    peak_blocks = (
        (geometry[:, 0] <= x_peak) & (geometry[:, 1] >= x_peak) &
        (geometry[:, 2] <= y_peak) & (geometry[:, 3] >= y_peak)
    ) if geometry.size else np.asarray([], dtype=bool)
    agent_debug_log(
        "plot-symmetry",
        "H1,H2,H4,H5",
        "analysis/tde_star_profile/aurora/plot_xy_mhd_shape.py:symmetry_debug_metrics",
        "assembled density symmetry and raw meshblock coverage",
        {
            "file": str(path),
            "time": float(raw.get("time", np.nan)),
            "cycle": int(raw.get("cycle", -1)),
            "shape": list(dens.shape),
            "x_peak": x_peak,
            "y_peak": y_peak,
            "window_max_density": float(np.nanmax(dens_window)),
            "window_min_density": float(np.nanmin(dens_window)),
            "max_abs_y_mirror_diff": max(pair_diffs) if pair_diffs else None,
            "max_relative_y_mirror_diff": max(pair_scales) if pair_scales else None,
            "density_just_above_y0_at_peak_x": float(dens[above_y, near_x]),
            "density_just_below_y0_at_peak_x": float(dens[below_y, near_x]),
            "above_y": float(y[above_y]),
            "below_y": float(y[below_y]),
            "n_meshblocks": int(raw.get("n_mbs", 0)),
            "crosses_y0_blocks": crosses_y0,
            "peak_covering_blocks": int(np.count_nonzero(peak_blocks)),
            "peak_covering_levels": sorted(set(int(v) for v in logical[peak_blocks, 3])) if geometry.size and np.any(peak_blocks) else [],
        },
    )
    log_mesh_symmetry(path, raw, x_peak)
    if max_pair is not None:
        agent_debug_log(
            "plot-max-asymmetry",
            "H1,H2,H4,H5",
            "analysis/tde_star_profile/aurora/plot_xy_mhd_shape.py:symmetry_debug_metrics",
            "maximum density mirror mismatch location",
            {
                "file": str(path),
                "time": float(raw.get("time", np.nan)),
                "cycle": int(raw.get("cycle", -1)),
                "max_pair": max_pair,
                "sample_a": raw_density_sample(raw, max_pair["x"], max_pair["y_a"]),
                "sample_b": raw_density_sample(raw, max_pair["x"], max_pair["y_b"]),
            },
        )
    if int(raw.get("cycle", -1)) in (0, 400, 4000, 7746):
        log_raw_point_samples(path, raw, x_peak, float(y[above_y]), float(y[below_y]))


def log_mesh_symmetry(path: Path, raw: dict, x_peak: float) -> None:
    geometry = np.asarray(raw["mb_geometry"])
    logical = np.asarray(raw["mb_logical"])
    if geometry.size == 0:
        return
    near = (
        (geometry[:, 1] >= x_peak - 2.0) & (geometry[:, 0] <= x_peak + 2.0) &
        (geometry[:, 3] >= -2.0) & (geometry[:, 2] <= 2.0)
    )
    top = near & (geometry[:, 2] >= 0.0)
    bottom = near & (geometry[:, 3] <= 0.0)
    cross = near & (geometry[:, 2] < 0.0) & (geometry[:, 3] > 0.0)
    level_counts_top = {}
    level_counts_bottom = {}
    for lev in sorted(set(int(v) for v in logical[near, 3])) if np.any(near) else []:
        level_counts_top[str(lev)] = int(np.count_nonzero(top & (logical[:, 3] == lev)))
        level_counts_bottom[str(lev)] = int(np.count_nonzero(bottom & (logical[:, 3] == lev)))
    agent_debug_log(
        "plot-mesh-symmetry",
        "H4",
        "analysis/tde_star_profile/aurora/plot_xy_mhd_shape.py:log_mesh_symmetry",
        "meshblock level symmetry near star",
        {
            "file": str(path),
            "time": float(raw.get("time", np.nan)),
            "cycle": int(raw.get("cycle", -1)),
            "x_peak": x_peak,
            "near_blocks": int(np.count_nonzero(near)),
            "top_blocks": int(np.count_nonzero(top)),
            "bottom_blocks": int(np.count_nonzero(bottom)),
            "cross_y0_blocks": int(np.count_nonzero(cross)),
            "top_level_counts": level_counts_top,
            "bottom_level_counts": level_counts_bottom,
        },
    )


def raw_density_sample(raw: dict, x_target: float, y_target: float) -> dict:
    geometry = np.asarray(raw["mb_geometry"])
    logical = np.asarray(raw["mb_logical"])
    dens_blocks = np.asarray(raw["mb_data"]["dens"])
    candidates = np.where(
        (geometry[:, 0] <= x_target) & (x_target <= geometry[:, 1]) &
        (geometry[:, 2] <= y_target) & (y_target <= geometry[:, 3])
    )[0]
    samples = []
    for block_num in candidates:
        block = dens_blocks[block_num]
        nx1 = block.shape[2]
        nx2 = block.shape[1]
        x1min, x1max, x2min, x2max = geometry[block_num, 0:4]
        dx1 = (x1max - x1min) / nx1
        dx2 = (x2max - x2min) / nx2
        i = int(np.clip(np.floor((x_target - x1min) / dx1), 0, nx1 - 1))
        j = int(np.clip(np.floor((y_target - x2min) / dx2), 0, nx2 - 1))
        samples.append(
            {
                "block": int(block_num),
                "logical": [int(v) for v in logical[block_num].tolist()],
                "level": int(logical[block_num, 3]),
                "bounds": [float(v) for v in geometry[block_num, 0:4].tolist()],
                "i": i,
                "j": j,
                "x_center": float(x1min + (i + 0.5) * dx1),
                "y_center": float(x2min + (j + 0.5) * dx2),
                "density": float(block[0, j, i]),
            }
        )
    samples.sort(key=lambda item: (-item["level"], item["block"]))
    return {"target": [x_target, y_target], "samples": samples[:4], "n_candidates": int(len(candidates))}


def log_raw_point_samples(path: Path, raw: dict, x_peak: float, above_y: float, below_y: float) -> None:
    # region agent log
    above = raw_density_sample(raw, x_peak, above_y)
    below = raw_density_sample(raw, x_peak, below_y)
    agent_debug_log(
        "plot-raw-samples",
        "H1,H2,H4",
        "analysis/tde_star_profile/aurora/plot_xy_mhd_shape.py:log_raw_point_samples",
        "raw meshblock samples bracketing y=0 at peak x",
        {
            "file": str(path),
            "time": float(raw.get("time", np.nan)),
            "cycle": int(raw.get("cycle", -1)),
            "x_peak": x_peak,
            "above": above,
            "below": below,
        },
    )
    # endregion


def overlay_meshblocks(ax, bounds: np.ndarray) -> None:
    segments = []
    for x1min, x1max, x2min, x2max in bounds:
        segments.extend(
            [
                [(x1min, x2min), (x1max, x2min)],
                [(x1max, x2min), (x1max, x2max)],
                [(x1max, x2max), (x1min, x2max)],
                [(x1min, x2max), (x1min, x2min)],
            ]
        )
    if segments:
        ax.add_collection(LineCollection(segments, colors="white", linewidths=0.25, alpha=0.7))


def plot_one(
    path: Path,
    output_path: Path,
    quantity: str,
    use_log10: bool,
    floor: float,
    vmin: float,
    vmax: float,
    show_mesh: bool,
    zoom_half_width: float,
    dpi: int,
) -> None:
    x, y, field, dens = load_quantity(path, quantity)
    z = display_field(field, use_log10, floor)
    dens_for_center = np.nan_to_num(dens, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    max_j, max_i = np.unravel_index(np.argmax(dens_for_center), dens_for_center.shape)
    x_peak = float(x[max_i])
    y_peak = float(y[max_j])
    symmetry_debug_metrics(path, x, y, dens, x_peak, y_peak)
    label = f"log10({quantity})" if use_log10 else quantity

    plot_vmin = plot_vmax = None
    if vmin is not None and vmax is not None:
        plot_vmin, plot_vmax = display_limits(vmin, vmax, use_log10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, title in zip(axes, ["Full domain", "Density-maximum zoom"]):
        mesh = ax.pcolormesh(x, y, z, shading="auto", vmin=plot_vmin, vmax=plot_vmax)
        ax.plot([x_peak], [y_peak], marker="+", color="white", markersize=10, mew=1.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax, label=label, shrink=0.9)

    if show_mesh:
        overlay_meshblocks(axes[0], read_meshblock_bounds(path))

    axes[1].set_xlim(x_peak - zoom_half_width, x_peak + zoom_half_width)
    axes[1].set_ylim(y_peak - zoom_half_width, y_peak + zoom_half_width)
    fig.suptitle(f"{path.name}  peak=({x_peak:.6g}, {y_peak:.6g})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-domain and star-zoom PNGs from AthenaK xy_mhd binary slices."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Run directory to scan.")
    parser.add_argument("--case", required=True, help="Case name used below output-root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post"),
    )
    parser.add_argument("--id", default="xy_mhd", help="AthenaK output id to scan.")
    parser.add_argument("--quantity", default="dens", help="dens, bcc1, bcc2, bcc3, or bmag.")
    parser.add_argument("--log10", action="store_true", default=True, help="Plot log10(quantity).")
    parser.add_argument("--linear", action="store_false", dest="log10", help="Plot quantity directly.")
    parser.add_argument("--floor", type=float, default=1.0e-30, help="Positive floor for log10 plots.")
    parser.add_argument("--vmin", type=float, default=None, help="Lower color limit in quantity units.")
    parser.add_argument("--vmax", type=float, default=None, help="Upper color limit in quantity units.")
    parser.add_argument("--show-mesh", action="store_true", help="Overlay meshblock boundaries on full-domain panel.")
    parser.add_argument("--zoom-half-width", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = find_xy_files(args.input_dir, args.id)
    if not files:
        raise FileNotFoundError(f"No *.{args.id}.*.bin files found under {args.input_dir}")

    output_dir = args.output_root / args.case
    for path in files:
        output_path = output_dir / f"{path.stem}_{args.quantity}.png"
        print(f"{path} -> {output_path}")
        plot_one(
            path,
            output_path,
            args.quantity,
            args.log10,
            args.floor,
            args.vmin,
            args.vmax,
            args.show_mesh,
            args.zoom_half_width,
            args.dpi,
        )


if __name__ == "__main__":
    main()
