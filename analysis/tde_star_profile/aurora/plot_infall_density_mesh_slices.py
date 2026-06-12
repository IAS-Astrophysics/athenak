#!/usr/bin/env python3
"""Plot xy density slices in grid and star-centered frames with meshblocks."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
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


def frame_number(path: Path) -> int:
    match = re.search(r"\.(\d+)\.bin$", path.name)
    return int(match.group(1)) if match else -1


def find_slice_files(run_dir: Path, output_id: str):
    candidates = [
        run_dir / "bin",
        run_dir,
        run_dir / "bin" / "rank_00000000",
        run_dir / "rank_00000000",
    ]
    files = []
    for base in candidates:
        files.extend(base.glob(f"*.{output_id}.*.bin"))
    if not files:
        files.extend(
            path
            for path in run_dir.rglob(f"*.{output_id}.*.bin")
            if "rank_" not in path.parent.name or path.parent.name == "rank_00000000"
        )
    return sorted(set(files), key=lambda path: (frame_number(path), str(path)))


def uses_rank_dirs(path: Path) -> bool:
    return "rank_00000000" in path.parts


def read_slice(path: Path) -> dict:
    bin_convert = load_bin_convert()
    reader = (
        bin_convert.read_all_ranks_binary_as_athdf
        if uses_rank_dirs(path)
        else bin_convert.read_binary_as_athdf
    )
    return reader(str(path), quantities=["dens"])


def read_raw(path: Path) -> dict:
    bin_convert = load_bin_convert()
    reader = bin_convert.read_all_ranks_binary if uses_rank_dirs(path) else bin_convert.read_binary
    return reader(str(path))


def as_xy_density(data: dict) -> np.ndarray:
    dens = np.squeeze(np.asarray(data["dens"]))
    if dens.ndim != 2:
        raise ValueError(f"Expected a 2D density slice, got shape {dens.shape}")
    return dens


def mesh_segments(bounds: np.ndarray, x_shift: float = 0.0, y_shift: float = 0.0):
    segments = []
    for x1min, x1max, x2min, x2max in bounds:
        x1min -= x_shift
        x1max -= x_shift
        x2min -= y_shift
        x2max -= y_shift
        segments.extend(
            [
                [(x1min, x2min), (x1max, x2min)],
                [(x1max, x2min), (x1max, x2max)],
                [(x1max, x2max), (x1min, x2max)],
                [(x1min, x2max), (x1min, x2min)],
            ]
        )
    return segments


def overlay_mesh(ax, bounds: np.ndarray, x_shift: float = 0.0, y_shift: float = 0.0) -> None:
    segments = mesh_segments(bounds, x_shift=x_shift, y_shift=y_shift)
    if segments:
        ax.add_collection(LineCollection(segments, colors="white", linewidths=0.28, alpha=0.65))


def choose_color_limits(log_density: np.ndarray, user_vmin: Optional[float], user_vmax: Optional[float]):
    finite = log_density[np.isfinite(log_density)]
    if finite.size == 0:
        return -16.0, -3.0
    vmax = float(user_vmax) if user_vmax is not None else float(np.nanmax(finite))
    if user_vmin is not None:
        vmin = float(user_vmin)
    else:
        vmin = max(vmax - 10.0, float(np.nanpercentile(finite, 0.5)))
    if not vmin < vmax:
        vmin = vmax - 1.0
    return vmin, vmax


def plot_density_slice(
    path: Path,
    output_path: Path,
    label: str,
    zoom_half_width: float,
    floor: float,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int,
) -> dict:
    data = read_slice(path)
    raw = read_raw(path)

    x = np.asarray(data["x1v"])
    y = np.asarray(data["x2v"])
    dens = as_xy_density(data)
    if dens.shape != (y.size, x.size):
        raise ValueError(f"Density shape {dens.shape} does not match x/y sizes {(y.size, x.size)}")

    safe_dens = np.nan_to_num(dens, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    peak_j, peak_i = np.unravel_index(np.argmax(safe_dens), safe_dens.shape)
    peak_x = float(x[peak_i])
    peak_y = float(y[peak_j])
    rho_max = float(dens[peak_j, peak_i])
    log_density = np.log10(np.clip(dens, floor, None))
    plot_vmin, plot_vmax = choose_color_limits(log_density, vmin, vmax)

    bounds = np.asarray(raw["mb_geometry"])[:, [0, 1, 2, 3]]
    time = float(raw.get("time", np.nan))
    cycle = int(raw.get("cycle", -1))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
    panels = [
        (axes[0], x, y, "grid frame", 0.0, 0.0),
        (axes[1], x - peak_x, y - peak_y, "star frame", peak_x, peak_y),
    ]
    for ax, x_plot, y_plot, title, x_shift, y_shift in panels:
        mesh = ax.pcolormesh(x_plot, y_plot, log_density, shading="auto", vmin=plot_vmin, vmax=plot_vmax)
        overlay_mesh(ax, bounds, x_shift=x_shift, y_shift=y_shift)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x" if title == "grid frame" else "x - x_peak")
        ax.set_ylabel("y" if title == "grid frame" else "y - y_peak")
        ax.plot([0.0 if title == "star frame" else peak_x], [0.0 if title == "star frame" else peak_y],
                marker="+", color="black", markersize=10, mew=2.0)
        ax.plot([0.0 - x_shift], [0.0 - y_shift], marker="o", color="white", markeredgecolor="black", markersize=4)
        fig.colorbar(mesh, ax=ax, label="log10 density", shrink=0.92)

    axes[1].set_xlim(-zoom_half_width, zoom_half_width)
    axes[1].set_ylim(-zoom_half_width, zoom_half_width)
    fig.suptitle(
        f"{label}: {path.name}\n"
        f"t={time:.6g}, cycle={cycle}, peak=({peak_x:.6g}, {peak_y:.6g}), rho_max={rho_max:.6e}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {
        "input": str(path),
        "output": str(output_path),
        "label": label,
        "time": time,
        "cycle": cycle,
        "frame": frame_number(path),
        "peak_x": peak_x,
        "peak_y": peak_y,
        "rho_max": rho_max,
        "n_meshblocks": int(np.asarray(raw["mb_geometry"]).shape[0]),
        "vmin_log10": plot_vmin,
        "vmax_log10": plot_vmax,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--id", default="xy_mhd")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument(
        "--frames",
        default="",
        help="Comma-separated frame numbers to plot, e.g. 0,25,50. Empty means all selected by stride/latest.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride when not using --latest-only.")
    parser.add_argument("--zoom-half-width", type=float, default=4.0)
    parser.add_argument("--floor", type=float, default=1.0e-30)
    parser.add_argument("--vmin", type=float, default=None, help="Lower color limit in log10 density.")
    parser.add_argument("--vmax", type=float, default=None, help="Upper color limit in log10 density.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = find_slice_files(args.run_dir, args.id)
    if not files:
        raise FileNotFoundError(f"No *.{args.id}.*.bin files found under {args.run_dir}")
    if args.frames:
        wanted = {int(value.strip()) for value in args.frames.split(",") if value.strip()}
        files = [path for path in files if frame_number(path) in wanted]
        missing = sorted(wanted.difference(frame_number(path) for path in files))
        if missing:
            raise FileNotFoundError(f"Requested frames not found: {missing}")
    elif args.latest_only:
        files = [files[-1]]
    elif args.stride > 1:
        files = files[:: args.stride]

    summaries = []
    for path in files:
        output_path = args.output_dir / f"{args.label}_density_mesh_{frame_number(path):05d}.png"
        print(f"{path} -> {output_path}")
        summaries.append(
            plot_density_slice(
                path,
                output_path,
                args.label,
                args.zoom_half_width,
                args.floor,
                args.vmin,
                args.vmax,
                args.dpi,
            )
        )

    summary_path = args.output_dir / f"{args.label}_density_mesh_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
