#!/usr/bin/env python3
"""Make a six-panel xy-slice movie from AthenaK MHD binary outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_PYTHON = REPO_ROOT / "vis" / "python"
if str(VIS_PYTHON) not in sys.path:
    sys.path.insert(0, str(VIS_PYTHON))

import bin_convert  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot density, pressure, and magnetic beta in global and star-centered xy frames."
    )
    parser.add_argument(
        "bin_dir",
        type=Path,
        help="Directory containing AthenaK xy_mhd .bin files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for frames and movie. Defaults to <run>/viz_xy_mhd.",
    )
    parser.add_argument("--fps", type=int, default=3, help="Movie frame rate.")
    parser.add_argument("--zoom-width", type=float, default=10.0, help="Star-centered x width in M.")
    parser.add_argument("--dpi", type=int, default=160, help="Frame DPI.")
    return parser.parse_args()


def files_for(bin_dir: Path) -> list[Path]:
    files = sorted(bin_dir.glob("*.xy_mhd.*.bin"))
    if not files:
        files = sorted(bin_dir.glob("*xy_mhd*.bin"))
    if not files:
        raise FileNotFoundError(f"No xy_mhd binary files found in {bin_dir}")
    return files


def read_slice(path: Path) -> dict:
    quantities = ["dens", "press", "bcc1", "bcc2", "bcc3"]
    return bin_convert.read_binary_as_athdf(str(path), quantities=quantities, dtype=np.float32)


def slice2d(data: dict, name: str) -> np.ndarray:
    arr = np.asarray(data[name])
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Expected {name} to have shape (1, ny, nx), got {arr.shape}")
    return arr[0]


def safe_log10(values: np.ndarray, floor: float) -> np.ndarray:
    return np.log10(np.clip(values, floor, None))


def magnetic_beta(data: dict) -> np.ndarray:
    press = slice2d(data, "press")
    b1 = slice2d(data, "bcc1")
    b2 = slice2d(data, "bcc2")
    b3 = slice2d(data, "bcc3")
    pmag = 0.5 * (b1 * b1 + b2 * b2 + b3 * b3)
    beta = np.full_like(press, np.nan, dtype=np.float32)
    np.divide(press, pmag, out=beta, where=pmag > 0.0)
    return beta


def star_center(data: dict) -> tuple[float, float]:
    dens = slice2d(data, "dens")
    j, i = np.unravel_index(np.nanargmax(dens), dens.shape)
    return float(data["x1v"][i]), float(data["x2v"][j])


def collect_limits(paths: list[Path]) -> tuple[dict[str, tuple[float, float]], list[tuple[float, float]], bool]:
    dens_min, dens_max = np.inf, -np.inf
    press_min, press_max = np.inf, -np.inf
    beta_min, beta_max = np.inf, -np.inf
    centers: list[tuple[float, float]] = []
    finite_beta_found = False

    for path in paths:
        data = read_slice(path)
        dens_log = safe_log10(slice2d(data, "dens"), 1.0e-30)
        press_log = safe_log10(slice2d(data, "press"), 1.0e-40)
        dens_min = min(dens_min, float(np.nanmin(dens_log)))
        dens_max = max(dens_max, float(np.nanmax(dens_log)))
        press_min = min(press_min, float(np.nanmin(press_log)))
        press_max = max(press_max, float(np.nanmax(press_log)))
        centers.append(star_center(data))

        beta = magnetic_beta(data)
        finite = np.isfinite(beta) & (beta > 0.0)
        if np.any(finite):
            finite_beta_found = True
            beta_log = np.log10(beta[finite])
            beta_min = min(beta_min, float(np.nanpercentile(beta_log, 1.0)))
            beta_max = max(beta_max, float(np.nanpercentile(beta_log, 99.0)))

    if not finite_beta_found:
        beta_min, beta_max = -2.0, 8.0

    limits = {
        "density": (dens_min, dens_max),
        "pressure": (press_min, press_max),
        "beta": (beta_min, beta_max),
    }
    return limits, centers, finite_beta_found


def draw_panel(ax, x: np.ndarray, y: np.ndarray, field: np.ndarray, title: str, cmap: str, clim: tuple[float, float]):
    mesh = ax.pcolormesh(x, y, field, shading="nearest", cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax.set_title(title)
    ax.set_xlabel("x [M]")
    ax.set_ylabel("y [M]")
    ax.set_aspect("equal", adjustable="box")
    return mesh


def make_frame(
    path: Path,
    frame_path: Path,
    limits: dict[str, tuple[float, float]],
    center: tuple[float, float],
    zoom_width: float,
    finite_beta_found: bool,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_slice(path)
    x = np.asarray(data["x1v"])
    y = np.asarray(data["x2v"])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    cx, cy = center
    xxc = xx - cx
    yyc = yy - cy

    dens_log = safe_log10(slice2d(data, "dens"), 1.0e-30)
    press_log = safe_log10(slice2d(data, "press"), 1.0e-40)
    beta = magnetic_beta(data)
    beta_log = np.full_like(beta, np.nan, dtype=np.float32)
    finite_beta = np.isfinite(beta) & (beta > 0.0)
    beta_log[finite_beta] = np.log10(beta[finite_beta])
    beta_plot = np.ma.masked_invalid(beta_log)

    cmap_beta = plt.get_cmap("cividis").copy()
    cmap_beta.set_bad("#d9d9d9")

    fields = [
        (dens_log, "log10 density", "viridis", limits["density"]),
        (press_log, "log10 pressure", "magma", limits["pressure"]),
        (beta_plot, "log10 magnetic beta", cmap_beta, limits["beta"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    for col, (field, title, cmap, clim) in enumerate(fields):
        mesh = draw_panel(axes[0, col], xx, yy, field, f"Global {title}", cmap, clim)
        axes[0, col].set_xlim(float(x[0]), float(x[-1]))
        axes[0, col].set_ylim(float(y[0]), float(y[-1]))
        axes[0, col].plot(cx, cy, marker="+", color="white", markersize=9, markeredgewidth=1.7)
        fig.colorbar(mesh, ax=axes[0, col], shrink=0.82)

        mesh = draw_panel(axes[1, col], xxc, yyc, field, f"Star-centered {title}", cmap, clim)
        half = 0.5 * zoom_width
        axes[1, col].set_xlim(-half, half)
        axes[1, col].set_ylim(-half, half)
        axes[1, col].plot(0.0, 0.0, marker="+", color="white", markersize=9, markeredgewidth=1.7)
        fig.colorbar(mesh, ax=axes[1, col], shrink=0.82)

        if col == 2 and not finite_beta_found:
            for row in range(2):
                axes[row, col].text(
                    0.5,
                    0.06,
                    "B = 0 everywhere; beta is infinite/undefined",
                    transform=axes[row, col].transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
                )

    fig.suptitle(f"{path.name}    t = {data['Time']:.6g} M    star center = ({cx:.3f}, {cy:.3f}) M")
    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)


def encode_movie(frames_dir: Path, movie_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(movie_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    bin_dir = args.bin_dir.resolve()
    run_dir = bin_dir.parent
    output_dir = (args.output_dir or (run_dir / "viz_xy_mhd")).resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    movie_path = output_dir / "xy_mhd_density_pressure_beta.mp4"

    paths = files_for(bin_dir)
    limits, centers, finite_beta_found = collect_limits(paths)
    for frame_index, (path, center) in enumerate(zip(paths, centers)):
        frame_path = frames_dir / f"frame_{frame_index:05d}.png"
        print(f"writing {frame_path}")
        make_frame(path, frame_path, limits, center, args.zoom_width, finite_beta_found, args.dpi)

    print(f"encoding {movie_path}")
    encode_movie(frames_dir, movie_path, args.fps)
    print(movie_path)


if __name__ == "__main__":
    main()
