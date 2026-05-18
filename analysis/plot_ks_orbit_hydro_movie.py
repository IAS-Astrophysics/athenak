#!/usr/bin/env python3
"""Make xy-slice diagnostics for the non-magnetized circular-orbit TOV test."""

from __future__ import annotations

import argparse
import math
import re
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory or its bin directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--zoom-width", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pixels", type=float, default=8.0e6)
    parser.add_argument("--orbit-radius", type=float, default=None)
    parser.add_argument("--orbit-omega", type=float, default=None)
    parser.add_argument("--orbit-phase", type=float, default=0.0)
    return parser.parse_args()


def bin_dir_for(path: Path) -> Path:
    path = path.resolve()
    if any(path.glob("*.xy_mhd.*.bin")):
        return path
    if (path / "bin").is_dir():
        return path / "bin"
    return path


def files_for(bin_dir: Path) -> list[Path]:
    files = sorted(bin_dir.glob("*.xy_mhd.*.bin"))
    if not files:
        files = sorted(bin_dir.glob("*xy_mhd*.bin"))
    if not files:
        raise FileNotFoundError(f"No xy_mhd binary files found in {bin_dir}")
    return files


def read_slice(path: Path) -> dict:
    quantities = ["dens", "press", "velx", "vely", "velz"]
    return bin_convert.read_binary_as_athdf(str(path), quantities=quantities, dtype=np.float32)


def slice2d(data: dict, name: str) -> np.ndarray:
    arr = np.asarray(data[name])
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Expected {name} to have shape (1, ny, nx), got {arr.shape}")
    return arr[0]


def safe_log10(values: np.ndarray, floor: float) -> np.ndarray:
    return np.log10(np.clip(values, floor, None))


def four_velocity_magnitude(data: dict) -> np.ndarray:
    ux = slice2d(data, "velx")
    uy = slice2d(data, "vely")
    uz = slice2d(data, "velz")
    return np.sqrt(np.maximum(ux*ux + uy*uy + uz*uz, 0.0))


def star_center(data: dict) -> tuple[float, float]:
    dens = slice2d(data, "dens")
    j, i = np.unravel_index(np.nanargmax(dens), dens.shape)
    return float(data["x1v"][i]), float(data["x2v"][j])


def parse_orbit_from_log(run_dir: Path) -> tuple[float | None, float | None]:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return None, None
    radius = None
    omega = None
    for line in log_path.read_text(errors="replace").splitlines():
        if match := re.search(r"Orbit radius:\s*([-+0-9.eE]+)", line):
            radius = float(match.group(1))
        if match := re.search(r"Omega=dphi/dt:\s*([-+0-9.eE]+)", line):
            omega = float(match.group(1))
    return radius, omega


def expected_center(time: float, radius: float, omega: float, phase: float) -> tuple[float, float]:
    phi = phase + omega*time
    return radius*math.cos(phi), radius*math.sin(phi)


def collect(paths: list[Path]) -> tuple[dict[str, tuple[float, float]], list[tuple[float, float]], list[float]]:
    mins = {"rho": math.inf, "press": math.inf, "umag": math.inf}
    maxs = {"rho": -math.inf, "press": -math.inf, "umag": -math.inf}
    centers: list[tuple[float, float]] = []
    times: list[float] = []

    for path in paths:
        data = read_slice(path)
        rho = safe_log10(slice2d(data, "dens"), 1.0e-30)
        press = safe_log10(slice2d(data, "press"), 1.0e-40)
        umag = four_velocity_magnitude(data)
        for key, field in (("rho", rho), ("press", press), ("umag", umag)):
            finite = np.isfinite(field)
            if np.any(finite):
                mins[key] = min(mins[key], float(np.nanpercentile(field[finite], 0.5)))
                maxs[key] = max(maxs[key], float(np.nanpercentile(field[finite], 99.8)))
        centers.append(star_center(data))
        times.append(float(data["Time"]))

    return {key: (mins[key], maxs[key]) for key in mins}, centers, times


def stride_for(shape: tuple[int, int], max_pixels: float) -> int:
    pixels = shape[0]*shape[1]
    if pixels <= max_pixels:
        return 1
    return int(math.ceil(math.sqrt(pixels/max_pixels)))


def decimate(field: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return field
    return field[::stride, ::stride]


def draw_image(ax, x: np.ndarray, y: np.ndarray, field: np.ndarray, title: str,
               cmap: str, clim: tuple[float, float], center: tuple[float, float],
               shift: bool, stride: int):
    xs = x[::stride]
    ys = y[::stride]
    plot_field = decimate(field, stride)
    cx, cy = center
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    if shift:
        extent = [extent[0] - cx, extent[1] - cx, extent[2] - cy, extent[3] - cy]
    image = ax.imshow(plot_field, origin="lower", extent=extent, cmap=cmap,
                      vmin=clim[0], vmax=clim[1], interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x [M]" if not shift else "x - x_star [M]")
    ax.set_ylabel("y [M]" if not shift else "y - y_star [M]")
    ax.set_aspect("equal", adjustable="box")
    return image


def overlay_orbit(ax, times: list[float], centers: list[tuple[float, float]], frame_index: int,
                  radius: float | None, omega: float | None, phase: float, shift: bool,
                  current_center: tuple[float, float]) -> None:
    cx, cy = current_center
    if radius is not None and omega is not None:
        theta = np.linspace(phase, phase + 2.0*np.pi, 720)
        ox = radius*np.cos(theta)
        oy = radius*np.sin(theta)
        if shift:
            ox = ox - cx
            oy = oy - cy
        ax.plot(ox, oy, "--", color="white", linewidth=0.9, alpha=0.65, label="geodesic")

        past_t = np.asarray(times[:frame_index + 1])
        ex = radius*np.cos(phase + omega*past_t)
        ey = radius*np.sin(phase + omega*past_t)
        if shift:
            ex = ex - cx
            ey = ey - cy
        ax.plot(ex, ey, color="#ffcc33", linewidth=1.6, label="geodesic trail")
        ax.plot(ex[-1], ey[-1], marker="o", color="#ffcc33", markersize=4)

    trail = np.asarray(centers[:frame_index + 1])
    mx = trail[:, 0]
    my = trail[:, 1]
    if shift:
        mx = mx - cx
        my = my - cy
    ax.plot(mx, my, color="#40e0d0", linewidth=1.4, label="density-peak trail")
    ax.plot(mx[-1], my[-1], marker="+", color="#40e0d0", markersize=8, markeredgewidth=1.5)
    if not shift:
        ax.plot(0.0, 0.0, marker="x", color="black", markersize=7, markeredgewidth=1.5)


def make_frame(path: Path, frame_path: Path, limits: dict[str, tuple[float, float]],
               centers: list[tuple[float, float]], times: list[float], frame_index: int,
               radius: float | None, omega: float | None, phase: float,
               zoom_width: float, max_pixels: float, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_slice(path)
    x = np.asarray(data["x1v"])
    y = np.asarray(data["x2v"])
    center = centers[frame_index]
    rho = safe_log10(slice2d(data, "dens"), 1.0e-30)
    press = safe_log10(slice2d(data, "press"), 1.0e-40)
    umag = four_velocity_magnitude(data)
    stride = stride_for(rho.shape, max_pixels)

    fields = [
        (rho, "log10 density", "viridis", limits["rho"]),
        (press, "log10 pressure", "magma", limits["press"]),
        (umag, "|u^i|", "cividis", limits["umag"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.4), constrained_layout=True)
    for col, (field, title, cmap, clim) in enumerate(fields):
        image = draw_image(axes[0, col], x, y, field, f"Global {title}", cmap, clim,
                           center, False, stride)
        overlay_orbit(axes[0, col], times, centers, frame_index, radius, omega,
                      phase, False, center)
        axes[0, col].set_xlim(float(x[0]), float(x[-1]))
        axes[0, col].set_ylim(float(y[0]), float(y[-1]))
        fig.colorbar(image, ax=axes[0, col], shrink=0.82)

        image = draw_image(axes[1, col], x, y, field, f"Star-centered {title}", cmap, clim,
                           center, True, stride)
        overlay_orbit(axes[1, col], times, centers, frame_index, radius, omega,
                      phase, True, center)
        half = 0.5*zoom_width
        axes[1, col].set_xlim(-half, half)
        axes[1, col].set_ylim(-half, half)
        fig.colorbar(image, ax=axes[1, col], shrink=0.82)

    if radius is not None and omega is not None:
        ex, ey = expected_center(times[frame_index], radius, omega, phase)
        dx = center[0] - ex
        dy = center[1] - ey
        offset = f"density-peak minus geodesic = ({dx:.3g}, {dy:.3g}) M"
    else:
        offset = "geodesic metadata not found"
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.75)
    fig.suptitle(
        f"{path.name}    t = {times[frame_index]:.6g} M    "
        f"density peak = ({center[0]:.3f}, {center[1]:.3f}) M    {offset}"
    )
    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)


def encode_movie(frames_dir: Path, movie_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(movie_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    bin_dir = bin_dir_for(args.run_dir)
    run_dir = bin_dir.parent
    output_dir = (args.output_dir or (run_dir / "viz_ks_orbit_hydro")).resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    movie_path = output_dir / "ks_orbit_hydro_shape.mp4"

    paths = files_for(bin_dir)
    radius, omega = parse_orbit_from_log(run_dir)
    if args.orbit_radius is not None:
        radius = args.orbit_radius
    if args.orbit_omega is not None:
        omega = args.orbit_omega

    limits, centers, times = collect(paths)
    for frame_index, path in enumerate(paths):
        frame_path = frames_dir / f"frame_{frame_index:05d}.png"
        print(f"writing {frame_path}")
        make_frame(path, frame_path, limits, centers, times, frame_index, radius, omega,
                   args.orbit_phase, args.zoom_width, args.max_pixels, args.dpi)

    print(f"encoding {movie_path}")
    encode_movie(frames_dir, movie_path, args.fps)
    print(movie_path)


if __name__ == "__main__":
    main()
