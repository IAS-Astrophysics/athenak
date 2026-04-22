from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_bin_convert():
    vis_dir = repo_root() / "vis" / "python"
    if str(vis_dir) not in sys.path:
        sys.path.insert(0, str(vis_dir))
    import bin_convert  # type: ignore

    return bin_convert


def available_run_dirs() -> list[Path]:
    base = repo_root() / "analysis" / "z4c_tov_ks_xy_plane"
    candidates = [base / "run_short", base / "run_t005", base / "run_medium", base / "run"]
    return [path for path in candidates if path.exists()]


def select_run_dir(preferred: str | None = None) -> Path:
    runs = available_run_dirs()
    if not runs:
        raise FileNotFoundError("No z4c_tov_ks_xy_plane run directories found.")
    if preferred is not None:
        target = repo_root() / "analysis" / "z4c_tov_ks_xy_plane" / preferred
        if target.exists():
            return target
    return runs[0]


def plane_files(run_dir: Path, kind: str) -> list[Path]:
    return sorted((run_dir / "bin").glob(f"z4c_tov_ks_xy.xy_{kind}.*.bin"))


def dump_numbers(run_dir: Path, kind: str = "mhd") -> list[int]:
    dumps = []
    for path in plane_files(run_dir, kind):
        dumps.append(int(path.stem.split(".")[-1]))
    return dumps


def read_plane(run_dir: Path, kind: str, dump: int | None = None) -> dict:
    files = plane_files(run_dir, kind)
    if not files:
        raise FileNotFoundError(f"No xy-plane {kind} files found in {run_dir}")
    if dump is None:
        path = files[-1]
    else:
        matches = [path for path in files if path.stem.endswith(f"{dump:05d}")]
        if not matches:
            raise FileNotFoundError(f"Could not find dump {dump:05d} for {kind} in {run_dir}")
        path = matches[0]
    bin_convert = _load_bin_convert()
    return bin_convert.read_binary_as_athdf(str(path))


def _history_names(header_line: str) -> list[str]:
    names = []
    for token in header_line.split():
        if "]=" in token:
            names.append(token.split("]=")[1])
    return names


def read_history(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"History file {path} is too short")
    names = _history_names(lines[1])
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return pd.DataFrame(data, columns=names)


def grid_xy(data: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(data["x1v"], data["x2v"], indexing="xy")


def cell_centered(data: dict, field: str) -> np.ndarray:
    arr = np.asarray(data[field])
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"Expected a 2D slice stored as shape (1, ny, nx) for {field}")
    return arr[0]


def plot_history(mhd_hist: pd.DataFrame, z4c_hist: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    axes[0].plot(mhd_hist["time"], mhd_hist["mass"], marker="o")
    axes[0].set_title("Rest Mass")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("mass")

    axes[1].plot(mhd_hist["time"], mhd_hist["tot-E"], marker="o", label="tot-E")
    axes[1].set_title("Total Energy")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("energy")

    axes[2].plot(z4c_hist["time"], z4c_hist["H-norm2"], marker="o", label="H")
    axes[2].plot(z4c_hist["time"], z4c_hist["C-norm2"], marker="s", label="C")
    axes[2].plot(z4c_hist["time"], z4c_hist["M-norm2"], marker="^", label="M")
    axes[2].set_title("Constraint Norms")
    axes[2].set_xlabel("time")
    axes[2].set_yscale("log")
    axes[2].legend()

    return fig, axes


def plot_plane_overview(
    mhd: dict,
    adm: dict,
    z4c: dict,
    star_center_x: float = 8.0,
    bh_center_x: float = 0.0,
    star_center_y: float = 0.0,
) -> tuple[plt.Figure, np.ndarray]:
    x1, x2 = grid_xy(mhd)

    panels = [
        ("log10 dens", np.log10(np.clip(cell_centered(mhd, "dens"), 1.0e-30, None)), "viridis"),
        ("press", cell_centered(mhd, "press"), "magma"),
        ("z4c_alpha", cell_centered(z4c, "z4c_alpha"), "cividis"),
        ("adm_psi4", cell_centered(adm, "adm_psi4"), "plasma"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (title, field, cmap) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(x1, x2, field, shading="nearest", cmap=cmap)
        ax.plot([bh_center_x], [0.0], marker="x", color="white", markersize=8, mew=2)
        ax.plot([star_center_x], [star_center_y], marker="+", color="white", markersize=10, mew=2)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax, shrink=0.88)

    return fig, axes


def plot_star_zoom(
    mhd: dict,
    z4c: dict,
    star_center_x: float = 8.0,
    half_width: float = 2.5,
) -> tuple[plt.Figure, np.ndarray]:
    x1, x2 = grid_xy(mhd)
    dens = np.log10(np.clip(cell_centered(mhd, "dens"), 1.0e-30, None))
    chi = cell_centered(z4c, "z4c_chi")
    theta = cell_centered(z4c, "z4c_Theta")

    panels = [
        ("log10 dens", dens, "viridis"),
        ("z4c_chi", chi, "cividis"),
        ("z4c_Theta", theta, "coolwarm"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (title, field, cmap) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(x1, x2, field, shading="nearest", cmap=cmap)
        ax.plot([star_center_x], [0.0], marker="+", color="white", markersize=10, mew=2)
        ax.set_xlim(star_center_x - half_width, star_center_x + half_width)
        ax.set_ylim(-half_width, half_width)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax, shrink=0.88)

    return fig, axes
