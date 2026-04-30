#!/usr/bin/env python3
"""Run and plot the compact dynbbh ADM beam diagnostic."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_comparisons import column, read_binary_slice, read_tab, read_trk  # noqa: E402


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_input(path: Path) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = defaultdict(dict)
    block = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            block = line.strip("<>")
            continue
        if block and "=" in line:
            key, value = line.split("=", 1)
            blocks[block][key.strip()] = value.strip()
    return blocks


def get_float(blocks: dict[str, dict[str, str]], block: str, key: str, default: float) -> float:
    return float(blocks.get(block, {}).get(key, default))


def run_case(root: Path, run_dir: Path, exe: Path, input_file: Path, basename: str,
             tlim: float, nlim: int, snapshot_dt: float, track_dt: float,
             nx1: int | None, nx2: int | None, nx3: int | None,
             mb_nx1: int | None, mb_nx2: int | None, mb_nx3: int | None,
             nlevel: int | None, ppc: float | None, ntrack: int | None,
             beam_spread: float | None, mpi_ranks: int, skip_run: bool) -> None:
    if skip_run:
        return
    athena_cmd = [
        str(exe),
        "-i",
        str(input_file),
        "-d",
        str(run_dir),
        f"job/basename={basename}",
        f"time/tlim={tlim}",
        f"time/nlim={nlim}",
        f"output1/dt={tlim}",
        f"output2/dt={track_dt}",
        f"output3/dt={snapshot_dt}",
        f"output4/dt={snapshot_dt}",
        f"output5/dt={snapshot_dt}",
        "adm/dynamic=true",
    ]
    overrides = {
        "mesh/nx1": nx1,
        "mesh/nx2": nx2,
        "mesh/nx3": nx3,
        "meshblock/nx1": mb_nx1,
        "meshblock/nx2": mb_nx2,
        "meshblock/nx3": mb_nx3,
        "dyn_radiation/nlevel": nlevel,
        "particles/ppc": ppc,
        "output2/nparticles": ntrack,
        "rad_srcterms/spread": beam_spread,
    }
    for key, value in overrides.items():
        if value is not None:
            athena_cmd.append(f"{key}={value}")
    cmd = athena_cmd
    if mpi_ranks > 1:
        cmd = ["mpirun", "-n", str(mpi_ranks), *athena_cmd]
    env = os.environ.copy()
    if mpi_ranks <= 1:
        env.setdefault("OMPI_MCA_btl", "self")
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, env=env)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "logs" / f"{basename}.out").write_text(result.stdout, encoding="utf-8")
    (run_dir / "logs" / f"{basename}.err").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{basename} failed with exit code {result.returncode}")


def latest(run_dir: Path, basename: str, suffix: str) -> Path:
    matches = sorted(run_dir.glob(f"{basename}{suffix}"))
    for subdir in ("bin", "tab", "trk"):
        matches += sorted((run_dir / subdir).glob(f"{basename}{suffix}"))
    if not matches:
        raise FileNotFoundError(f"missing output matching {basename}{suffix}")
    return sorted(matches)[-1]


def all_outputs(run_dir: Path, basename: str, suffix: str) -> list[Path]:
    matches = sorted(run_dir.glob(f"{basename}{suffix}"))
    for subdir in ("bin", "tab", "trk"):
        matches += sorted((run_dir / subdir).glob(f"{basename}{suffix}"))
    return sorted(set(matches))


def squeeze_slice(data, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    arr = np.asarray(data.variables[variable], dtype=float)
    if arr.ndim == 2:
        return data.x1f, data.x2f, arr, r"$x$", r"$y$"
    if arr.ndim != 3:
        raise ValueError(f"{data.path} variable {variable} has unsupported shape {arr.shape}")
    if arr.shape[0] == 1:
        return data.x1f, data.x2f, arr[0, :, :], r"$x$", r"$y$"
    if arr.shape[1] == 1:
        return data.x1f, data.x3f, arr[:, 0, :], r"$x$", r"$z$"
    if arr.shape[2] == 1:
        return data.x2f, data.x3f, arr[:, :, 0], r"$y$", r"$z$"
    raise ValueError(f"{data.path} is not a 2D slice: {arr.shape}")


def binary_positions(time: float, sep: float, q: float) -> tuple[tuple[float, float],
                                                               tuple[float, float]]:
    omega = sep ** -1.5
    c = float(np.cos(omega * time))
    s = float(np.sin(omega * time))
    r1 = q / (1.0 + q) * sep
    r2 = -sep / (1.0 + q)
    return (r1 * c, r1 * s), (r2 * c, r2 * s)


def draw_xy_guides(ax: plt.Axes, blocks: dict[str, dict[str, str]], time: float) -> None:
    sep = get_float(blocks, "problem", "sep", 4.0)
    q = get_float(blocks, "problem", "q", 1.0)
    bh1, bh2 = binary_positions(time, sep, q)
    ax.scatter([bh1[0], bh2[0]], [bh1[1], bh2[1]], s=48, marker="x",
               color="#21d4e8", linewidths=1.5, label="BH centers")

    p1 = get_float(blocks, "rad_srcterms", "pos_1", -5.5)
    p2 = get_float(blocks, "rad_srcterms", "pos_2", -0.5)
    d1 = get_float(blocks, "rad_srcterms", "dir_1", 1.0)
    d2 = get_float(blocks, "rad_srcterms", "dir_2", 0.08)
    d3 = get_float(blocks, "rad_srcterms", "dir_3", 0.02)
    dnorm = float(np.sqrt(d1 * d1 + d2 * d2 + d3 * d3))
    d1, d2 = d1 / dnorm, d2 / dnorm
    svals = np.linspace(0.0, 14.0, 100)
    ax.plot(p1 + svals * d1, p2 + svals * d2, color="#8bc34a", lw=1.1,
            ls="--", label="coordinate source axis")
    ax.scatter([p1], [p2], s=34, facecolor="none", edgecolor="white", linewidths=1.1)


def draw_xz_guides(ax: plt.Axes, blocks: dict[str, dict[str, str]], time: float) -> None:
    sep = get_float(blocks, "problem", "sep", 4.0)
    q = get_float(blocks, "problem", "q", 1.0)
    bh1, bh2 = binary_positions(time, sep, q)
    ax.scatter([bh1[0], bh2[0]], [0.0, 0.0], s=48, marker="x",
               color="#21d4e8", linewidths=1.5)

    p1 = get_float(blocks, "rad_srcterms", "pos_1", -5.5)
    p3 = get_float(blocks, "rad_srcterms", "pos_3", -0.5)
    d1 = get_float(blocks, "rad_srcterms", "dir_1", 1.0)
    d2 = get_float(blocks, "rad_srcterms", "dir_2", 0.08)
    d3 = get_float(blocks, "rad_srcterms", "dir_3", 0.02)
    dnorm = float(np.sqrt(d1 * d1 + d2 * d2 + d3 * d3))
    d1, d3 = d1 / dnorm, d3 / dnorm
    svals = np.linspace(0.0, 14.0, 100)
    ax.plot(p1 + svals * d1, p3 + svals * d3, color="#8bc34a", lw=1.1, ls="--")
    ax.scatter([p1], [p3], s=34, facecolor="none", edgecolor="white", linewidths=1.1)


def plot_rad_slices(rad_xy, rad_xz, blocks, fig_dir: Path) -> None:
    x1f, x2f, rxy, _, _ = squeeze_slice(rad_xy, "r00")
    x1fz, x3f, rxz, _, _ = squeeze_slice(rad_xz, "r00")
    vmax = max(float(np.nanmax(rxy)), float(np.nanmax(rxz)), 1.0e-30)
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    im = axes[0].pcolormesh(x1f, x2f, rxy, shading="auto", cmap="magma", norm=norm)
    draw_xy_guides(axes[0], blocks, rad_xy.time)
    axes[0].set_xlim(x1f[0], x1f[-1])
    axes[0].set_ylim(x2f[0], x2f[-1])
    axes[0].set_title(r"$R^{tt}$, $x-y$ slice")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_aspect("equal")

    axes[1].pcolormesh(x1fz, x3f, rxz, shading="auto", cmap="magma", norm=norm)
    draw_xz_guides(axes[1], blocks, rad_xz.time)
    axes[1].set_xlim(x1fz[0], x1fz[-1])
    axes[1].set_ylim(x3f[0], x3f[-1])
    axes[1].set_title(r"$R^{tt}$, $x-z$ slice")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$z$")
    axes[1].set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes, shrink=0.88)
    cbar.set_label(r"$R^{tt}$")
    fig.savefig(fig_dir / "dynbbh_beam_rtt_slices.png", dpi=190)
    plt.close(fig)


def plot_adm_background(adm_xy, blocks, fig_dir: Path) -> None:
    x1f, x2f, alpha, _, _ = squeeze_slice(adm_xy, "adm_alpha")
    _, _, psi4, _, _ = squeeze_slice(adm_xy, "adm_psi4")
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    im = ax.pcolormesh(x1f, x2f, alpha, shading="auto", cmap="viridis")
    ax.set_xlim(x1f[0], x1f[-1])
    ax.set_ylim(x2f[0], x2f[-1])
    finite = psi4[np.isfinite(psi4)]
    if finite.size and float(np.nanmax(finite)) > float(np.nanmin(finite)):
        levels = np.linspace(float(np.nanpercentile(finite, 15)),
                             float(np.nanpercentile(finite, 95)), 6)
        xc = 0.5 * (x1f[:-1] + x1f[1:])
        yc = 0.5 * (x2f[:-1] + x2f[1:])
        ax.contour(xc, yc, psi4, levels=levels, colors="white", linewidths=0.65, alpha=0.65)
    draw_xy_guides(ax, blocks, adm_xy.time)
    ax.set_title("Superposed orbiting KS ADM background")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\alpha$")
    fig.savefig(fig_dir / "dynbbh_beam_adm_background.png", dpi=190)
    plt.close(fig)


def plot_particles_and_lineout(tab_path: Path, trk_path: Path, blocks, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    names, data = read_tab(tab_path)
    x = column(names, data, "x1v", 2)
    rtt = column(names, data, "r00", -10)
    order = np.argsort(x)
    axes[0].plot(x[order], rtt[order], color="tab:blue", lw=1.5)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$R^{tt}$")
    axes[0].set_title("Lineout through beam core")

    times, frames = read_trk(trk_path)
    if frames.size:
        for tag in range(frames.shape[1]):
            axes[1].plot(frames[:, tag, 0], frames[:, tag, 1], lw=1.0, alpha=0.8)
        axes[1].scatter(frames[0, :, 0], frames[0, :, 1], s=18, color="black",
                        label="start")
        axes[1].scatter(frames[-1, :, 0], frames[-1, :, 1], s=18, color="#ffcc33",
                        label="end")
        draw_xy_guides(axes[1], blocks, float(times[-1]))
        axes[1].legend(frameon=False, loc="best")
    axes[1].set_title("Null beam-edge tracers")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_xlim(-6.5, 2.5)
    axes[1].set_ylim(-2.2, 2.2)
    axes[1].set_aspect("equal")
    fig.savefig(fig_dir / "dynbbh_beam_particles_lineout.png", dpi=190)
    fig.savefig(fig_dir / "dynbbh_beam_particles.png", dpi=190)
    plt.close(fig)


def plot_summary(rad_xy, rad_xz, adm_xy, tab_path: Path, trk_path: Path,
                 blocks, fig_dir: Path) -> None:
    x1f, x2f, rxy, _, _ = squeeze_slice(rad_xy, "r00")
    x1fz, x3f, rxz, _, _ = squeeze_slice(rad_xz, "r00")
    _, _, alpha, _, _ = squeeze_slice(adm_xy, "adm_alpha")
    vmax = max(float(np.nanmax(rxy)), float(np.nanmax(rxz)), 1.0e-30)
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), constrained_layout=True)
    im = axes[0, 0].pcolormesh(x1f, x2f, rxy, shading="auto", cmap="magma", norm=norm)
    draw_xy_guides(axes[0, 0], blocks, rad_xy.time)
    axes[0, 0].set_xlim(x1f[0], x1f[-1])
    axes[0, 0].set_ylim(x2f[0], x2f[-1])
    axes[0, 0].set_title(r"$R^{tt}$, $x-y$")
    axes[0, 0].set_aspect("equal")

    axes[0, 1].pcolormesh(x1fz, x3f, rxz, shading="auto", cmap="magma", norm=norm)
    draw_xz_guides(axes[0, 1], blocks, rad_xz.time)
    axes[0, 1].set_xlim(x1fz[0], x1fz[-1])
    axes[0, 1].set_ylim(x3f[0], x3f[-1])
    axes[0, 1].set_title(r"$R^{tt}$, $x-z$")
    axes[0, 1].set_aspect("equal")

    im_alpha = axes[1, 0].pcolormesh(x1f, x2f, alpha, shading="auto", cmap="viridis")
    draw_xy_guides(axes[1, 0], blocks, adm_xy.time)
    axes[1, 0].set_xlim(x1f[0], x1f[-1])
    axes[1, 0].set_ylim(x2f[0], x2f[-1])
    axes[1, 0].set_title(r"ADM lapse $\alpha$")
    axes[1, 0].set_aspect("equal")

    times, frames = read_trk(trk_path)
    if frames.size:
        for tag in range(frames.shape[1]):
            axes[1, 1].plot(frames[:, tag, 0], frames[:, tag, 1], lw=1.0, alpha=0.8)
        axes[1, 1].scatter(frames[0, :, 0], frames[0, :, 1], s=16, color="black")
        axes[1, 1].scatter(frames[-1, :, 0], frames[-1, :, 1], s=16, color="#ffcc33")
        draw_xy_guides(axes[1, 1], blocks, float(times[-1]))
    axes[1, 1].set_title("Null edge tracers")
    axes[1, 1].set_xlim(-6.5, 2.5)
    axes[1, 1].set_ylim(-2.2, 2.2)
    axes[1, 1].set_aspect("equal")

    for ax in axes.flat:
        ax.set_xlabel(r"$x$")
    axes[0, 0].set_ylabel(r"$y$")
    axes[0, 1].set_ylabel(r"$z$")
    axes[1, 0].set_ylabel(r"$y$")
    axes[1, 1].set_ylabel(r"$y$")
    fig.colorbar(im, ax=axes[0, :], shrink=0.88, label=r"$R^{tt}$")
    fig.colorbar(im_alpha, ax=axes[1, 0], shrink=0.85, label=r"$\alpha$")
    fig.savefig(fig_dir / "dynbbh_beam_summary.png", dpi=190)
    plt.close(fig)

    names, data = read_tab(tab_path)
    x = column(names, data, "x1v", 2)
    rtt = column(names, data, "r00", -10)
    print(f"dynbbh beam lineout max Rtt={float(np.nanmax(rtt)):.6e}")
    print(f"dynbbh beam lineout integral={float(np.trapezoid(rtt[np.argsort(x)], np.sort(x))):.6e}")


def plot_time_series(rad_paths: list[Path], trk_path: Path, blocks, fig_dir: Path,
                     basename: str, max_frames: int) -> list[Path]:
    if not rad_paths:
        return []
    times, frames = read_trk(trk_path)
    out_dir = fig_dir / "dynbbh_beam_timeseries" / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    data_series = [read_binary_slice(path, ["r00"]) for path in rad_paths]
    data_series = [data for data in data_series if data.time > 0.0]
    if max_frames > 0 and len(data_series) > max_frames:
        keep = np.linspace(0, len(data_series) - 1, max_frames, dtype=int)
        data_series = [data_series[int(i)] for i in keep]
    vmax = max(float(np.nanmax(np.asarray(data.variables["r00"], dtype=float)))
               for data in data_series)
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=max(vmax, 1.0e-30))

    frame_paths: list[Path] = []
    for frame_id, data in enumerate(data_series):
        x1f, x2f, rxy, _, _ = squeeze_slice(data, "r00")
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        im = ax.pcolormesh(x1f, x2f, rxy, shading="auto", cmap="magma", norm=norm)
        draw_xy_guides(ax, blocks, data.time)
        if frames.size:
            nshow = int(np.searchsorted(times, data.time, side="right"))
            if nshow > 0:
                for tag in range(frames.shape[1]):
                    ax.plot(frames[:nshow, tag, 0], frames[:nshow, tag, 1],
                            color="white", lw=0.8, alpha=0.72)
                ax.scatter(frames[0, :, 0], frames[0, :, 1], s=10, color="black",
                           zorder=5)
                ax.scatter(frames[nshow - 1, :, 0], frames[nshow - 1, :, 1],
                           s=12, color="#ffcc33", zorder=5)
        ax.set_xlim(x1f[0], x1f[-1])
        ax.set_ylim(x2f[0], x2f[-1])
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(rf"dynbbh beam $R^{{tt}}$, $t={data.time:.2f}$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$R^{tt}$")
        out = out_dir / f"{basename}_snapshot_{frame_id:03d}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        frame_paths.append(out)

    if frame_paths:
        images = [plt.imread(path) for path in frame_paths]
        ncols = min(3, len(images))
        nrows = int(np.ceil(len(images) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                                 squeeze=False, constrained_layout=True)
        for ax in axes.flat:
            ax.axis("off")
        for ax, image, data in zip(axes.flat, images, data_series):
            ax.imshow(image)
            ax.set_title(rf"$t={data.time:.2f}$")
        montage = fig_dir / "dynbbh_beam_timeseries.png"
        fig.savefig(montage, dpi=160)
        plt.close(fig)
        frame_paths.append(montage)
    return frame_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/dynbbh_beam_figures"))
    parser.add_argument("--fig-dir", type=Path,
                        default=Path("dyngr_radiation_paper/figures"))
    parser.add_argument("--input", type=Path,
                        default=Path("dyngr_radiation_paper/inputs/dynbbh_beam_particles.athinput"))
    parser.add_argument("--basename", default="paper_dynbbh_beam")
    parser.add_argument("--athena", type=Path, default=None,
                        help="Athena executable compiled with -D PROBLEM=dynbbh")
    parser.add_argument("--tlim", type=float, default=6.0,
                        help="Final time for the figure run; overrides the smoke input.")
    parser.add_argument("--nlim", type=int, default=200,
                        help="Cycle cap for the figure run; overrides the smoke input.")
    parser.add_argument("--snapshot-dt", type=float, default=2.0,
                        help="Binary slice cadence for time-series snapshots.")
    parser.add_argument("--track-dt", type=float, default=0.25,
                        help="Tracked-particle output cadence for the figure run.")
    parser.add_argument("--nx1", type=int, default=None)
    parser.add_argument("--nx2", type=int, default=None)
    parser.add_argument("--nx3", type=int, default=None)
    parser.add_argument("--mb-nx1", type=int, default=None)
    parser.add_argument("--mb-nx2", type=int, default=None)
    parser.add_argument("--mb-nx3", type=int, default=None)
    parser.add_argument("--nlevel", type=int, default=None)
    parser.add_argument("--ppc", type=float, default=None)
    parser.add_argument("--ntrack", type=int, default=None)
    parser.add_argument("--beam-spread", type=float, default=None,
                        help="Override rad_srcterms/spread in degrees.")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--max-snapshot-frames", type=int, default=8)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    fig_dir = args.fig_dir if args.fig_dir.is_absolute() else root / args.fig_dir
    input_file = args.input if args.input.is_absolute() else root / args.input
    if args.athena is None:
        dynbbh_exe = root / "build_dynbbh_rad" / "src" / "athena"
        exe = dynbbh_exe if dynbbh_exe.exists() else root / "build" / "src" / "athena"
    else:
        exe = args.athena if args.athena.is_absolute() else root / args.athena
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    blocks = read_input(input_file)
    run_case(root, run_dir, exe, input_file, args.basename, args.tlim, args.nlim,
             args.snapshot_dt, args.track_dt, args.nx1, args.nx2, args.nx3,
             args.mb_nx1, args.mb_nx2, args.mb_nx3, args.nlevel, args.ppc,
             args.ntrack, args.beam_spread, args.mpi_ranks, args.skip_run)

    rad_xy = read_binary_slice(latest(run_dir, args.basename, ".rad_xy.*.bin"),
                               ["r00", "r01", "r02", "r03"])
    rad_xz = read_binary_slice(latest(run_dir, args.basename, ".rad_xz.*.bin"),
                               ["r00", "r01", "r02", "r03"])
    adm_xy = read_binary_slice(latest(run_dir, args.basename, ".adm_xy.*.bin"),
                               ["adm_alpha", "adm_psi4"])
    tab_path = latest(run_dir, args.basename, ".rad_coord.*.tab")
    trk_path = latest(run_dir, args.basename, ".trk")

    plot_rad_slices(rad_xy, rad_xz, blocks, fig_dir)
    plot_adm_background(adm_xy, blocks, fig_dir)
    plot_particles_and_lineout(tab_path, trk_path, blocks, fig_dir)
    plot_summary(rad_xy, rad_xz, adm_xy, tab_path, trk_path, blocks, fig_dir)
    snapshot_paths = plot_time_series(
        all_outputs(run_dir, args.basename, ".rad_xy.*.bin"), trk_path, blocks,
        fig_dir, args.basename, args.max_snapshot_frames)

    for name in (
        "dynbbh_beam_rtt_slices.png",
        "dynbbh_beam_adm_background.png",
        "dynbbh_beam_particles_lineout.png",
        "dynbbh_beam_summary.png",
        "dynbbh_beam_timeseries.png",
    ):
        print(f"wrote {fig_dir / name}")
    if snapshot_paths:
        print(f"wrote {len(snapshot_paths) - 1} snapshot frames under "
              f"{fig_dir / 'dynbbh_beam_timeseries' / args.basename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
