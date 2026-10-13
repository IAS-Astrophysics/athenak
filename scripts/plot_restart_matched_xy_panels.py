#!/usr/bin/env python3
"""Plot XY ATHDF slices closest in time to restart files."""

import argparse
import csv
import re
import struct
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, "/home/hzhu/athenak/vis/python")
import bin_convert  # noqa: E402


RST_TIME_OFFSET = 256


def restart_time(path):
    with path.open("rb") as fp:
        data = fp.read(30000)
    marker = b"<par_end>\n"
    marker_at = data.find(marker)
    if marker_at < 0:
        raise ValueError(f"{path} does not contain <par_end>")
    offset = marker_at + len(marker) + RST_TIME_OFFSET
    time = struct.unpack_from("<d", data, offset)[0]
    if not np.isfinite(time) or time < 0.0:
        raise ValueError(f"could not read a valid restart time from {path}")
    return float(time)


def athdf_time(path):
    with h5py.File(path, "r") as h5f:
        return float(h5f.attrs["Time"])


def file_index(path):
    match = re.search(r"\.(\d+)\.[^.]+$", path.name)
    return match.group(1) if match else path.stem


def load_xy_fields(path):
    if path.suffix == ".bin":
        return load_xy_fields_bin(path)
    if path.suffix != ".athdf":
        raise ValueError(f"unsupported slice format: {path}")
    return load_xy_fields_athdf(path)


def load_xy_fields_bin(path):
    filedata = bin_convert.read_binary(str(path))
    dens = np.asarray(filedata["mb_data"]["dens"])[:, 0, :, :]
    temp = np.asarray(filedata["mb_data"]["temperature"])[:, 0, :, :]
    press = np.asarray(filedata["mb_data"]["press"])[:, 0, :, :]
    bx = np.asarray(filedata["mb_data"]["bcc1"])[:, 0, :, :]
    by = np.asarray(filedata["mb_data"]["bcc2"])[:, 0, :, :]
    bz = np.asarray(filedata["mb_data"]["bcc3"])[:, 0, :, :]
    beta = 2.0 * press / np.maximum(bx * bx + by * by + bz * bz, 1.0e-300)

    geometry = np.asarray(filedata["mb_geometry"], dtype=float)
    levels = np.asarray(filedata["mb_logical"])[:, 3]

    return {
        "time": float(filedata["time"]),
        "cycle": int(filedata["cycle"]),
        "levels": levels,
        "x1f": geometry[:, 0:2],
        "x2f": geometry[:, 2:4],
        "root_x1": (float(filedata["x1min"]), float(filedata["x1max"])),
        "root_x2": (float(filedata["x2min"]), float(filedata["x2max"])),
        "fields": {
            "density": dens,
            "temperature": temp,
            "beta": beta,
        },
    }


def load_xy_fields_athdf(path):
    with h5py.File(path, "r") as h5f:
        names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5f.attrs["VariableNames"]
        ]
        uov_names = names[: h5f["uov"].shape[0]]
        b_names = names[h5f["uov"].shape[0] :]
        uov = h5f["uov"]
        bfield = h5f["B"]

        dens = uov[uov_names.index("dens"), :, 0, :, :]
        temp = uov[uov_names.index("temperature"), :, 0, :, :]
        press = uov[uov_names.index("press"), :, 0, :, :]
        bx = bfield[b_names.index("bcc1"), :, 0, :, :]
        by = bfield[b_names.index("bcc2"), :, 0, :, :]
        bz = bfield[b_names.index("bcc3"), :, 0, :, :]
        beta = 2.0 * press / np.maximum(bx * bx + by * by + bz * bz, 1.0e-300)

        return {
            "time": float(h5f.attrs["Time"]),
            "cycle": int(h5f.attrs["NumCycles"]),
            "levels": np.asarray(h5f["Levels"][:]),
            "x1f": np.asarray(h5f["x1f"][:]),
            "x2f": np.asarray(h5f["x2f"][:]),
            "root_x1": tuple(float(x) for x in h5f.attrs["RootGridX1"][:2]),
            "root_x2": tuple(float(x) for x in h5f.attrs["RootGridX2"][:2]),
            "fields": {
                "density": np.asarray(dens),
                "temperature": np.asarray(temp),
                "beta": np.asarray(beta),
            },
        }


def plot_blocks(ax, data, field, extent, cmap, limits):
    xmin, xmax, ymin, ymax = extent
    image = None
    order = np.argsort(data["levels"])
    for block in order:
        x0 = float(data["x1f"][block, 0])
        x1 = float(data["x1f"][block, -1])
        y0 = float(data["x2f"][block, 0])
        y1 = float(data["x2f"][block, -1])
        if x1 < xmin or x0 > xmax or y1 < ymin or y0 > ymax:
            continue
        arr = np.asarray(data["fields"][field][block])
        arr = np.ma.masked_invalid(np.ma.masked_less_equal(arr, 0.0))
        image = ax.imshow(
            arr,
            origin="lower",
            extent=(x0, x1, y0, y1),
            interpolation="nearest",
            cmap=cmap,
            norm=LogNorm(vmin=limits[0], vmax=limits[1]),
            aspect="equal",
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x (M)")
    ax.grid(alpha=0.12, linewidth=0.5)
    return image


def auto_limits(data, field, extent, fallback):
    xmin, xmax, ymin, ymax = extent
    chunks = []
    for block in range(data["fields"][field].shape[0]):
        x0 = float(data["x1f"][block, 0])
        x1 = float(data["x1f"][block, -1])
        y0 = float(data["x2f"][block, 0])
        y1 = float(data["x2f"][block, -1])
        if x1 < xmin or x0 > xmax or y1 < ymin or y0 > ymax:
            continue
        values = np.asarray(data["fields"][field][block]).ravel()
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            chunks.append(values)
    if not chunks:
        return fallback
    values = np.concatenate(chunks)
    lo, hi = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
        return fallback
    return float(lo), float(hi)


def make_panel(path, rst_name, rst_t, out_path, mode, width):
    data = load_xy_fields(path)
    if width is None:
        x0, x1 = data["root_x1"]
        y0, y1 = data["root_x2"]
        extent = (x0, x1, y0, y1)
        label = "full XY domain"
    else:
        half = 0.5 * width
        extent = (-half, half, -half, half)
        label = f"XY center +/- {half:g} M"

    fields = [
        ("density", "Density", "viridis", (1.0e-5, 1.0)),
        ("temperature", "Temperature", "inferno", (1.0e-4, 1.0e-1)),
        ("beta", r"Plasma $\beta$", "coolwarm", (1.0e-2, 1.0e2)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (field, title, cmap, limits) in zip(axes, fields):
        limits = auto_limits(data, field, extent, limits)
        image = plot_blocks(ax, data, field, extent, cmap, limits)
        ax.set_title(title)
        if image is not None:
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
    axes[0].set_ylabel("y (M)")
    if rst_t is None:
        title = f"{path.name}  t={data['time']:.6g}  cycle={data['cycle']}  {label}"
    else:
        title = (
            f"{rst_name}  restart t={rst_t:.6g}  "
            f"{path.name} t={data['time']:.6g}  {label}"
        )
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rst-dir", type=Path)
    parser.add_argument("--athdf-dir", type=Path)
    parser.add_argument("--input-dir", type=Path, help="plot all torus.slice_x3 .bin/.athdf files")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--zoom-width", type=float, default=100.0)
    args = parser.parse_args()

    if args.input_dir is not None:
        slice_paths = sorted(args.input_dir.glob("torus.slice_x3.*.athdf"))
        if not slice_paths:
            slice_paths = sorted(args.input_dir.glob("torus.slice_x3.*.bin"))
        if not slice_paths:
            raise SystemExit(f"no torus.slice_x3.*.bin/.athdf files found in {args.input_dir}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "slice_files.csv").open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["slice_file", "slice_time", "cycle"])
            for path in slice_paths:
                data = load_xy_fields(path)
                writer.writerow([path.name, f"{data['time']:.17g}", data["cycle"]])

        for path in slice_paths:
            idx = file_index(path)
            base = f"slice_x3_{idx}"
            make_panel(
                path,
                path.name,
                None,
                args.out_dir / f"{base}.full_xy.png",
                "full",
                None,
            )
            make_panel(
                path,
                path.name,
                None,
                args.out_dir / f"{base}.zoom_pm50_xy.png",
                "zoom",
                args.zoom_width,
            )
            print(f"wrote {base}")
        return

    if args.rst_dir is None or args.athdf_dir is None:
        raise SystemExit("--rst-dir and --athdf-dir are required unless --input-dir is used")

    rst_paths = sorted(args.rst_dir.glob("torus.*.rst"))
    athdf_paths = sorted(args.athdf_dir.glob("torus.slice_x3.*.athdf"))
    if not rst_paths:
        raise SystemExit(f"no restart files found in {args.rst_dir}")
    if not athdf_paths:
        raise SystemExit(f"no slice_x3 ATHDF files found in {args.athdf_dir}")

    athdf_times = [(athdf_time(path), path) for path in athdf_paths]
    rows = []
    for rst_path in rst_paths:
        rst_t = restart_time(rst_path)
        ath_t, ath_path = min(athdf_times, key=lambda item: abs(item[0] - rst_t))
        rows.append((rst_path, rst_t, ath_path, ath_t, abs(ath_t - rst_t)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "restart_to_slice_matches.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["restart_file", "restart_time", "slice_file", "slice_time", "abs_dt"]
        )
        for rst_path, rst_t, ath_path, ath_t, delta in rows:
            writer.writerow([rst_path.name, f"{rst_t:.17g}", ath_path.name, f"{ath_t:.17g}", f"{delta:.17g}"])

    for rst_path, rst_t, ath_path, _ath_t, _delta in rows:
        rst_idx = file_index(rst_path)
        ath_idx = file_index(ath_path)
        base = f"rst_{rst_idx}_slice_x3_{ath_idx}"
        make_panel(
            ath_path,
            rst_path.name,
            rst_t,
            args.out_dir / f"{base}.full_xy.png",
            "full",
            None,
        )
        make_panel(
            ath_path,
            rst_path.name,
            rst_t,
            args.out_dir / f"{base}.zoom_pm50_xy.png",
            "zoom",
            args.zoom_width,
        )
        print(f"wrote {base}")


if __name__ == "__main__":
    main()
