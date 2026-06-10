#!/usr/bin/env python3
"""Plot AthenaK binary slice files centered on the binary."""

import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, "/home/hzhu/athenak/vis/python")
import bin_convert  # noqa: E402


def read_par_value(parfile, name, default):
    if not parfile.exists():
        return default
    prefix = name + "="
    for raw in parfile.read_text().splitlines():
        stripped = raw.split("#", 1)[0].replace(" ", "")
        if stripped.startswith(prefix):
            try:
                return float(stripped.split("=", 1)[1])
            except ValueError:
                return default
    return default


def bh_positions(time, sep, q):
    omega = sep ** -1.5
    phase = omega * time
    c = math.cos(phase)
    s = math.sin(phase)
    r1 = q / (1.0 + q) * sep
    r2 = 1.0 / (1.0 + q) * sep
    return (r1 * c, r1 * s), (-r2 * c, -r2 * s)


def slice_array(arr):
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D slice after squeezing, got shape {arr.shape}")
    return arr


def block_arrays(data, field):
    if field == "beta":
        out = []
        for p, bx, by, bz in zip(
            data["mb_data"]["press"],
            data["mb_data"]["bcc1"],
            data["mb_data"]["bcc2"],
            data["mb_data"]["bcc3"],
        ):
            b2 = slice_array(bx) ** 2 + slice_array(by) ** 2 + slice_array(bz) ** 2
            out.append(2.0 * slice_array(p) / np.maximum(b2, 1.0e-300))
        return out
    source = "dens" if field == "rho" else field
    return [slice_array(arr) for arr in data["mb_data"][source]]


def plane_info(slice_id):
    if slice_id == "slice_x3":
        return {
            "label": "xy",
            "xidx": (0, 1),
            "yidx": (2, 3),
            "xlabel": "x (M)",
            "ylabel": "y (M)",
            "out_suffix": "centered_xy",
            "mark_bh": True,
        }
    if slice_id == "slice_x2":
        return {
            "label": "xz",
            "xidx": (0, 1),
            "yidx": (4, 5),
            "xlabel": "x (M)",
            "ylabel": "z (M)",
            "out_suffix": "centered_xz",
            "mark_bh": False,
        }
    if slice_id == "slice_x1":
        return {
            "label": "yz",
            "xidx": (2, 3),
            "yidx": (4, 5),
            "xlabel": "y (M)",
            "ylabel": "z (M)",
            "out_suffix": "centered_yz",
            "mark_bh": False,
        }
    raise ValueError(f"unknown slice id {slice_id}")


def infer_slice_id(path):
    for slice_id in ("slice_x1", "slice_x2", "slice_x3"):
        if slice_id in path.name:
            return slice_id
    raise ValueError(f"could not infer slice id from {path}")


def geom_extent(geom, info):
    geom = list(map(float, geom))
    x0, x1 = geom[info["xidx"][0]], geom[info["xidx"][1]]
    y0, y1 = geom[info["yidx"][0]], geom[info["yidx"][1]]
    return x0, x1, y0, y1


def finite_range(arrays, geometries, extent, default, info):
    xmin, xmax, ymin, ymax = extent
    vals = []
    for geom, arr in zip(geometries, arrays):
        x0, x1, y0, y1 = geom_extent(geom, info)
        if x1 < xmin or x0 > xmax or y1 < ymin or y0 > ymax:
            continue
        arr = np.asarray(arr)
        ny, nx = arr.shape
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        i0 = max(0, int(np.floor((xmin - x0) / dx)))
        i1 = min(nx, int(np.ceil((xmax - x0) / dx)))
        j0 = max(0, int(np.floor((ymin - y0) / dy)))
        j1 = min(ny, int(np.ceil((ymax - y0) / dy)))
        if i0 >= i1 or j0 >= j1:
            continue
        finite = arr[j0:j1, i0:i1]
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return default
    allv = np.concatenate(vals)
    lo, hi = np.nanpercentile(allv, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
        return default
    return float(lo), float(hi)


def plot_one(args):
    (path_s, outdir_s, parfile_s, width, sep, q, mark_holes, circle_radius,
     auto_density, density_vmin, density_vmax) = args
    path = Path(path_s)
    outdir = Path(outdir_s)
    parfile = Path(parfile_s)
    data = bin_convert.read_binary(str(path))
    time = float(data["time"])
    cycle = int(data["cycle"])
    slice_id = infer_slice_id(path)
    info = plane_info(slice_id)
    pos1, pos2 = bh_positions(time, sep, q)

    fields = [
        ("rho", "Density", "viridis", (1.0e-5, 1.0)),
        ("temperature", "Temperature", "inferno", (1.0e-4, 1.0e-1)),
        ("beta", r"Plasma $\beta$", "coolwarm", (1.0e-2, 1.0e2)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    xmin, xmax = -0.5 * width, 0.5 * width
    ymin, ymax = -0.5 * width, 0.5 * width
    extent = (xmin, xmax, ymin, ymax)

    for ax, (field, title, cmap, default_range) in zip(axes, fields):
        arrays = block_arrays(data, field)
        if field == "rho" and density_vmin is not None and density_vmax is not None:
            vmin, vmax = density_vmin, density_vmax
        elif field == "rho" and auto_density:
            vmin, vmax = finite_range(arrays, data["mb_geometry"], extent, default_range, info)
        else:
            vmin, vmax = default_range
        image = None
        for geom, arr in zip(data["mb_geometry"], arrays):
            x0, x1, y0, y1 = geom_extent(geom, info)
            if x1 < xmin or x0 > xmax or y1 < ymin or y0 > ymax:
                continue
            masked = np.ma.masked_invalid(np.asarray(arr))
            image = ax.imshow(
                masked,
                origin="lower",
                extent=(x0, x1, y0, y1),
                interpolation="nearest",
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                aspect="equal",
            )
        if image is not None:
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        if circle_radius > 0.0 and info["mark_bh"]:
            for xh, yh in (pos1, pos2):
                ax.add_patch(plt.Circle((xh, yh), circle_radius, fill=False,
                                        edgecolor="red", linewidth=1.3))
        if mark_holes and info["mark_bh"]:
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], "wo", ms=4, mec="k", mew=0.7)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.set_xlabel(info["xlabel"])
        ax.grid(alpha=0.15)
    axes[0].set_ylabel(info["ylabel"])
    fig.suptitle(f"{path.stem}  {info['label']}  time={time:.1f}  cycle={cycle}")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{path.stem}.{info['out_suffix']}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--width", type=float, default=200.0)
    parser.add_argument("-n", "--nproc", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--mark-holes", action="store_true", help="draw BH centroid markers")
    parser.add_argument("--circle-radius", type=float, default=-1.0,
                        help="draw red empty circles of this radius; negative uses excise_1_rad")
    parser.add_argument("--auto-density", action="store_true",
                        help="autoscale density using positive finite values in the plotted viewport")
    parser.add_argument("--density-vmin", type=float, default=None,
                        help="fixed density colorbar lower bound")
    parser.add_argument("--density-vmax", type=float, default=None,
                        help="fixed density colorbar upper bound")
    parser.add_argument("--plane", choices=("x1", "x2", "x3", "all"), default="x3",
                        help="slice plane to plot: x1=yz, x2=xz, x3=xy")
    args = parser.parse_args()

    slice_ids = ["slice_x1", "slice_x2", "slice_x3"] if args.plane == "all" else [f"slice_{args.plane}"]
    files = []
    for slice_id in slice_ids:
        files.extend(sorted((args.run_dir / "bin").glob(f"torus.{slice_id}.*.bin"))[:: max(args.stride, 1)])
    if not files:
        raise SystemExit(f"No requested slice .bin files found in {args.run_dir / 'bin'}")
    parfile = args.run_dir / "parfile.par"
    sep = read_par_value(parfile, "sep", 25.0)
    q = read_par_value(parfile, "q", 1.0)
    circle_radius = args.circle_radius
    if circle_radius < 0.0:
        circle_radius = read_par_value(parfile, "sink_radius",
                        read_par_value(parfile, "excise_1_rad", 4.0))
    if (args.density_vmin is None) != (args.density_vmax is None):
        raise SystemExit("--density-vmin and --density-vmax must be provided together")
    if args.density_vmin is not None and (
            args.density_vmin <= 0.0 or args.density_vmax <= args.density_vmin):
        raise SystemExit("--density-vmin must be positive and less than --density-vmax")
    tasks = [
        (str(path), str(args.out_dir), str(parfile), args.width, sep, q,
         args.mark_holes, circle_radius, args.auto_density,
         args.density_vmin, args.density_vmax)
        for path in files
    ]
    if args.nproc > 1:
        with Pool(args.nproc) as pool:
            for out in pool.imap_unordered(plot_one, tasks):
                print(out)
    else:
        for task in tasks:
            print(plot_one(task))


if __name__ == "__main__":
    main()
