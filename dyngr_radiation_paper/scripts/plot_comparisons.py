#!/usr/bin/env python3
"""Create comparison figures for dyngr_radiation_method.tex."""

from __future__ import annotations

import argparse
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BinarySlice:
    path: Path
    time: float
    cycle: int
    variables: dict[str, np.ndarray]
    x1f: np.ndarray
    x2f: np.ndarray
    x3f: np.ndarray
    x1v: np.ndarray
    x2v: np.ndarray
    x3v: np.ndarray


def read_tab(path: Path) -> tuple[list[str], np.ndarray]:
    header = ""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# gid"):
                header = line[1:].strip()
                break
    if not header:
        raise RuntimeError(f"could not find column header in {path}")
    names = header.split()
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return names, data


def _header_value(header: list[str], block_name: str, key_name: str) -> str:
    block = ""
    target = f"<{block_name.strip('<>')}>"
    for line in header:
        if not line:
            continue
        if line.startswith("<"):
            block = line.strip()
            continue
        if block == target and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == key_name:
                return value.split("#", 1)[0].strip()
    raise KeyError(f"could not find {target}/{key_name}")


def read_binary_slice(path: Path, quantities: list[str]) -> BinarySlice:
    """Read selected variables from an AthenaK binary slice output."""
    path = Path(path)
    with path.open("rb") as fp:
        fp.seek(0, 2)
        file_size = fp.tell()
        fp.seek(0, 0)

        code_header = fp.readline().split()
        if not code_header or code_header[0] != b"Athena":
            raise TypeError(f"{path} is not an Athena binary dump")
        version = code_header[-1].split(b"=")[-1]
        if version != b"1.1":
            raise TypeError(f"{path} has unsupported binary version {version!r}")

        preheader_count = int(fp.readline().split(b"=")[-1])
        preheader = {}
        for _ in range(preheader_count - 1):
            key, value = [item.strip() for item in fp.readline().decode("utf-8").split("=")]
            preheader[key] = value

        time = float(preheader["time"])
        cycle = int(preheader["cycle"])
        location_size = int(preheader["size of location"])
        variable_size = int(preheader["size of variable"])
        loc_dtype = np.float64 if location_size == 8 else np.float32
        var_dtype = np.float64 if variable_size == 8 else np.float32

        nvars = int(fp.readline().split(b"=")[-1])
        var_names = [item.decode("utf-8") for item in fp.readline().split()[1:]]
        header_size = int(fp.readline().split(b"=")[-1])
        header = [
            line.decode("utf-8").split("#", 1)[0].strip()
            for line in fp.read(header_size).split(b"\n")
        ]
        header = [line for line in header if line]

        missing = sorted(set(quantities) - set(var_names))
        if missing:
            raise KeyError(f"{path} is missing variables {missing}; has {var_names}")
        quantity_indices = [var_names.index(name) for name in quantities]

        nghost = int(_header_value(header, "mesh", "nghost"))
        root_size = np.array(
            [
                int(_header_value(header, "mesh", "nx1")),
                int(_header_value(header, "mesh", "nx2")),
                int(_header_value(header, "mesh", "nx3")),
            ],
            dtype=np.int64,
        )
        xmins = np.array(
            [
                float(_header_value(header, "mesh", "x1min")),
                float(_header_value(header, "mesh", "x2min")),
                float(_header_value(header, "mesh", "x3min")),
            ]
        )
        xmaxs = np.array(
            [
                float(_header_value(header, "mesh", "x1max")),
                float(_header_value(header, "mesh", "x2max")),
                float(_header_value(header, "mesh", "x3max")),
            ]
        )

        mb_logical: list[np.ndarray] = []
        mb_data: dict[str, list[np.ndarray]] = {name: [] for name in quantities}
        block_size = None

        while fp.tell() < file_size:
            mb_index = np.frombuffer(fp.read(6 * 4), dtype=np.int32).astype(np.int64) - nghost
            nx_out = np.array(
                [
                    mb_index[1] - mb_index[0] + 1,
                    mb_index[3] - mb_index[2] + 1,
                    mb_index[5] - mb_index[4] + 1,
                ],
                dtype=np.int64,
            )
            if block_size is None:
                block_size = nx_out

            logical = np.frombuffer(fp.read(4 * 4), dtype=np.int32).astype(np.int64)
            fp.read(6 * location_size)
            data = np.fromfile(fp, dtype=var_dtype, count=int(nx_out.prod()) * nvars)
            data = data.reshape(nvars, nx_out[2], nx_out[1], nx_out[0])

            mb_logical.append(logical)
            for name, idx in zip(quantities, quantity_indices):
                mb_data[name].append(data[idx])

    if block_size is None:
        raise ValueError(f"{path} has no meshblock data")

    mb_logical_arr = np.asarray(mb_logical, dtype=np.int64)
    levels = mb_logical_arr[:, 3]
    locations = mb_logical_arr[:, :3]
    level = int(levels.max())

    nx_vals: list[int] = []
    for dim in range(3):
        if block_size[dim] == 1 and root_size[dim] > 1:
            other_locations = [
                (lev, loc[(dim + 1) % 3], loc[(dim + 2) % 3])
                for lev, loc in zip(levels, locations)
            ]
            nx_vals.append(1 if len(set(other_locations)) == len(other_locations) else 0)
        elif block_size[dim] == 1:
            nx_vals.append(1)
        else:
            nx_vals.append(int(root_size[dim] * 2**level))
    if any(n == 0 for n in nx_vals):
        raise NotImplementedError("non-slice reduction binary outputs are not supported")
    nx1, nx2, nx3 = nx_vals

    out = {name: np.zeros((nx3, nx2, nx1), dtype=np.float64) for name in quantities}
    for block_num, block_level in enumerate(levels):
        scale = int(2 ** (level - block_level))
        location = locations[block_num]
        starts = np.array(
            [
                location[0] * block_size[0] * scale if nx1 > 1 else 0,
                location[1] * block_size[1] * scale if nx2 > 1 else 0,
                location[2] * block_size[2] * scale if nx3 > 1 else 0,
            ],
            dtype=np.int64,
        )
        stops = starts + np.array(
            [
                block_size[0] * scale if nx1 > 1 else 1,
                block_size[1] * scale if nx2 > 1 else 1,
                block_size[2] * scale if nx3 > 1 else 1,
            ],
            dtype=np.int64,
        )
        for name in quantities:
            block = np.asarray(mb_data[name][block_num], dtype=np.float64)
            if scale > 1:
                block = np.repeat(np.repeat(np.repeat(block, scale, axis=2), scale, axis=1),
                                  scale, axis=0)
            out[name][starts[2]:stops[2], starts[1]:stops[1], starts[0]:stops[0]] = block

    xfs = [np.linspace(xmins[d], xmaxs[d], nx_vals[d] + 1) for d in range(3)]
    xvs = [0.5 * (xf[:-1] + xf[1:]) for xf in xfs]
    variables = {name: arr[0] if arr.shape[0] == 1 else arr for name, arr in out.items()}
    return BinarySlice(path=path, time=time, cycle=cycle, variables=variables,
                       x1f=xfs[0], x2f=xfs[1], x3f=xfs[2],
                       x1v=xvs[0], x2v=xvs[1], x3v=xvs[2])


def column(names: list[str], data: np.ndarray, preferred: str, fallback: int) -> np.ndarray:
    try:
        return data[:, names.index(preferred)]
    except ValueError:
        return data[:, fallback]


def read_errs(path: Path) -> dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if lines and lines[0].startswith("#") and len(lines) > 1:
        labels = lines[0][1:].split()
        values = [float(x) for x in lines[1].split()]
        out = dict(zip(labels, values))
        return {key: out[key] for key in labels if key.endswith("_L1") or key == "RMS-L1"}
    text = "\n".join(lines)
    numbers = [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", text)]
    keys = ["RMS-L1", "d_L1", "M1_L1", "E_L1"]
    return {key: numbers[i] for i, key in enumerate(keys[:len(numbers)])}


def read_trk(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    marker = b"# AthenaK tracked particle data at time="
    starts = [m.start() for m in re.finditer(re.escape(marker), raw)]
    times: list[float] = []
    frames: list[np.ndarray] = []
    for idx, start in enumerate(starts):
        line_end = raw.find(b"\n", start)
        if line_end < 0:
            continue
        line = raw[start:line_end].decode("ascii", errors="ignore")
        mt = re.search(r"time=\s*([-+0-9.eE]+)", line)
        mn = re.search(r"ntracked_prtcls=\s*(\d+)", line)
        if not (mt and mn):
            continue
        ntrack = int(mn.group(1))
        data_start = line_end + 1
        if raw[data_start:data_start + 2] == b" \n":
            data_start += 2
        data_end = starts[idx + 1] - 1 if idx + 1 < len(starts) else len(raw)
        chunk = raw[data_start:data_end]
        need = 6 * ntrack * 4
        if len(chunk) < need:
            continue
        vals = struct.unpack("<" + "f" * (6 * ntrack), chunk[:need])
        times.append(float(mt.group(1)))
        frames.append(np.asarray(vals, dtype=float).reshape(ntrack, 6))
    if not frames:
        return np.empty(0), np.empty((0, 0, 6))
    return np.asarray(times), np.asarray(frames)


def plot_beam(tab_dir: Path, fig_dir: Path) -> None:
    cases = [
        ("legacy CKS", "paper_beam_rad_cks.rad_coord.00001.tab"),
        ("dyn CKS", "paper_beam_dyn_cks.rad_coord.00001.tab"),
        ("dyn ADM flat", "paper_beam_dyn_adm.rad_coord.00001.tab"),
    ]
    plt.figure(figsize=(7.2, 4.2))
    ref_x = ref_y = None
    for label, fname in cases:
        names, data = read_tab(tab_dir / fname)
        x = column(names, data, "x2v", 2)
        y = column(names, data, "r00", -10)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        plt.plot(x, y, marker="o", ms=3, lw=1.3, label=label)
        if ref_x is None:
            ref_x, ref_y = x, y
        else:
            common = min(len(ref_y), len(y))
            err = np.max(np.abs(y[:common] - ref_y[:common]))
            print(f"beam max |{label}-legacy| = {err:.6e}")
    plt.xlabel(r"$y$")
    plt.ylabel(r"$R^{00}$")
    plt.title("Beam test: legacy CKS vs dynamical radiation")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "beam_comparison.png", dpi=180)
    plt.close()


def load_beam_map(bin_dir: Path, fname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = read_binary_slice(bin_dir / fname, ["r00"])
    field = np.asarray(data.variables["r00"], dtype=float)
    if field.ndim != 2:
        raise ValueError(f"{fname} is not a 2D slice")
    return data.x1f, data.x2f, field


def plot_beam_2d(bin_dir: Path, fig_dir: Path) -> None:
    cases = [
        ("legacy CKS", "paper_beam_rad_cks.rad_xy.00001.bin"),
        ("dyn CKS", "paper_beam_dyn_cks.rad_xy.00001.bin"),
        ("dyn ADM flat", "paper_beam_dyn_adm.rad_xy.00001.bin"),
    ]
    maps = []
    for label, fname in cases:
        x1f, x2f, field = load_beam_map(bin_dir, fname)
        maps.append((label, x1f, x2f, field))

    vmax = max(float(np.nanmax(field)) for _, _, _, field in maps)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), sharex=True, sharey=True)
    for ax, (label, x1f, x2f, field) in zip(axes, maps):
        im = ax.pcolormesh(x1f, x2f, field, shading="auto", vmin=0.0, vmax=vmax,
                           cmap="magma")
        ax.set_title(label)
        ax.set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.18, top=0.76, wspace=0.12)
    cax = fig.add_axes([0.90, 0.20, 0.018, 0.56])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$R^{00}$")
    fig.suptitle(r"Flat-space beam radiation field", y=0.95)
    fig.savefig(fig_dir / "beam_2d_fields.png", dpi=180)
    plt.close(fig)

    ref = maps[0][3]
    residuals = [(maps[1][0], maps[1][1], maps[1][2], maps[1][3] - ref),
                 (maps[2][0], maps[2][1], maps[2][2], maps[2][3] - ref)]
    rmax = max(float(np.nanmax(np.abs(resid))) for _, _, _, resid in residuals)
    if rmax == 0.0:
        rmax = 1.0e-16
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharex=True, sharey=True)
    for ax, (label, x1f, x2f, resid) in zip(axes, residuals):
        im = ax.pcolormesh(x1f, x2f, resid, shading="auto", vmin=-rmax, vmax=rmax,
                           cmap="coolwarm")
        ax.set_title(f"{label}\nminus legacy CKS")
        ax.set_xlabel(r"$x$")
        print(f"beam 2D max |{label}-legacy| = {np.max(np.abs(resid)):.6e}")
    axes[0].set_ylabel(r"$y$")
    fig.subplots_adjust(left=0.09, right=0.84, bottom=0.18, top=0.74, wspace=0.12)
    cax = fig.add_axes([0.87, 0.20, 0.022, 0.54])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\Delta R^{00}$")
    fig.suptitle(r"Flat-space beam residuals", y=0.95)
    fig.savefig(fig_dir / "beam_2d_residuals.png", dpi=180)
    plt.close(fig)


def plot_lwave(err_files: list[Path], fig_dir: Path) -> None:
    labels = ["legacy CKS", "dyn CKS"]
    err_maps = [read_errs(path) for path in err_files]
    keys = sorted(set().union(*[set(e.keys()) for e in err_maps]))
    if not keys:
        keys = ["error"]
        err_maps = [{"error": 0.0}, {"error": 0.0}]
    x = np.arange(len(keys))
    width = 0.38
    plt.figure(figsize=(7.2, 4.2))
    for n, (label, emap) in enumerate(zip(labels, err_maps)):
        vals = [max(abs(emap.get(key, np.nan)), 1.0e-30) for key in keys]
        plt.bar(x + (n - 0.5) * width, vals, width, label=label)
    plt.yscale("log")
    plt.xticks(x, keys, rotation=35, ha="right")
    plt.ylabel("L1 error")
    plt.title("Radiation-hydro linear wave")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "linear_wave_errors.png", dpi=180)
    plt.close()


def plot_dynbbh(tab_dir: Path, trk_dir: Path, fig_dir: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    ax_main = plt.gca()
    tab_path = tab_dir / "paper_dynbbh_beam.rad_coord.00001.tab"
    if tab_path.exists():
        names, data = read_tab(tab_path)
        x = column(names, data, "x1v", 2)
        y = column(names, data, "r00", -10)
        order = np.argsort(x)
        ax_main.plot(x[order], y[order], color="tab:blue", lw=1.4,
                     label=r"beam $R^{00}$ slice")
        ax_main.set_xlabel(r"$x$")
        ax_main.set_ylabel(r"$R^{00}$")
        ax2 = ax_main.twinx()
    else:
        ax2 = ax_main
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")
    trk_path = trk_dir / "paper_dynbbh_beam.trk"
    if trk_path.exists():
        times, frames = read_trk(trk_path)
        if frames.size:
            ax2.set_ylabel(r"particle $y$")
            for tag in range(frames.shape[1]):
                ax2.plot(frames[:, tag, 0], frames[:, tag, 1], lw=1.0, alpha=0.75)
            ax2.scatter(frames[0, :, 0], frames[0, :, 1], s=12, color="black", label="edge particles")
    plt.title("Superposed-BBH ADM beam and null edge tracers")
    handles, labels = ax_main.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        plt.legend(handles + handles2, labels + labels2, frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir / "dynbbh_beam_particles.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--paper-dir", type=Path, default=Path("dyngr_radiation_paper"))
    parser.add_argument("--tab-dir", type=Path, default=Path("tab"))
    parser.add_argument("--bin-dir", type=Path, default=Path("bin"))
    parser.add_argument("--trk-dir", type=Path, default=Path("trk"))
    parser.add_argument("--lwave-legacy", type=Path, default=Path("/tmp/paper_lwave_rad-errs.dat"))
    parser.add_argument("--lwave-dyn", type=Path, default=Path("/tmp/paper_lwave_dyn-errs.dat"))
    args = parser.parse_args()

    fig_dir = args.paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_beam(args.tab_dir, fig_dir)
    plot_beam_2d(args.bin_dir, fig_dir)
    plot_lwave([args.lwave_legacy, args.lwave_dyn], fig_dir)
    plot_dynbbh(args.tab_dir, args.trk_dir, fig_dir)


if __name__ == "__main__":
    main()
