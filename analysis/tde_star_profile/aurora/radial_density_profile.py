#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def output_number(path: Path) -> str:
    parts = path.name.split(".")
    return parts[-2] if len(parts) >= 3 else path.stem


def find_files(run_dir: Path, output_id: str) -> List[Path]:
    search_dirs = [run_dir / "bin", run_dir]
    files: List[Path] = []
    for search_dir in search_dirs:
        files.extend(search_dir.glob(f"*.{output_id}.*.bin"))
    if not files:
        files.extend(run_dir.rglob(f"*.{output_id}.*.bin"))
    return sorted(set(files), key=lambda path: (output_number(path), str(path)))


def get_from_header(header: Iterable[str], blockname: str, keyname: str) -> str:
    blockname = blockname.strip()
    keyname = keyname.strip()
    if not blockname.startswith("<"):
        blockname = "<" + blockname
    if not blockname.endswith(">"):
        blockname += ">"
    block = "<none>"
    for line in header:
        if line.startswith("<"):
            block = line
            continue
        key, value = line.split("=")
        if block == blockname and key.strip() == keyname:
            return value
    raise KeyError(f"no parameter called {blockname}/{keyname}")


def read_binary_metadata(fp) -> Dict[str, object]:
    code_header = fp.readline().split()
    if len(code_header) < 1:
        raise TypeError("unknown file format")
    if code_header[0] != b"Athena":
        raise TypeError(f"bad file format {code_header[0]!r}")
    version = code_header[-1].split(b"=")[-1]
    if version != b"1.1":
        raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

    pheader_count = int(fp.readline().split(b"=")[-1])
    pheader = {}
    for _ in range(pheader_count - 1):
        key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
        pheader[key] = val

    nvars = int(fp.readline().split(b"=")[-1])
    var_names = [v.decode("utf-8") for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b"=")[-1])
    header = [
        line.decode("utf-8").split("#")[0].strip()
        for line in fp.read(header_size).split(b"\n")
    ]
    header = [line for line in header if line]

    locsizebytes = int(pheader["size of location"])
    varsizebytes = int(pheader["size of variable"])
    if locsizebytes not in (4, 8):
        raise ValueError(f"unsupported location size {locsizebytes}")
    if varsizebytes not in (4, 8):
        raise ValueError(f"unsupported variable size {varsizebytes}")

    return {
        "time": float(pheader["time"]),
        "cycle": int(pheader["cycle"]),
        "loc_dtype": np.float64 if locsizebytes == 8 else np.float32,
        "var_dtype": np.float64 if varsizebytes == 8 else np.float32,
        "locsizebytes": locsizebytes,
        "varsizebytes": varsizebytes,
        "nvars": nvars,
        "var_names": var_names,
        "header": header,
        "nghost": int(get_from_header(header, "<mesh>", "nghost")),
    }


def accumulate_block(
    density: np.ndarray,
    geometry: np.ndarray,
    center: Tuple[float, float, float],
    edges: np.ndarray,
    sums: Dict[str, np.ndarray],
) -> None:
    nx3, nx2, nx1 = density.shape
    x1min, x1max, x2min, x2max, x3min, x3max = [float(v) for v in geometry]
    dx1 = (x1max - x1min) / nx1
    dx2 = (x2max - x2min) / nx2
    dx3 = (x3max - x3min) / nx3
    cell_volume = dx1 * dx2 * dx3

    x = x1min + (np.arange(nx1, dtype=np.float64) + 0.5) * dx1 - center[0]
    y = x2min + (np.arange(nx2, dtype=np.float64) + 0.5) * dx2 - center[1]
    z = x3min + (np.arange(nx3, dtype=np.float64) + 0.5) * dx3 - center[2]
    dr = edges[1] - edges[0]
    nbins = len(edges) - 1

    x2 = x[None, :] ** 2
    y2 = y[:, None] ** 2
    for kk, zc in enumerate(z):
        radius = np.sqrt(x2 + y2 + zc * zc)
        idx = np.floor((radius - edges[0]) / dr).astype(np.int64)
        valid = (idx >= 0) & (idx < nbins) & np.isfinite(density[kk])
        if not np.any(valid):
            continue
        shell = idx[valid]
        rho = density[kk][valid].astype(np.float64, copy=False)
        weight = np.full(rho.shape, cell_volume, dtype=np.float64)
        np.add.at(sums["volume"], shell, weight)
        np.add.at(sums["rho_volume"], shell, rho * weight)
        np.add.at(sums["rho2_volume"], shell, rho * rho * weight)
        np.add.at(sums["cell_count"], shell, 1)
        np.maximum.at(sums["rho_max"], shell, rho)
        np.minimum.at(sums["rho_min"], shell, rho)


def profile_one(path: Path, center: Tuple[float, float, float], edges: np.ndarray) -> Dict[str, object]:
    filesize = path.stat().st_size
    sums = {
        "volume": np.zeros(len(edges) - 1, dtype=np.float64),
        "rho_volume": np.zeros(len(edges) - 1, dtype=np.float64),
        "rho2_volume": np.zeros(len(edges) - 1, dtype=np.float64),
        "cell_count": np.zeros(len(edges) - 1, dtype=np.int64),
        "rho_max": np.full(len(edges) - 1, -np.inf, dtype=np.float64),
        "rho_min": np.full(len(edges) - 1, np.inf, dtype=np.float64),
    }
    global_rho_max = -np.inf
    global_rho_max_radius = np.nan
    n_mbs = 0
    nonfinite_density_cells = 0

    with path.open("rb") as fp:
        meta = read_binary_metadata(fp)
        if "dens" not in meta["var_names"]:
            raise KeyError(f"{path} has no dens variable; variables={meta['var_names']}")
        dens_index = list(meta["var_names"]).index("dens")
        var_dtype = meta["var_dtype"]
        loc_dtype = meta["loc_dtype"]
        varsizebytes = int(meta["varsizebytes"])
        nvars = int(meta["nvars"])
        nghost = int(meta["nghost"])

        while fp.tell() < filesize:
            raw_index = fp.read(24)
            if not raw_index:
                break
            if len(raw_index) != 24:
                raise EOFError(f"truncated meshblock index in {path}")
            mb_index = np.frombuffer(raw_index, dtype=np.int32).astype(np.int64) - nghost
            nx1_out = int((mb_index[1] - mb_index[0]) + 1)
            nx2_out = int((mb_index[3] - mb_index[2]) + 1)
            nx3_out = int((mb_index[5] - mb_index[4]) + 1)
            fp.read(16)
            geometry = np.frombuffer(fp.read(6 * int(meta["locsizebytes"])), dtype=loc_dtype)
            cells = nx1_out * nx2_out * nx3_out

            fp.seek(dens_index * cells * varsizebytes, 1)
            density = np.fromfile(fp, dtype=var_dtype, count=cells)
            fp.seek((nvars - dens_index - 1) * cells * varsizebytes, 1)
            density = density.reshape(nx3_out, nx2_out, nx1_out)
            accumulate_block(density, geometry, center, edges, sums)

            finite_density = np.isfinite(density)
            nonfinite_density_cells += int(density.size - np.count_nonzero(finite_density))
            if not np.any(finite_density):
                n_mbs += 1
                continue
            density_for_max = np.where(finite_density, density, -np.inf)
            local_max_index = int(np.argmax(density_for_max))
            local_max = float(density_for_max.reshape(-1)[local_max_index])
            if local_max > global_rho_max:
                kk, jj, ii = np.unravel_index(local_max_index, density.shape)
                x1min, x1max, x2min, x2max, x3min, x3max = [float(v) for v in geometry]
                dx1 = (x1max - x1min) / nx1_out
                dx2 = (x2max - x2min) / nx2_out
                dx3 = (x3max - x3min) / nx3_out
                x = x1min + (ii + 0.5) * dx1 - center[0]
                y = x2min + (jj + 0.5) * dx2 - center[1]
                z = x3min + (kk + 0.5) * dx3 - center[2]
                global_rho_max = local_max
                global_rho_max_radius = float(np.sqrt(x*x + y*y + z*z))
            n_mbs += 1

    volume = sums["volume"]
    rho_mean = np.divide(
        sums["rho_volume"], volume,
        out=np.full_like(volume, np.nan),
        where=volume > 0.0,
    )
    rho_rms = np.sqrt(np.divide(
        sums["rho2_volume"], volume,
        out=np.full_like(volume, np.nan),
        where=volume > 0.0,
    ))
    rho_min = np.where(np.isfinite(sums["rho_min"]), sums["rho_min"], np.nan)
    rho_max = np.where(np.isfinite(sums["rho_max"]), sums["rho_max"], np.nan)

    return {
        "path": str(path),
        "output_number": output_number(path),
        "time": float(meta["time"]),
        "cycle": int(meta["cycle"]),
        "n_mbs": n_mbs,
        "r_inner": edges[:-1].copy(),
        "r_outer": edges[1:].copy(),
        "r_center": 0.5 * (edges[:-1] + edges[1:]),
        "volume": volume,
        "cell_count": sums["cell_count"],
        "rho_mean": rho_mean,
        "rho_rms": rho_rms,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "rho_global_max": global_rho_max,
        "rho_global_max_radius": global_rho_max_radius,
        "mass_inside_rmax": float(np.sum(sums["rho_volume"])),
        "nonfinite_density_cells": nonfinite_density_cells,
    }


def write_profiles_csv(path: Path, case: str, profiles: List[Dict[str, object]]) -> None:
    columns = [
        "case", "output_number", "time", "cycle", "r_inner", "r_center", "r_outer",
        "rho_mean", "rho_rms", "rho_min", "rho_max", "volume", "cell_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for profile in profiles:
            for idx, radius in enumerate(profile["r_center"]):
                writer.writerow({
                    "case": case,
                    "output_number": profile["output_number"],
                    "time": profile["time"],
                    "cycle": profile["cycle"],
                    "r_inner": profile["r_inner"][idx],
                    "r_center": radius,
                    "r_outer": profile["r_outer"][idx],
                    "rho_mean": profile["rho_mean"][idx],
                    "rho_rms": profile["rho_rms"][idx],
                    "rho_min": profile["rho_min"][idx],
                    "rho_max": profile["rho_max"][idx],
                    "volume": profile["volume"][idx],
                    "cell_count": profile["cell_count"][idx],
                })


def write_summary_csv(path: Path, case: str, profiles: List[Dict[str, object]]) -> None:
    columns = [
        "case", "output_number", "time", "cycle", "n_mbs", "rho_global_max",
        "rho_global_max_radius", "mass_inside_rmax", "nonfinite_density_cells",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for profile in profiles:
            writer.writerow({column: profile[column] if column != "case" else case for column in columns})


def plot_overlay(path: Path, profiles: List[Dict[str, object]], rho_floor: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    denom = max(len(profiles) - 1, 1)
    for idx, profile in enumerate(profiles):
        color = cmap(idx / denom)
        rho = np.clip(profile["rho_mean"], rho_floor, None)
        ax.plot(profile["r_center"], rho, color=color, lw=1.6, label=f"t={profile['time']:.3g}")
    ax.set_yscale("log")
    ax.set_xlabel("r from star center")
    ax.set_ylabel("shell-averaged density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(path: Path, profiles: List[Dict[str, object]]) -> None:
    times = np.asarray([profile["time"] for profile in profiles], dtype=np.float64)
    rho_max = np.asarray([profile["rho_global_max"] for profile in profiles], dtype=np.float64)
    mass = np.asarray([profile["mass_inside_rmax"] for profile in profiles], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(times, rho_max, marker="o")
    axes[0].set_ylabel("max density")
    axes[0].set_yscale("log")
    axes[1].plot(times, mass, marker="o")
    axes[1].set_ylabel("mass inside r_max")
    axes[1].set_xlabel("time")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute spherical shell density profiles from AthenaK mhd_w_bcc binary outputs."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post"))
    parser.add_argument("--id", default="mhd_w_bcc", help="AthenaK binary output id.")
    parser.add_argument("--center", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--r-max", type=float, default=2.0)
    parser.add_argument("--dr", type=float, default=0.0125)
    parser.add_argument("--rho-floor", type=float, default=1.0e-16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = find_files(args.run_dir, args.id)
    if not files:
        raise FileNotFoundError(f"No *.{args.id}.*.bin files found under {args.run_dir}")

    edges = np.arange(0.0, args.r_max + 0.5 * args.dr, args.dr, dtype=np.float64)
    if len(edges) < 2:
        raise ValueError("r_max/dr must define at least one radial bin")

    profiles = [profile_one(path, tuple(args.center), edges) for path in files]
    profiles.sort(key=lambda item: (float(item["time"]), str(item["output_number"])))

    out_dir = args.output_root / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    write_profiles_csv(out_dir / "radial_density_profiles.csv", args.case, profiles)
    write_summary_csv(out_dir / "radial_density_summary.csv", args.case, profiles)
    plot_overlay(out_dir / "radial_density_profile_overlay.png", profiles, args.rho_floor)
    plot_summary(out_dir / "radial_density_summary.png", profiles)
    metadata = {
        "case": args.case,
        "run_dir": str(args.run_dir),
        "output_id": args.id,
        "center": list(args.center),
        "r_max": args.r_max,
        "dr": args.dr,
        "n_profiles": len(profiles),
        "files": [profile["path"] for profile in profiles],
    }
    (out_dir / "radial_density_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(out_dir / "radial_density_profiles.csv")
    print(out_dir / "radial_density_profile_overlay.png")


if __name__ == "__main__":
    main()
