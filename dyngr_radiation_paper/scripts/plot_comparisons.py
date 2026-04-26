#!/usr/bin/env python3
"""Create comparison figures for dyngr_radiation_method.tex."""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    parser.add_argument("--trk-dir", type=Path, default=Path("trk"))
    parser.add_argument("--lwave-legacy", type=Path, default=Path("/tmp/paper_lwave_rad-errs.dat"))
    parser.add_argument("--lwave-dyn", type=Path, default=Path("/tmp/paper_lwave_dyn-errs.dat"))
    args = parser.parse_args()

    fig_dir = args.paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_beam(args.tab_dir, fig_dir)
    plot_lwave([args.lwave_legacy, args.lwave_dyn], fig_dir)
    plot_dynbbh(args.tab_dir, args.trk_dir, fig_dir)


if __name__ == "__main__":
    main()
