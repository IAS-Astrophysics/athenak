#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib
matplotlib.use("Agg")
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


def uses_rank_dirs(path: Path) -> bool:
    return "rank_00000000" in path.parts


def find_files(input_dir: Path, output_id: str) -> List[Path]:
    files = sorted((input_dir / "bin").glob(f"*.{output_id}.*.bin"))
    if not files:
        files = sorted(input_dir.glob(f"*.{output_id}.*.bin"))
    if not files:
        files = sorted(input_dir.rglob(f"*.{output_id}.*.bin"))
    return files


def read_slice(path: Path, quantities: List[str]) -> Dict[str, np.ndarray]:
    bin_convert = load_bin_convert()
    reader = (
        bin_convert.read_all_ranks_binary_as_athdf
        if uses_rank_dirs(path)
        else bin_convert.read_binary_as_athdf
    )
    return reader(str(path), quantities=quantities)


def read_raw(path: Path) -> Dict[str, object]:
    bin_convert = load_bin_convert()
    reader = bin_convert.read_all_ranks_binary if uses_rank_dirs(path) else bin_convert.read_binary
    return reader(str(path))


def as_xy(data: Dict[str, np.ndarray], quantity: str) -> np.ndarray:
    arr = np.asarray(data[quantity])
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D XY data for {quantity}, got shape {arr.shape}")
    return arr


def mirror_metric(x: np.ndarray, y: np.ndarray, field: np.ndarray, parity: int,
                  x_center: Optional[float], x_half_width: Optional[float],
                  y_half_width: Optional[float]) -> Dict[str, object]:
    finite = np.isfinite(field)
    if x_center is None:
        dens_for_center = np.nan_to_num(np.abs(field), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        _, i_peak = np.unravel_index(np.argmax(dens_for_center), dens_for_center.shape)
        x_center = float(x[i_peak])
    x_mask = np.ones_like(x, dtype=bool)
    y_mask = np.ones_like(y, dtype=bool)
    if x_half_width is not None:
        x_mask &= np.abs(x - x_center) <= x_half_width
    if y_half_width is not None:
        y_mask &= np.abs(y) <= y_half_width

    pair_abs = []
    pair_rel = []
    l2_num = 0.0
    l2_den = 0.0
    pair_count = 0
    max_pair = None
    x_sub = x[x_mask]
    y_sub = y[y_mask]
    values = field[np.ix_(y_mask, x_mask)]
    finite_sub = finite[np.ix_(y_mask, x_mask)]
    for j, yv in enumerate(y_sub):
        jm = int(np.argmin(np.abs(y_sub + yv)))
        if j >= jm:
            continue
        lhs = values[j, :]
        rhs = parity * values[jm, :]
        valid = finite_sub[j, :] & finite_sub[jm, :]
        if not np.any(valid):
            continue
        diff = np.abs(lhs[valid] - rhs[valid])
        scale = np.maximum(np.maximum(np.abs(lhs[valid]), np.abs(rhs[valid])), 1.0e-300)
        rel = diff / scale
        pair_abs.append(float(np.max(diff)))
        pair_rel.append(float(np.max(rel)))
        l2_num += float(np.sum(diff**2))
        l2_den += float(np.sum(scale**2))
        pair_count += int(np.count_nonzero(valid))
        local = int(np.argmax(diff))
        valid_indices = np.flatnonzero(valid)
        ix = int(valid_indices[local])
        if max_pair is None or float(diff[local]) > max_pair["abs_diff"]:
            max_pair = {
                "abs_diff": float(diff[local]),
                "rel_diff": float(rel[local]),
                "x": float(x_sub[ix]),
                "y_a": float(y_sub[j]),
                "y_b": float(y_sub[jm]),
                "value_a": float(lhs[ix]),
                "parity_value_b": float(rhs[ix]),
            }
    return {
        "x_center": x_center,
        "l2_abs": float(np.sqrt(l2_num)) if pair_count else None,
        "l2_rel": float(np.sqrt(l2_num)/np.sqrt(l2_den)) if pair_count and l2_den > 0.0 else None,
        "linf_abs": max(pair_abs) if pair_abs else None,
        "linf_rel": max(pair_rel) if pair_rel else None,
        "pair_count": pair_count,
        "max_pair": max_pair,
    }


def mirror_difference(field: np.ndarray, parity: int) -> np.ndarray:
    return field - parity*np.flip(field, axis=0)


def output_number(path: Path) -> str:
    parts = path.name.split(".")
    return parts[-2] if len(parts) >= 3 else path.stem


def summarize_file(path: Path, output_id: str, quantities: List[str],
                   parity: Dict[str, int], x_half_width: Optional[float],
                   y_half_width: Optional[float],
                   x_center_override: Optional[float] = None) -> Dict[str, object]:
    data = read_slice(path, quantities)
    raw = read_raw(path)
    x = np.asarray(data["x1v"])
    y = np.asarray(data["x2v"])
    x_center = x_center_override
    if "dens" in data:
        dens = as_xy(data, "dens")
        dens_for_center = np.nan_to_num(dens, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        max_j, max_i = np.unravel_index(np.argmax(dens_for_center), dens_for_center.shape)
        x_center = float(x[max_i])
    metrics = {}
    for quantity in quantities:
        metrics[quantity] = mirror_metric(
            x, y, as_xy(data, quantity), parity.get(quantity, 1),
            x_center, x_half_width, y_half_width)
    return {
        "file": str(path),
        "output_id": output_id,
        "output_number": output_number(path),
        "time": float(raw.get("time", np.nan)),
        "cycle": int(raw.get("cycle", -1)),
        "quantities": metrics,
    }


def flatten_results(results: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for output in results["outputs"]:
        for quantity, metrics in output["quantities"].items():
            rows.append({
                "case": results["case"],
                "output_id": output["output_id"],
                "output_number": output["output_number"],
                "time": output["time"],
                "cycle": output["cycle"],
                "quantity": quantity,
                "l2_abs": metrics["l2_abs"],
                "l2_rel": metrics["l2_rel"],
                "linf_abs": metrics["linf_abs"],
                "linf_rel": metrics["linf_rel"],
                "pair_count": metrics["pair_count"],
                "x_center": metrics["x_center"],
            })
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    columns = [
        "case", "output_id", "output_number", "time", "cycle", "quantity",
        "l2_abs", "l2_rel", "linf_abs", "linf_rel", "pair_count", "x_center",
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(columns) + "\n")
        for row in rows:
            stream.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


def plot_norms(out_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({(str(r["output_id"]), str(r["quantity"])) for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for output_id, quantity in keys:
        group = [r for r in rows if r["output_id"] == output_id and r["quantity"] == quantity]
        group.sort(key=lambda r: (float(r["time"]) if np.isfinite(float(r["time"])) else float(r["output_number"])))
        x = np.asarray([
            float(r["time"]) if np.isfinite(float(r["time"])) else float(r["output_number"])
            for r in group
        ])
        l2 = np.asarray([float(r["l2_rel"]) if r["l2_rel"] is not None else np.nan for r in group])
        linf = np.asarray([float(r["linf_rel"]) if r["linf_rel"] is not None else np.nan for r in group])
        label = f"{output_id}:{quantity}"
        axes[0].plot(x, l2, marker="o", label=label)
        axes[1].plot(x, linf, marker="o", label=label)
    axes[0].set_ylabel("relative L2")
    axes[1].set_ylabel("relative Linf")
    axes[1].set_xlabel("time")
    for ax in axes:
        ydata = np.concatenate([line.get_ydata() for line in ax.lines]) if ax.lines else np.asarray([])
        if np.any(np.isfinite(ydata) & (ydata > 0.0)):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "symmetry_norms.png", dpi=160)
    plt.close(fig)


def plot_difference_slices(out_dir: Path, outputs: List[Dict[str, object]],
                           quantity_sets: Dict[str, Tuple[List[str], Dict[str, int]]]) -> None:
    latest_by_id: Dict[str, Dict[str, object]] = {}
    for output in outputs:
        prev = latest_by_id.get(output["output_id"])
        if prev is None or int(output["cycle"]) > int(prev["cycle"]):
            latest_by_id[output["output_id"]] = output

    for output_id, output in latest_by_id.items():
        quantities, parity = quantity_sets[output_id]
        path = Path(str(output["file"]))
        data = read_slice(path, quantities)
        x = np.asarray(data["x1v"])
        y = np.asarray(data["x2v"])
        for quantity in quantities:
            field = as_xy(data, quantity)
            diff = mirror_difference(field, parity.get(quantity, 1))
            vmax = np.nanmax(np.abs(diff))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(7, 4.5))
            im = ax.pcolormesh(x, y, diff, shading="auto", cmap="coolwarm",
                               vmin=-vmax, vmax=vmax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{output_id} {quantity} y-reflection diff, cycle {output['cycle']}")
            fig.colorbar(im, ax=ax, label=f"{quantity} - parity*mirror")
            fig.tight_layout()
            fig.savefig(out_dir / f"{output_id}_{quantity}_mirror_diff.png", dpi=160)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute y-reflection symmetry metrics.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post"))
    parser.add_argument("--x-half-width", type=float, default=2.0)
    parser.add_argument("--y-half-width", type=float, default=2.0)
    args = parser.parse_args()

    results = {"case": args.case, "run_dir": str(args.run_dir), "outputs": []}
    x_centers_by_output = {}
    mhd_files = find_files(args.run_dir, "xy_mhd")
    for path in mhd_files:
        summary = summarize_file(
            path, "xy_mhd", ["dens"], {"dens": 1}, args.x_half_width, args.y_half_width)
        x_centers_by_output[output_number(path)] = summary["quantities"]["dens"]["x_center"]
        results["outputs"].append(summary)

    z4c_quantities = [
        "z4c_chi", "z4c_Gamx", "z4c_Gamy", "z4c_Gamz", "z4c_alpha",
        "z4c_gxx", "z4c_gxy", "z4c_gyy", "z4c_Khat",
    ]
    z4c_parity = {
        "z4c_chi": 1,
        "z4c_Gamx": 1,
        "z4c_Gamy": -1,
        "z4c_Gamz": 1,
        "z4c_alpha": 1,
        "z4c_gxx": 1,
        "z4c_gxy": -1,
        "z4c_gyy": 1,
        "z4c_Khat": 1,
    }
    for path in find_files(args.run_dir, "xy_z4c"):
        results["outputs"].append(summarize_file(
            path, "xy_z4c", z4c_quantities, z4c_parity,
            args.x_half_width, args.y_half_width,
            x_center_override=x_centers_by_output.get(output_number(path))))

    out_dir = args.output_root / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "symmetry_metrics.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = flatten_results(results)
    write_csv(out_dir / "symmetry_metrics.csv", rows)
    plot_norms(out_dir, rows)
    plot_difference_slices(out_dir, results["outputs"], {
        "xy_mhd": (["dens"], {"dens": 1}),
        "xy_z4c": (z4c_quantities, z4c_parity),
    })
    print(out_path)


if __name__ == "__main__":
    main()
