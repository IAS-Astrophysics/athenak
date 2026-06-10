#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/athenak_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MASK_THRESHOLDS = (1.0e-12, 1.0e-10, 1.0e-8)
TINY = 1.0e-100

MHD_PARITY_Y = {
    "dens": 1, "eint": 1, "press": 1, "temperature": 1,
    "velx": 1, "vely": -1, "velz": 1,
    "bcc1": 1, "bcc2": -1, "bcc3": 1,
}
MHD_PARITY_Z = {
    "dens": 1, "eint": 1, "press": 1, "temperature": 1,
    "velx": 1, "vely": 1, "velz": -1,
    "bcc1": 1, "bcc2": 1, "bcc3": -1,
}
Z4C_PARITY_Y = {
    "z4c_chi": 1,
    "z4c_gxx": 1, "z4c_gxy": -1, "z4c_gxz": 1,
    "z4c_gyy": 1, "z4c_gyz": -1, "z4c_gzz": 1,
    "z4c_Khat": 1,
    "z4c_Axx": 1, "z4c_Axy": -1, "z4c_Axz": 1,
    "z4c_Ayy": 1, "z4c_Ayz": -1, "z4c_Azz": 1,
    "z4c_Gamx": 1, "z4c_Gamy": -1, "z4c_Gamz": 1,
    "z4c_Theta": 1,
    "z4c_alpha": 1,
    "z4c_betax": 1, "z4c_betay": -1, "z4c_betaz": 1,
    "z4c_Bx": 1, "z4c_By": -1, "z4c_Bz": 1,
}
Z4C_PARITY_Z = {
    "z4c_chi": 1,
    "z4c_gxx": 1, "z4c_gxy": 1, "z4c_gxz": -1,
    "z4c_gyy": 1, "z4c_gyz": -1, "z4c_gzz": 1,
    "z4c_Khat": 1,
    "z4c_Axx": 1, "z4c_Axy": 1, "z4c_Axz": -1,
    "z4c_Ayy": 1, "z4c_Ayz": -1, "z4c_Azz": 1,
    "z4c_Gamx": 1, "z4c_Gamy": 1, "z4c_Gamz": -1,
    "z4c_Theta": 1,
    "z4c_alpha": 1,
    "z4c_betax": 1, "z4c_betay": 1, "z4c_betaz": -1,
    "z4c_Bx": 1, "z4c_By": 1, "z4c_Bz": -1,
}
ADM_PARITY_Y = {
    "adm_gxx": 1, "adm_gxy": -1, "adm_gxz": 1,
    "adm_gyy": 1, "adm_gyz": -1, "adm_gzz": 1,
    "adm_Kxx": 1, "adm_Kxy": -1, "adm_Kxz": 1,
    "adm_Kyy": 1, "adm_Kyz": -1, "adm_Kzz": 1,
    "adm_psi4": 1,
    "adm_alpha": 1, "adm_betax": 1, "adm_betay": -1, "adm_betaz": 1,
}
ADM_PARITY_Z = {
    "adm_gxx": 1, "adm_gxy": 1, "adm_gxz": -1,
    "adm_gyy": 1, "adm_gyz": -1, "adm_gzz": 1,
    "adm_Kxx": 1, "adm_Kxy": 1, "adm_Kxz": -1,
    "adm_Kyy": 1, "adm_Kyz": -1, "adm_Kzz": 1,
    "adm_psi4": 1,
    "adm_alpha": 1, "adm_betax": 1, "adm_betay": 1, "adm_betaz": -1,
}


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


def read_raw(path: Path) -> Dict[str, object]:
    bin_convert = load_bin_convert()
    reader = bin_convert.read_all_ranks_binary if uses_rank_dirs(path) else bin_convert.read_binary
    return reader(str(path))


def read_slice(path: Path, quantities: List[str]) -> Dict[str, np.ndarray]:
    bin_convert = load_bin_convert()
    reader = (
        bin_convert.read_all_ranks_binary_as_athdf
        if uses_rank_dirs(path)
        else bin_convert.read_binary_as_athdf
    )
    return reader(str(path), quantities=quantities)


def output_number(path: Path) -> str:
    parts = path.name.split(".")
    return parts[-2] if len(parts) >= 3 else path.stem


def output_id(path: Path) -> str:
    parts = path.name.split(".")
    return parts[-3] if len(parts) >= 4 else ""


def plane_for_output_id(out_id: str) -> Optional[str]:
    if out_id.startswith("xy_"):
        return "xy"
    if out_id.startswith("xz_"):
        return "xz"
    return None


def kind_for_output_id(out_id: str) -> str:
    if out_id.endswith("_mhd"):
        return "mhd"
    if out_id.endswith("_z4c"):
        return "z4c"
    if out_id.endswith("_adm"):
        return "adm"
    if out_id.endswith("_tmunu"):
        return "tmunu"
    return "unknown"


def find_files(input_dir: Path) -> List[Path]:
    candidates = sorted((input_dir / "bin").glob("*.bin"))
    if not candidates:
        candidates = sorted(input_dir.glob("*.bin"))
    if not candidates:
        candidates = sorted(input_dir.rglob("*.bin"))
    return [
        path for path in candidates
        if plane_for_output_id(output_id(path)) in ("xy", "xz")
        and kind_for_output_id(output_id(path)) in ("mhd", "z4c", "adm", "tmunu")
    ]


def available_quantities(path: Path, requested: Iterable[str]) -> List[str]:
    raw = read_raw(path)
    names = set(str(name) for name in raw.get("var_names", []))
    return [name for name in requested if name in names]


def requested_for(kind: str) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    if kind == "mhd":
        names = list(MHD_PARITY_Y)
        return names, MHD_PARITY_Y, MHD_PARITY_Z
    if kind == "z4c":
        names = list(Z4C_PARITY_Y)
        return names, Z4C_PARITY_Y, Z4C_PARITY_Z
    if kind == "adm":
        names = list(ADM_PARITY_Y)
        return names, ADM_PARITY_Y, ADM_PARITY_Z
    return [], {}, {}


def as_plane(data: Dict[str, np.ndarray], quantity: str) -> np.ndarray:
    arr = np.asarray(data[quantity])
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D slice for {quantity}, got shape {arr.shape}")
    return arr.astype(np.float64, copy=False)


def plane_coords(data: Dict[str, np.ndarray], plane: str) -> Tuple[np.ndarray, np.ndarray, str]:
    x = np.asarray(data["x1v"])
    if plane == "xy":
        return x, np.asarray(data["x2v"]), "y"
    if plane == "xz":
        return x, np.asarray(data["x3v"]), "z"
    raise ValueError(f"Unsupported plane {plane}")


def density_context(data: Dict[str, np.ndarray], plane: str) -> Optional[Dict[str, object]]:
    if "dens" not in data:
        return None
    x, mirror_coord, axis_name = plane_coords(data, plane)
    dens = as_plane(data, "dens")
    dens_for_center = np.nan_to_num(np.abs(dens), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    _, i_peak = np.unravel_index(np.argmax(dens_for_center), dens_for_center.shape)
    return {
        "x": x,
        "mirror_coord": mirror_coord,
        "axis_name": axis_name,
        "density": dens,
        "x_center": float(x[i_peak]),
    }


def build_masks(density: Optional[np.ndarray]) -> Dict[str, Optional[np.ndarray]]:
    masks: Dict[str, Optional[np.ndarray]] = {"all": None}
    if density is not None:
        for threshold in MASK_THRESHOLDS:
            masks[f"rho_gt_{threshold:.0e}"] = density > threshold
    return masks


def mirror_metric(
    x: np.ndarray,
    mirror_coord: np.ndarray,
    field: np.ndarray,
    parity: int,
    x_center: Optional[float],
    x_half_width: Optional[float],
    mirror_half_width: Optional[float],
    pair_mask: Optional[np.ndarray],
) -> Dict[str, object]:
    finite = np.isfinite(field)
    if x_center is None:
        field_for_center = np.nan_to_num(
            np.abs(field), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        _, i_peak = np.unravel_index(np.argmax(field_for_center), field_for_center.shape)
        x_center = float(x[i_peak])

    x_mask = np.ones_like(x, dtype=bool)
    mirror_axis_mask = np.ones_like(mirror_coord, dtype=bool)
    if x_half_width is not None:
        x_mask &= np.abs(x - x_center) <= x_half_width
    if mirror_half_width is not None:
        mirror_axis_mask &= np.abs(mirror_coord) <= mirror_half_width

    x_sub = x[x_mask]
    mirror_sub = mirror_coord[mirror_axis_mask]
    values = field[np.ix_(mirror_axis_mask, x_mask)]
    finite_sub = finite[np.ix_(mirror_axis_mask, x_mask)]
    mask_sub = None
    if pair_mask is not None:
        mask_sub = pair_mask[np.ix_(mirror_axis_mask, x_mask)]

    peak_values = np.abs(values[finite_sub])
    peak_abs = float(np.max(peak_values)) if peak_values.size else 0.0

    l2_abs_num = 0.0
    l2_local_rel_num = 0.0
    linf_abs: Optional[float] = None
    linf_local_rel: Optional[float] = None
    pair_count = 0
    max_pair = None

    for j, coord in enumerate(mirror_sub):
        jm = int(np.argmin(np.abs(mirror_sub + coord)))
        if j >= jm:
            continue
        lhs = values[j, :]
        rhs = parity * values[jm, :]
        valid = finite_sub[j, :] & finite_sub[jm, :]
        if mask_sub is not None:
            valid &= mask_sub[j, :] & mask_sub[jm, :]
        if not np.any(valid):
            continue

        diff = np.abs(lhs[valid] - rhs[valid])
        local_scale = np.maximum(np.maximum(np.abs(lhs[valid]), np.abs(rhs[valid])), TINY)
        local_rel = diff / local_scale
        l2_abs_num += float(np.sum(diff**2))
        l2_local_rel_num += float(np.sum(local_rel**2))
        pair_count += int(np.count_nonzero(valid))

        local_linf_abs = float(np.max(diff))
        local_linf_rel = float(np.max(local_rel))
        linf_abs = local_linf_abs if linf_abs is None else max(linf_abs, local_linf_abs)
        linf_local_rel = (
            local_linf_rel if linf_local_rel is None else max(linf_local_rel, local_linf_rel)
        )

        local = int(np.argmax(diff))
        valid_indices = np.flatnonzero(valid)
        ix = int(valid_indices[local])
        if max_pair is None or float(diff[local]) > max_pair["abs_diff"]:
            max_pair = {
                "abs_diff": float(diff[local]),
                "local_rel_diff": float(local_rel[local]),
                "peak_rel_diff": float(diff[local] / peak_abs) if peak_abs > 0.0 else None,
                "x": float(x_sub[ix]),
                "mirror_a": float(mirror_sub[j]),
                "mirror_b": float(mirror_sub[jm]),
                "value_a": float(lhs[ix]),
                "parity_value_b": float(rhs[ix]),
            }

    l2_abs = float(np.sqrt(l2_abs_num)) if pair_count else None
    l2_local_rel = float(np.sqrt(l2_local_rel_num / pair_count)) if pair_count else None
    linf_peak_rel = (
        float(linf_abs / peak_abs) if linf_abs is not None and peak_abs > 0.0 else None
    )
    l2_peak_rel = (
        float(l2_abs / (np.sqrt(pair_count) * peak_abs))
        if l2_abs is not None and pair_count and peak_abs > 0.0 else None
    )
    return {
        "x_center": x_center,
        "peak_abs": peak_abs,
        "l2_abs": l2_abs,
        "linf_abs": linf_abs,
        "l2_local_rel": l2_local_rel,
        "linf_local_rel": linf_local_rel,
        "l2_peak_rel": l2_peak_rel,
        "linf_peak_rel": linf_peak_rel,
        "pair_count": pair_count,
        "max_pair": max_pair,
    }


def mirror_difference(field: np.ndarray, parity: int) -> np.ndarray:
    return field - parity*np.flip(field, axis=0)


def summarize_file(
    path: Path,
    plane: str,
    kind: str,
    quantities: List[str],
    parity: Dict[str, int],
    density: Optional[np.ndarray],
    x_center: Optional[float],
    x_half_width: Optional[float],
    mirror_half_width: Optional[float],
) -> Dict[str, object]:
    data = read_slice(path, quantities)
    raw = read_raw(path)
    x, mirror_coord, axis_name = plane_coords(data, plane)
    masks = build_masks(density)
    quantity_metrics = {}
    for quantity in quantities:
        field = as_plane(data, quantity)
        quantity_metrics[quantity] = {}
        for mask_name, mask in masks.items():
            quantity_metrics[quantity][mask_name] = mirror_metric(
                x, mirror_coord, field, parity.get(quantity, 1), x_center,
                x_half_width, mirror_half_width, mask)
    return {
        "file": str(path),
        "plane": plane,
        "mirror_axis": axis_name,
        "kind": kind,
        "output_id": output_id(path),
        "output_number": output_number(path),
        "time": float(raw.get("time", np.nan)),
        "cycle": int(raw.get("cycle", -1)),
        "quantities": quantity_metrics,
    }


def flatten_results(results: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for output in results["outputs"]:
        for quantity, mask_metrics in output["quantities"].items():
            for mask_name, metrics in mask_metrics.items():
                max_pair = metrics.get("max_pair") or {}
                rows.append({
                    "case": results["case"],
                    "plane": output["plane"],
                    "mirror_axis": output["mirror_axis"],
                    "kind": output["kind"],
                    "output_id": output["output_id"],
                    "output_number": output["output_number"],
                    "time": output["time"],
                    "cycle": output["cycle"],
                    "quantity": quantity,
                    "mask": mask_name,
                    "pair_count": metrics["pair_count"],
                    "x_center": metrics["x_center"],
                    "peak_abs": metrics["peak_abs"],
                    "l2_abs": metrics["l2_abs"],
                    "linf_abs": metrics["linf_abs"],
                    "l2_local_rel": metrics["l2_local_rel"],
                    "linf_local_rel": metrics["linf_local_rel"],
                    "l2_peak_rel": metrics["l2_peak_rel"],
                    "linf_peak_rel": metrics["linf_peak_rel"],
                    "max_abs_diff": max_pair.get("abs_diff"),
                    "max_local_rel_diff": max_pair.get("local_rel_diff"),
                    "max_peak_rel_diff": max_pair.get("peak_rel_diff"),
                    "max_x": max_pair.get("x"),
                    "max_mirror_a": max_pair.get("mirror_a"),
                    "max_mirror_b": max_pair.get("mirror_b"),
                    "max_value_a": max_pair.get("value_a"),
                    "max_parity_value_b": max_pair.get("parity_value_b"),
                })
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    columns = [
        "case", "plane", "mirror_axis", "kind", "output_id", "output_number",
        "time", "cycle", "quantity", "mask", "pair_count", "x_center", "peak_abs",
        "l2_abs", "linf_abs", "l2_local_rel", "linf_local_rel",
        "l2_peak_rel", "linf_peak_rel", "max_abs_diff", "max_local_rel_diff",
        "max_peak_rel_diff", "max_x", "max_mirror_a", "max_mirror_b",
        "max_value_a", "max_parity_value_b",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_norms(out_dir: Path, rows: List[Dict[str, object]]) -> None:
    plot_rows = [
        row for row in rows
        if row["mask"] in ("all", "rho_gt_1e-10")
        and row["quantity"] in (
            "dens", "eint", "press", "temperature", "velx", "vely", "velz",
            "z4c_chi", "z4c_Gamy", "z4c_Gamz", "z4c_alpha",
            "adm_gxx", "adm_gxy", "adm_gxz", "adm_psi4",
        )
    ]
    if not plot_rows:
        return
    groups = sorted({
        (str(r["plane"]), str(r["output_id"]), str(r["quantity"]), str(r["mask"]))
        for r in plot_rows
    })
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for plane, out_id, quantity, mask_name in groups:
        group = [
            r for r in plot_rows
            if r["plane"] == plane and r["output_id"] == out_id
            and r["quantity"] == quantity and r["mask"] == mask_name
        ]
        group.sort(key=lambda r: (
            float(r["time"]) if np.isfinite(float(r["time"])) else float(r["output_number"])))
        xvals = np.asarray([
            float(r["time"]) if np.isfinite(float(r["time"])) else float(r["output_number"])
            for r in group
        ])
        l2 = np.asarray([
            float(r["l2_peak_rel"]) if r["l2_peak_rel"] not in (None, "") else np.nan
            for r in group
        ])
        linf = np.asarray([
            float(r["linf_peak_rel"]) if r["linf_peak_rel"] not in (None, "") else np.nan
            for r in group
        ])
        label = f"{plane}:{quantity}:{mask_name}"
        axes[0].plot(xvals, l2, marker="o", label=label)
        axes[1].plot(xvals, linf, marker="o", label=label)
    axes[0].set_ylabel("peak-relative L2")
    axes[1].set_ylabel("peak-relative Linf")
    axes[1].set_xlabel("time")
    for ax in axes:
        ydata = np.concatenate([line.get_ydata() for line in ax.lines]) if ax.lines else np.asarray([])
        if np.any(np.isfinite(ydata) & (ydata > 0.0)):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "symmetry_norms_peak_relative.png", dpi=160)
    plt.close(fig)


def latest_outputs(outputs: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    latest: Dict[str, Dict[str, object]] = {}
    for output in outputs:
        key = f"{output['plane']}:{output['output_id']}"
        prev = latest.get(key)
        if prev is None or int(output["cycle"]) > int(prev["cycle"]):
            latest[key] = output
    return latest


def plot_difference_slices(out_dir: Path, outputs: List[Dict[str, object]]) -> None:
    for output in latest_outputs(outputs).values():
        path = Path(str(output["file"]))
        kind = str(output["kind"])
        plane = str(output["plane"])
        requested, parity_y, parity_z = requested_for(kind)
        parity = parity_y if plane == "xy" else parity_z
        if not requested:
            continue
        quantities = list(output["quantities"].keys())
        if kind == "mhd":
            quantities = [
                q for q in ("dens", "eint", "press", "temperature", "velx", "vely", "velz")
                if q in quantities
            ]
        elif kind == "z4c":
            quantities = [q for q in ("z4c_chi", "z4c_Gamy", "z4c_Gamz", "z4c_alpha") if q in quantities]
        elif kind == "adm":
            quantities = [q for q in ("adm_gxx", "adm_gxy", "adm_gxz", "adm_psi4") if q in quantities]
        if not quantities:
            continue
        data = read_slice(path, quantities)
        x, mirror_coord, axis_name = plane_coords(data, plane)
        for quantity in quantities:
            field = as_plane(data, quantity)
            diff = mirror_difference(field, parity.get(quantity, 1))
            vmax = np.nanmax(np.abs(diff))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(7, 4.5))
            im = ax.pcolormesh(x, mirror_coord, diff, shading="auto", cmap="coolwarm",
                               vmin=-vmax, vmax=vmax)
            ax.set_xlabel("x")
            ax.set_ylabel(axis_name)
            ax.set_title(
                f"{output['output_id']} {quantity} mirror diff, cycle {output['cycle']}")
            fig.colorbar(im, ax=ax, label=f"{quantity} - parity*mirror")
            fig.tight_layout()
            fig.savefig(
                out_dir / f"{output['output_id']}_{quantity}_mirror_diff.png", dpi=160)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute y/z-reflection symmetry metrics for TOV diagnostic slices.")
    parser.add_argument("--run-dir", "--input-dir", dest="run_dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/post"))
    parser.add_argument("--x-half-width", type=float, default=2.0)
    parser.add_argument("--mirror-half-width", "--y-half-width", dest="mirror_half_width",
                        type=float, default=2.0)
    args = parser.parse_args()

    files = find_files(args.run_dir)
    results: Dict[str, object] = {
        "case": args.case,
        "run_dir": str(args.run_dir),
        "mask_thresholds": list(MASK_THRESHOLDS),
        "outputs": [],
    }
    density_by_key: Dict[Tuple[str, str], Dict[str, object]] = {}

    for path in files:
        out_id = output_id(path)
        plane = plane_for_output_id(out_id)
        kind = kind_for_output_id(out_id)
        if plane is None or kind != "mhd":
            continue
        requested, parity_y, parity_z = requested_for(kind)
        quantities = available_quantities(path, requested)
        if not quantities:
            continue
        data = read_slice(path, quantities)
        density_by_key[(plane, output_number(path))] = density_context(data, plane) or {}
        x, mirror_coord, _ = plane_coords(data, plane)
        dens_ctx = density_by_key[(plane, output_number(path))]
        density = dens_ctx.get("density") if dens_ctx else None
        x_center = dens_ctx.get("x_center") if dens_ctx else None
        parity = parity_y if plane == "xy" else parity_z
        raw = read_raw(path)
        metrics = {}
        for quantity in quantities:
            metrics[quantity] = {}
            for mask_name, mask in build_masks(density).items():
                metrics[quantity][mask_name] = mirror_metric(
                    x, mirror_coord, as_plane(data, quantity), parity.get(quantity, 1),
                    x_center, args.x_half_width, args.mirror_half_width, mask)
        results["outputs"].append({
            "file": str(path),
            "plane": plane,
            "mirror_axis": "y" if plane == "xy" else "z",
            "kind": kind,
            "output_id": out_id,
            "output_number": output_number(path),
            "time": float(raw.get("time", np.nan)),
            "cycle": int(raw.get("cycle", -1)),
            "quantities": metrics,
        })

    for path in files:
        out_id = output_id(path)
        plane = plane_for_output_id(out_id)
        kind = kind_for_output_id(out_id)
        if plane is None or kind == "mhd":
            continue
        requested, parity_y, parity_z = requested_for(kind)
        quantities = available_quantities(path, requested)
        if not quantities:
            continue
        dens_ctx = density_by_key.get((plane, output_number(path)), {})
        density = dens_ctx.get("density") if dens_ctx else None
        x_center = dens_ctx.get("x_center") if dens_ctx else None
        parity = parity_y if plane == "xy" else parity_z
        results["outputs"].append(summarize_file(
            path, plane, kind, quantities, parity, density, x_center,
            args.x_half_width, args.mirror_half_width))

    out_dir = args.output_root / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "symmetry_metrics.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    rows = flatten_results(results)
    write_csv(out_dir / "symmetry_metrics.csv", rows)
    plot_norms(out_dir, rows)
    plot_difference_slices(out_dir, results["outputs"])
    print(out_path)


if __name__ == "__main__":
    main()
