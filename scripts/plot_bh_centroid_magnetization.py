#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/hzhu/athenak/vis/python")
import bin_convert


RUN_DIR = Path(__file__).resolve().parent
OUT_DIR = RUN_DIR.parent / "plot_check" / "magnetization"
PARFILE = RUN_DIR / "parfile.par"
BASE_VARS = ("rho", "press", "temp", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3")
DERIVED_VARS = ("b_sq", "b_mag", "sigma", "beta")
TEMP_ABS_LIMIT = float(os.environ.get("TEMP_ABS_LIMIT", "inf"))
TEMP_GROWTH_LIMIT = float(os.environ.get("TEMP_GROWTH_LIMIT", "1.0e3"))


def read_par_value(parfile, name, default):
    if not parfile.exists():
        return default
    pattern = re.compile(r"^\s*" + re.escape(name) + r"\s*=\s*([^#\s]+)")
    for raw in parfile.read_text().splitlines():
        match = pattern.match(raw)
        if match:
            try:
                return float(match.group(1))
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


def derived_values(values):
    b_sq = values["bcc1"] ** 2 + values["bcc2"] ** 2 + values["bcc3"] ** 2
    return {
        "b_sq": b_sq,
        "b_mag": np.sqrt(b_sq),
        "sigma": b_sq / np.maximum(values["rho"], 1.0e-300),
        "beta": 2.0 * values["press"] / np.maximum(b_sq, 1.0e-300),
    }


def sample_point(data, x, y):
    best = None
    geoms = data["mb_geometry"]
    for mb, geom in enumerate(geoms):
        x0, x1, y0, y1, _z0, _z1 = geom
        if not (x0 <= x < x1 and y0 <= y < y1):
            continue
        dens = np.asarray(data["mb_data"]["dens"][mb])
        nz, ny, nx = dens.shape
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        ix = int(np.clip(math.floor((x - x0) / dx), 0, nx - 1))
        iy = int(np.clip(math.floor((y - y0) / dy), 0, ny - 1))
        iz = 0
        area = dx * dy
        if best is None or area < best["area"]:
            values = {
                "rho": float(dens[iz, iy, ix]),
                "press": float(np.asarray(data["mb_data"]["press"][mb])[iz, iy, ix]),
                "temp": float(np.asarray(data["mb_data"]["temperature"][mb])[iz, iy, ix]),
                "velx": float(np.asarray(data["mb_data"]["velx"][mb])[iz, iy, ix]),
                "vely": float(np.asarray(data["mb_data"]["vely"][mb])[iz, iy, ix]),
                "velz": float(np.asarray(data["mb_data"]["velz"][mb])[iz, iy, ix]),
                "bcc1": float(np.asarray(data["mb_data"]["bcc1"][mb])[iz, iy, ix]),
                "bcc2": float(np.asarray(data["mb_data"]["bcc2"][mb])[iz, iy, ix]),
                "bcc3": float(np.asarray(data["mb_data"]["bcc3"][mb])[iz, iy, ix]),
            }
            values.update({key: float(value) for key, value in derived_values(values).items()})
            best = {
                "area": area,
                "dx": dx,
                "dy": dy,
                "x_cell": x0 + (ix + 0.5) * dx,
                "y_cell": y0 + (iy + 0.5) * dy,
            }
            best.update(values)
    if best is None:
        raise RuntimeError("No slice block contains point ({:.6g}, {:.6g})".format(x, y))
    return best


def average_in_radius(data, x, y, radius):
    totals = {key: 0.0 for key in BASE_VARS + DERIVED_VARS}
    mins = {key: math.inf for key in BASE_VARS + DERIVED_VARS}
    maxs = {key: -math.inf for key in BASE_VARS + DERIVED_VARS}
    area_total = 0.0
    ncell = 0
    geoms = data["mb_geometry"]
    for mb, geom in enumerate(geoms):
        x0, x1, y0, y1, _z0, _z1 = geom
        if x1 < x - radius or x0 > x + radius or y1 < y - radius or y0 > y + radius:
            continue
        dens = np.asarray(data["mb_data"]["dens"][mb])[0]
        ny, nx = dens.shape
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny
        xc = x0 + (np.arange(nx) + 0.5) * dx
        yc = y0 + (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(xc, yc)
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        if not np.any(mask):
            continue

        values = {
            "rho": dens,
            "press": np.asarray(data["mb_data"]["press"][mb])[0],
            "temp": np.asarray(data["mb_data"]["temperature"][mb])[0],
            "velx": np.asarray(data["mb_data"]["velx"][mb])[0],
            "vely": np.asarray(data["mb_data"]["vely"][mb])[0],
            "velz": np.asarray(data["mb_data"]["velz"][mb])[0],
            "bcc1": np.asarray(data["mb_data"]["bcc1"][mb])[0],
            "bcc2": np.asarray(data["mb_data"]["bcc2"][mb])[0],
            "bcc3": np.asarray(data["mb_data"]["bcc3"][mb])[0],
        }
        values.update(derived_values(values))
        area = dx * dy
        count = int(np.count_nonzero(mask))
        area_total += area * count
        ncell += count
        for key, arr in values.items():
            selected = np.asarray(arr)[mask]
            totals[key] += float(np.sum(selected, dtype=np.float64) * area)
            finite = selected[np.isfinite(selected)]
            if finite.size:
                mins[key] = min(mins[key], float(np.min(finite)))
                maxs[key] = max(maxs[key], float(np.max(finite)))

    if area_total == 0.0:
        raise RuntimeError("No slice cells within radius {:.6g} of ({:.6g}, {:.6g})".format(radius, x, y))

    result = {
        "area": area_total,
        "ncell": ncell,
        "radius": radius,
    }
    for key in BASE_VARS + DERIVED_VARS:
        result[key] = totals[key] / area_total
        result[key + "_min"] = mins[key] if math.isfinite(mins[key]) else math.nan
        result[key + "_max"] = maxs[key] if math.isfinite(maxs[key]) else math.nan
    result["sigma_from_avg"] = result["b_sq"] / max(result["rho"], 1.0e-300)
    result["beta_from_avg"] = 2.0 * result["press"] / max(result["b_sq"], 1.0e-300)
    return result


def global_field_health(data):
    fields = ("dens", "press", "temperature", "bcc1", "bcc2", "bcc3")
    result = {}
    for field in fields:
        arr = np.concatenate([np.asarray(a).ravel() for a in data["mb_data"][field]])
        finite = np.isfinite(arr)
        result[field + "_nonfinite"] = int(arr.size - np.count_nonzero(finite))
        result[field + "_nan"] = int(np.count_nonzero(np.isnan(arr)))
        result[field + "_inf"] = int(np.count_nonzero(np.isinf(arr)))
        if np.any(finite):
            finite_arr = arr[finite]
            result[field + "_finite_min"] = float(np.min(finite_arr))
            result[field + "_finite_max"] = float(np.max(finite_arr))
        else:
            result[field + "_finite_min"] = math.nan
            result[field + "_finite_max"] = math.nan
    return result


def finite_or_nan(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def main():
    parser = argparse.ArgumentParser(
        description="Measure and plot BH-centroid and sink-radius diagnostics from AthenaK slice_x3 binaries."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DIR,
        help="run directory containing bin/torus.slice_x3.*.bin",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="output directory for CSV files and diagnostic plots",
    )
    parser.add_argument(
        "--parfile",
        type=Path,
        help="parfile to read sep/q/sink_radius from; defaults to RUN_DIR/parfile.par",
    )
    parser.add_argument(
        "--sink-radius",
        type=float,
        help="override sink averaging radius",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="process every Nth slice_x3 binary file",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir
    parfile = args.parfile if args.parfile is not None else run_dir / "parfile.par"

    out_dir.mkdir(parents=True, exist_ok=True)
    sep = read_par_value(parfile, "sep", 25.0)
    q = read_par_value(parfile, "q", 1.0)
    sink_radius = (
        args.sink_radius
        if args.sink_radius is not None
        else read_par_value(parfile, "sink_radius", 4.0)
    )
    rows = []
    paths = sorted(glob.glob(str(run_dir / "bin" / "torus.slice_x3.*.bin")))
    paths = paths[:: max(args.stride, 1)]
    for path in paths:
        data = bin_convert.read_binary(path)
        time = float(data["time"])
        cycle = int(data["cycle"])
        health = global_field_health(data)
        pos1, pos2 = bh_positions(time, sep, q)
        sample1 = sample_point(data, pos1[0], pos1[1])
        sample2 = sample_point(data, pos2[0], pos2[1])
        avg1 = average_in_radius(data, pos1[0], pos1[1], sink_radius)
        avg2 = average_in_radius(data, pos2[0], pos2[1], sink_radius)
        row = {
            "file": os.path.basename(path),
            "time": time,
            "cycle": cycle,
            "sink_radius": sink_radius,
            "bh1_x": pos1[0],
            "bh1_y": pos1[1],
            "bh2_x": pos2[0],
            "bh2_y": pos2[1],
        }
        for key, value in health.items():
            row["global_" + key] = value
        for prefix, sample in (("bh1", sample1), ("bh2", sample2)):
            for key in ("x_cell", "y_cell", "dx", "dy") + BASE_VARS + DERIVED_VARS:
                row[prefix + "_point_" + key] = sample[key]
        for prefix, avg in (("bh1", avg1), ("bh2", avg2)):
            avg_keys = ("area", "ncell", "radius", "sigma_from_avg", "beta_from_avg")
            for key in avg_keys:
                row[prefix + "_avg_" + key] = avg[key]
            for key in BASE_VARS + DERIVED_VARS:
                row[prefix + "_avg_" + key] = avg[key]
                row[prefix + "_avg_" + key + "_min"] = avg[key + "_min"]
                row[prefix + "_avg_" + key + "_max"] = avg[key + "_max"]
        rows.append(row)

    if not rows:
        raise RuntimeError("No slice_x3 binary files found")

    csv_path = out_dir / "bh_centroid_magnetization_xy.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    times = np.array([r["time"] for r in rows])
    t0 = times[0]
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(times - t0, [r["bh1_point_sigma"] for r in rows], "o-", label="BH1 point")
    axes[0].plot(times - t0, [r["bh2_point_sigma"] for r in rows], "s-", label="BH2 point")
    axes[0].plot(times - t0, [r["bh1_avg_sigma"] for r in rows], "o--", label="BH1 avg")
    axes[0].plot(times - t0, [r["bh2_avg_sigma"] for r in rows], "s--", label="BH2 avg")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\sigma = B^2/\rho$")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times - t0, [r["bh1_point_b_sq"] for r in rows], "o-", label="BH1 point")
    axes[1].plot(times - t0, [r["bh2_point_b_sq"] for r in rows], "s-", label="BH2 point")
    axes[1].plot(times - t0, [r["bh1_avg_b_sq"] for r in rows], "o--", label="BH1 avg")
    axes[1].plot(times - t0, [r["bh2_avg_b_sq"] for r in rows], "s--", label="BH2 avg")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$B^2$")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times - t0, [r["bh1_point_rho"] for r in rows], "o-", label="BH1 point")
    axes[2].plot(times - t0, [r["bh2_point_rho"] for r in rows], "s-", label="BH2 point")
    axes[2].plot(times - t0, [r["bh1_avg_rho"] for r in rows], "o--", label="BH1 avg")
    axes[2].plot(times - t0, [r["bh2_avg_rho"] for r in rows], "s--", label="BH2 avg")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"$\rho$")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times - t0, [r["bh1_point_press"] for r in rows], "o-", label="BH1 point")
    axes[3].plot(times - t0, [r["bh2_point_press"] for r in rows], "s-", label="BH2 point")
    axes[3].plot(times - t0, [r["bh1_avg_press"] for r in rows], "o--", label="BH1 avg")
    axes[3].plot(times - t0, [r["bh2_avg_press"] for r in rows], "s--", label="BH2 avg")
    axes[3].set_yscale("log")
    axes[3].set_xlabel(r"$t - t_0$ (M)")
    axes[3].set_ylabel(r"$p$")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("XY-slice centroid point values and sink-radius averages")
    fig.tight_layout()
    fig.savefig(out_dir / "bh_centroid_magnetization_xy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for bh, marker in (("bh1", "o"), ("bh2", "s")):
        axes[0].plot(times - t0, [r[f"{bh}_avg_temp"] for r in rows], marker + "--", label=bh.upper())
        axes[1].plot(times - t0, [r[f"{bh}_avg_b_mag"] for r in rows], marker + "--", label=bh.upper())
        axes[2].plot(times - t0, [r[f"{bh}_avg_beta"] for r in rows], marker + "--", label=bh.upper())
    axes[0].set_yscale("log")
    axes[0].set_ylabel("avg temperature")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"avg $|B|$")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"avg $\beta$")
    axes[2].set_xlabel(r"$t - t_0$ (M)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("Sink-radius averaged thermodynamic and magnetic diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "bh_centroid_sink_averages_xy.png", dpi=180)
    plt.close(fig)

    temp0 = max(
        finite_or_nan(rows[0]["bh1_avg_temp"]),
        finite_or_nan(rows[0]["bh2_avg_temp"]),
        finite_or_nan(rows[0]["global_temperature_finite_max"]),
    )
    if not math.isfinite(temp0) or temp0 <= 0.0:
        temp0 = math.nan
    status_rows = []
    for row in rows:
        temp_values = [
            finite_or_nan(row["bh1_avg_temp"]),
            finite_or_nan(row["bh2_avg_temp"]),
            finite_or_nan(row["global_temperature_finite_max"]),
        ]
        max_temp = max([v for v in temp_values if math.isfinite(v)] or [math.nan])
        growth = max_temp / temp0 if math.isfinite(max_temp) and math.isfinite(temp0) else math.nan
        nonfinite_temp = int(row["global_temperature_nonfinite"])
        nonfinite_press = int(row["global_press_nonfinite"])
        status = "pass"
        reasons = []
        if nonfinite_temp > 0 or nonfinite_press > 0:
            status = "fail"
            reasons.append("nonfinite thermodynamic cells")
        if math.isfinite(TEMP_ABS_LIMIT) and math.isfinite(max_temp) and max_temp > TEMP_ABS_LIMIT:
            status = "fail"
            reasons.append("temperature above absolute limit")
        if math.isfinite(growth) and growth > TEMP_GROWTH_LIMIT:
            status = "fail"
            reasons.append("temperature growth above limit")
        status_rows.append({
            "file": row["file"],
            "time": row["time"],
            "cycle": row["cycle"],
            "max_temperature": max_temp,
            "temperature_growth": growth,
            "temperature_nonfinite": nonfinite_temp,
            "pressure_nonfinite": nonfinite_press,
            "status": status,
            "reason": "; ".join(reasons),
        })

    health_csv = out_dir / "temperature_health_xy.csv"
    with health_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(status_rows[0].keys()))
        writer.writeheader()
        writer.writerows(status_rows)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(times - t0, [finite_or_nan(r["global_temperature_finite_max"]) for r in rows],
                 "k-", label="global finite max")
    axes[0].plot(times - t0, [finite_or_nan(r["bh1_avg_temp"]) for r in rows],
                 "o--", label="BH1 sink avg")
    axes[0].plot(times - t0, [finite_or_nan(r["bh2_avg_temp"]) for r in rows],
                 "s--", label="BH2 sink avg")
    if math.isfinite(TEMP_ABS_LIMIT):
        axes[0].axhline(TEMP_ABS_LIMIT, color="r", linestyle=":", label="absolute limit")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("temperature")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(times - t0, [r["temperature_nonfinite"] for r in status_rows],
                 "o-", label="temperature")
    axes[1].plot(times - t0, [r["pressure_nonfinite"] for r in status_rows],
                 "s-", label="pressure")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_xlabel(r"$t - t_0$ (M)")
    axes[1].set_ylabel("nonfinite cells")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle("Temperature health check")
    fig.tight_layout()
    fig.savefig(out_dir / "temperature_health_xy.png", dpi=180)
    plt.close(fig)

    first_fail = next((r for r in status_rows if r["status"] == "fail"), None)
    status_path = out_dir / "temperature_health_status.txt"
    with status_path.open("w") as handle:
        handle.write("temperature diagnostic status\n")
        handle.write("=============================\n")
        handle.write(f"TEMP_ABS_LIMIT = {TEMP_ABS_LIMIT:.6e}\n")
        handle.write(f"TEMP_GROWTH_LIMIT = {TEMP_GROWTH_LIMIT:.6e}\n")
        if first_fail is None:
            handle.write("overall_status = pass\n")
            handle.write("No nonfinite pressure/temperature cells and no configured temperature-limit failures.\n")
        else:
            handle.write("overall_status = fail\n")
            handle.write(
                "first_failure = {file}, time={time:.17g}, cycle={cycle}, "
                "max_temperature={max_temperature:.6e}, growth={temperature_growth:.6e}, "
                "temperature_nonfinite={temperature_nonfinite}, "
                "pressure_nonfinite={pressure_nonfinite}, reason={reason}\n".format(**first_fail))

    print("wrote", csv_path)
    print("wrote", out_dir / "bh_centroid_magnetization_xy.png")
    print("wrote", out_dir / "bh_centroid_sink_averages_xy.png")
    print("wrote", health_csv)
    print("wrote", out_dir / "temperature_health_xy.png")
    print("wrote", status_path)


if __name__ == "__main__":
    main()
