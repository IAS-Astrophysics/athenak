#!/usr/bin/env python3
"""Post-process AthenaK circumbinary-disk binary and history outputs.

The volume workflow intentionally combines separate AthenaK output groups:

* ``mhd_w`` or ``mhd_w_bcc`` supplies rest-frame density and pressure.
* ``mhd_u_d`` supplies Valencia ``D = sqrt(gamma) rho W`` for proper rest mass.
* ``angular_momentum`` and ``torque`` optionally supply the densitized
  diagnostics implemented by ``project/ttorus``.

All volume integrals multiply these densitized fields by coordinate cell
volume exactly once.  Native binary dumps contain leaf MeshBlocks, so no AMR
level is resampled or double counted.  Sliced and ghost-zone dumps are rejected.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SURFACE_QUANTITIES = (
    "mdot", "edot_f", "edot_em",
    "pxdot_f", "pydot_f", "pzdot_f",
    "pxdot_em", "pydot_em", "pzdot_em",
    "lxdot_f", "lydot_f", "lzdot_f",
    "lxdot_em", "lydot_em", "lzdot_em",
    "phiB", "area",
)


class DiagnosticsError(RuntimeError):
    """A user-facing input or data compatibility error."""


@dataclass
class BinaryBlock:
    logical_location: tuple[int, int, int, int]
    bounds: tuple[float, float, float, float, float, float]
    shape: tuple[int, int, int]
    data: dict[str, np.ndarray]

    @property
    def key(self):
        return self.logical_location


@dataclass
class BinaryDump:
    paths: tuple[Path, ...]
    time: float
    cycle: int
    variables: tuple[str, ...]
    parameters: dict[str, dict[str, str]]
    blocks: list[BinaryBlock]

    @property
    def run_name(self):
        return self.parameters.get("job", {}).get("basename", self.paths[0].stem)

    @property
    def run_root(self):
        parent = self.paths[0].resolve().parent
        return parent.parent if parent.name.startswith("rank_") else parent

    @property
    def match_key(self):
        return (self.run_root, self.run_name, self.cycle)


def parse_athena_input(text: str) -> dict[str, dict[str, str]]:
    """Parse block-scoped Athena input metadata without conflating repeated keys."""
    result: dict[str, dict[str, str]] = {}
    block = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"<([^>]+)>", line)
        if match:
            block = match.group(1).strip()
            result.setdefault(block, {})
            continue
        if block is not None and "=" in line:
            key, value = line.split("=", 1)
            result[block][key.strip()] = value.strip().strip("\"'")
    return result


def _read_binary_file(path: Path, load_blocks: bool = True) -> BinaryDump:
    path = path.resolve()
    with path.open("rb") as stream:
        first = stream.readline().decode("ascii", errors="strict").strip()
        if not first.startswith("Athena binary output version="):
            raise DiagnosticsError(f"{path}: not an AthenaK native binary output")
        metadata: dict[str, str] = {}
        variables: tuple[str, ...] | None = None
        while True:
            raw = stream.readline()
            if not raw:
                raise DiagnosticsError(f"{path}: truncated binary preheader")
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("variables:"):
                variables = tuple(line.split(":", 1)[1].split())
            elif "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip()] = value.strip()
            if line.startswith("header offset="):
                header_size = int(metadata["header offset"])
                header = stream.read(header_size).decode("utf-8", errors="strict")
                break
        if variables is None:
            raise DiagnosticsError(f"{path}: binary preheader has no variable list")
        nvar = int(metadata["number of variables"])
        if nvar != len(variables):
            raise DiagnosticsError(f"{path}: inconsistent binary variable count")
        parameters = parse_athena_input(header)
        if not load_blocks:
            return BinaryDump((path,), float(metadata["time"]),
                              int(metadata["cycle"]), variables, parameters, [])
        location_size = int(metadata["size of location"])
        variable_size = int(metadata["size of variable"])
        if location_size not in (4, 8) or variable_size not in (4, 8):
            raise DiagnosticsError(f"{path}: unsupported binary floating-point size")
        location_fmt = "f" if location_size == 4 else "d"
        variable_dtype = np.dtype("=f4" if variable_size == 4 else "=f8")
        blocks = []
        while True:
            raw_indices = stream.read(10*4)
            if not raw_indices:
                break
            if len(raw_indices) != 10*4:
                raise DiagnosticsError(f"{path}: truncated MeshBlock header")
            indices = struct.unpack("=10i", raw_indices)
            raw_bounds = stream.read(6*location_size)
            if len(raw_bounds) != 6*location_size:
                raise DiagnosticsError(f"{path}: truncated MeshBlock bounds")
            bounds = struct.unpack("=" + 6*location_fmt, raw_bounds)
            ni = indices[1] - indices[0] + 1
            nj = indices[3] - indices[2] + 1
            nk = indices[5] - indices[4] + 1
            count = nvar*ni*nj*nk
            raw_data = stream.read(count*variable_size)
            if len(raw_data) != count*variable_size:
                raise DiagnosticsError(f"{path}: truncated MeshBlock variables")
            values = np.frombuffer(raw_data, dtype=variable_dtype).reshape(
                (nvar, nk, nj, ni)).astype(np.float64, copy=False)
            block_data = {name: values[n] for n, name in enumerate(variables)}
            blocks.append(BinaryBlock(
                (indices[9], indices[6], indices[7], indices[8]),
                tuple(float(value) for value in bounds), (nk, nj, ni), block_data))
    return BinaryDump(
        (path,), float(metadata["time"]), int(metadata["cycle"]), variables,
        parameters, blocks)


def _expected_meshblock_shape(dump: BinaryDump) -> tuple[int, int, int] | None:
    block = dump.parameters.get("meshblock", {})
    try:
        return (int(block["nx3"]), int(block["nx2"]), int(block["nx1"]))
    except (KeyError, ValueError):
        return None


def validate_full_volume(dump: BinaryDump):
    expected = _expected_meshblock_shape(dump)
    if expected is None:
        raise DiagnosticsError(
            f"{dump.paths[0]}: metadata lacks meshblock/nx1,nx2,nx3")
    bad = [block.shape for block in dump.blocks if block.shape != expected]
    if bad:
        raise DiagnosticsError(
            f"{dump.paths[0]}: volume diagnostics require unsliced output without "
            f"ghost zones; expected block shape {expected}, found {bad[0]}")


def _shard_group_key(dump: BinaryDump):
    return (dump.run_root, dump.run_name, dump.time, dump.cycle, dump.variables)


def load_binary_sets(paths: Sequence[str | Path]) -> list[BinaryDump]:
    """Load outputs and merge ``single_file_per_rank`` shards by metadata."""
    groups: dict[tuple, list[BinaryDump]] = {}
    for path in paths:
        dump = _read_binary_file(Path(path))
        groups.setdefault(_shard_group_key(dump), []).append(dump)
    merged = []
    for shards in groups.values():
        first = shards[0]
        blocks = [block for shard in shards for block in shard.blocks]
        keys = [block.key for block in blocks]
        if len(keys) != len(set(keys)):
            shard_names = [str(shard.paths[0]) for shard in shards]
            raise DiagnosticsError(
                f"duplicate MeshBlocks while combining {shard_names}")
        dump = BinaryDump(tuple(s.paths[0] for s in shards), first.time, first.cycle,
                          first.variables, first.parameters, blocks)
        validate_full_volume(dump)
        merged.append(dump)
    return sorted(merged, key=lambda item: (str(item.run_root), item.time, item.cycle))


def group_binary_paths(paths: Sequence[str | Path], role: str):
    """Index many dumps without retaining their volume arrays in memory."""
    groups: dict[tuple, list[BinaryDump]] = {}
    for path in paths:
        header = _read_binary_file(Path(path), load_blocks=False)
        groups.setdefault(_shard_group_key(header), []).append(header)
    result = []
    match_keys = set()
    for shards in groups.values():
        first = shards[0]
        if first.match_key in match_keys:
            raise DiagnosticsError(f"multiple {role} dump sets match {first.match_key}")
        match_keys.add(first.match_key)
        result.append((first, tuple(shard.paths[0] for shard in shards)))
    return sorted(result, key=lambda item: (str(item[0].run_root),
                                             item[0].time, item[0].cycle))


def expand_paths(patterns: Iterable[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        if not matches:
            raise DiagnosticsError(f"no files match {pattern!r}")
        paths.extend(matches)
    return paths


def radial_bin_indices(radius, rmin: float, dr: float, nbins: int):
    radius = np.asarray(radius)
    finite = np.isfinite(radius)
    indices = np.full(radius.shape, -1, dtype=np.int64)
    indices[finite] = np.floor((radius[finite] - rmin)/dr).astype(np.int64)
    valid = finite & (indices >= 0) & (indices < nbins)
    return indices, valid


def annulus_areas(edges):
    edges = np.asarray(edges, dtype=np.float64)
    return math.pi*(edges[1:]**2 - edges[:-1]**2)


def _block_coordinates(block: BinaryBlock):
    nk, nj, ni = block.shape
    x1min, x1max, x2min, x2max, x3min, x3max = block.bounds
    x = x1min + (np.arange(ni) + 0.5)*(x1max - x1min)/ni
    y = x2min + (np.arange(nj) + 0.5)*(x2max - x2min)/nj
    z = x3min + (np.arange(nk) + 0.5)*(x3max - x3min)/nk
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dvolume = ((x1max - x1min)/ni * (x2max - x2min)/nj *
               (x3max - x3min)/nk)
    return xx, yy, zz, dvolume


def _block_map(dump: BinaryDump) -> dict[tuple[int, int, int, int], BinaryBlock]:
    return {block.key: block for block in dump.blocks}


def _matching_dump(primary: BinaryDump,
                   companions: Mapping[tuple, BinaryDump], role: str):
    try:
        companion = companions[primary.match_key]
    except KeyError as exc:
        raise DiagnosticsError(
            f"{primary.paths[0]}: no {role} dump with matching run, time, "
            "and cycle") from exc
    tolerance = 64.0*np.finfo(float).eps*max(1.0, abs(primary.time))
    if abs(companion.time - primary.time) > tolerance:
        raise DiagnosticsError(f"{primary.paths[0]}: {role} time does not match")
    if set(_block_map(companion)) != set(_block_map(primary)):
        raise DiagnosticsError(f"{primary.paths[0]}: {role} AMR hierarchy does not match")
    return companion


def analyze_volume_dump(primary: BinaryDump, conserved: BinaryDump,
                        angular: BinaryDump | None, torque: BinaryDump | None,
                        rmin: float, rmax: float | None, dr: float,
                        rho_min: float, radial_coordinate: str):
    if "dens" not in primary.variables:
        raise DiagnosticsError(f"{primary.paths[0]}: primitive dump lacks dens")
    if "dens" not in conserved.variables:
        raise DiagnosticsError(f"{conserved.paths[0]}: conserved dump lacks dens")
    if "adm" not in primary.parameters or "problem" not in primary.parameters:
        raise DiagnosticsError(
            f"{primary.paths[0]}: metadata is not a dyn-GRMHD dynbbh input")
    primary_blocks = _block_map(primary)
    conserved_blocks = _block_map(conserved)
    angular_names = ("Jx", "Jy", "Jz", "JEMx", "JEMy", "JEMz")
    torque_names = ("Tx", "Ty", "Tz")
    angular_blocks = _block_map(angular) if angular is not None else {}
    torque_blocks = _block_map(torque) if torque is not None else {}
    angular_total = np.zeros(len(angular_names))
    torque_total = np.zeros(len(torque_names))
    for dump, names in ((angular, angular_names), (torque, torque_names)):
        if dump is not None:
            missing = [name for name in names if name not in dump.variables]
            if missing:
                raise DiagnosticsError(
                    f"{dump.paths[0]}: missing {', '.join(missing)}")
    if rmax is None:
        def corner_radius(block):
            values = [abs(block.bounds[0]), abs(block.bounds[1]),
                      abs(block.bounds[2]), abs(block.bounds[3])]
            radial = math.hypot(max(values[:2]), max(values[2:]))
            if radial_coordinate == "spherical":
                radial = math.hypot(radial, max(abs(block.bounds[4]),
                                                abs(block.bounds[5])))
            return radial
        rmax = max(corner_radius(block) for block in primary.blocks)
    if not (dr > 0.0 and rmax > rmin >= 0.0 and rho_min >= 0.0):
        raise DiagnosticsError("require dr > 0, rmax > rmin >= 0, and rho-min >= 0")
    nbins = int(math.ceil((rmax - rmin)/dr))
    edges = rmin + dr*np.arange(nbins + 1, dtype=np.float64)
    mass = np.zeros(nbins)
    rho_integral = np.zeros(nbins)
    coordinate_volume = np.zeros(nbins)
    mass_radius = np.zeros(nbins)
    density_max = -math.inf
    density_max_position = np.full(3, np.nan)
    pressure_min = math.inf
    pressure_max = -math.inf

    for key, block in primary_blocks.items():
        cblock = conserved_blocks[key]
        if cblock.shape != block.shape or not np.allclose(cblock.bounds, block.bounds):
            raise DiagnosticsError(
                f"{primary.paths[0]}: companion block geometry differs")
        xx, yy, zz, dvolume = _block_coordinates(block)
        rho = block.data["dens"]
        densitized_d = cblock.data["dens"]
        radius = np.sqrt(xx*xx + yy*yy)
        if radial_coordinate == "spherical":
            radius = np.sqrt(radius*radius + zz*zz)
        indices, valid = radial_bin_indices(radius, rmin, dr, nbins)
        valid &= np.isfinite(rho) & np.isfinite(densitized_d) & (rho >= rho_min)
        if np.any(densitized_d[valid] < 0.0):
            raise DiagnosticsError(f"{conserved.paths[0]}: negative Valencia density")
        if np.any(valid):
            bins = indices[valid]
            cell_mass = densitized_d[valid]*dvolume
            mass += np.bincount(bins, weights=cell_mass, minlength=nbins)
            rho_integral += np.bincount(
                bins, weights=rho[valid]*dvolume, minlength=nbins)
            coordinate_volume += np.bincount(
                bins, weights=np.full(np.count_nonzero(valid), dvolume), minlength=nbins)
            mass_radius += np.bincount(
                bins, weights=cell_mass*radius[valid], minlength=nbins)
            if angular is not None:
                ablock = angular_blocks[key]
                if ablock.shape != block.shape or not np.allclose(
                        ablock.bounds, block.bounds):
                    raise DiagnosticsError(
                        f"{angular.paths[0]}: angular block geometry differs")
                values = np.stack([ablock.data[name][valid]
                                   for name in angular_names])
                if not np.all(np.isfinite(values)):
                    raise DiagnosticsError(
                        f"{angular.paths[0]}: non-finite angular momentum")
                angular_total += np.sum(values, axis=1)*dvolume
            if torque is not None:
                tblock = torque_blocks[key]
                if tblock.shape != block.shape or not np.allclose(
                        tblock.bounds, block.bounds):
                    raise DiagnosticsError(
                        f"{torque.paths[0]}: torque block geometry differs")
                values = np.stack([tblock.data[name][valid]
                                   for name in torque_names])
                if not np.all(np.isfinite(values)):
                    raise DiagnosticsError(f"{torque.paths[0]}: non-finite torque")
                torque_total += np.sum(values, axis=1)*dvolume
            flat_index = np.nanargmax(np.where(valid, rho, np.nan))
            if rho.flat[flat_index] > density_max:
                density_max = float(rho.flat[flat_index])
                density_max_position = np.array(
                    [xx.flat[flat_index], yy.flat[flat_index], zz.flat[flat_index]])
        if "press" in block.data:
            finite_pressure = block.data["press"][
                valid & np.isfinite(block.data["press"])]
            if finite_pressure.size:
                pressure_min = min(pressure_min, float(np.min(finite_pressure)))
                pressure_max = max(pressure_max, float(np.max(finite_pressure)))

    total_mass = float(np.sum(mass))
    if not math.isfinite(density_max):
        raise DiagnosticsError(
            f"{primary.paths[0]}: primitive density has no finite cells")
    characteristic_radius = (float(np.sum(mass_radius))/total_mass
                             if total_mass > 0.0 else math.nan)
    rho_mean = np.divide(rho_integral, coordinate_volume,
                         out=np.full(nbins, np.nan), where=coordinate_volume > 0.0)
    full_annulus = np.zeros(nbins, dtype=bool)
    surface_density = np.full(nbins, np.nan)
    if radial_coordinate == "cylindrical":
        mesh = primary.parameters.get("mesh", {})
        try:
            full_radius = min(float(mesh["x1max"]), -float(mesh["x1min"]),
                              float(mesh["x2max"]), -float(mesh["x2min"]))
        except (KeyError, ValueError):
            full_radius = -math.inf
        full_annulus = edges[1:] <= full_radius
        surface_density[full_annulus] = (
            mass[full_annulus]/annulus_areas(edges)[full_annulus])
    summary = {
        "run": primary.run_name,
        "run_directory": str(primary.run_root),
        "source": ";".join(str(path) for path in primary.paths),
        "time": primary.time,
        "cycle": primary.cycle,
        "disk_rest_mass": total_mass,
        "characteristic_radius": characteristic_radius,
        "density_max": density_max,
        "density_max_radius": float(np.linalg.norm(density_max_position[:2])
                                    if radial_coordinate == "cylindrical"
                                    else np.linalg.norm(density_max_position)),
        "density_max_x": density_max_position[0],
        "density_max_y": density_max_position[1],
        "density_max_z": density_max_position[2],
        "pressure_min": pressure_min if math.isfinite(pressure_min) else math.nan,
        "pressure_max": pressure_max if math.isfinite(pressure_max) else math.nan,
    }
    problem = primary.parameters.get("problem", {})
    for name in ("sep", "q", "a1", "a2", "use_traj_table"):
        summary[name] = problem.get(name, "")
    if angular is not None:
        for name, value in zip(("Jgas_x", "Jgas_y", "Jgas_z",
                                "Jem_x", "Jem_y", "Jem_z"), angular_total):
            summary[name] = value
        summary["Jgas_magnitude"] = float(np.linalg.norm(angular_total[:3]))
        summary["Jem_magnitude"] = float(np.linalg.norm(angular_total[3:]))
    if torque is not None:
        for name, value in zip(("torque_x", "torque_y", "torque_z"),
                               torque_total):
            summary[name] = value
        summary["torque_magnitude"] = float(np.linalg.norm(torque_total))
    profile = {
        "r_inner": edges[:-1], "r_outer": edges[1:],
        "radius": 0.5*(edges[:-1] + edges[1:]),
        "coordinate_volume": coordinate_volume,
        "rho_coordinate_volume_mean": rho_mean,
        "rest_mass": mass,
        "surface_density_coordinate_area": surface_density,
        "full_annulus_coverage": full_annulus,
    }
    return summary, profile


def read_history(path: str | Path):
    """Read Athena history files, including repeated headers after restarts."""
    records = []
    labels = None
    with Path(path).open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") and "[1]=" in line:
                numbered = re.findall(r"\[(\d+)\]=(\S+)", line)
                numbered.sort(key=lambda item: int(item[0]))
                labels = [name for _, name in numbered]
                if len(labels) != len(set(labels)):
                    raise DiagnosticsError(
                        f"{path}: truncated history labels are not unique")
            elif not line.startswith("#"):
                if labels is None:
                    raise DiagnosticsError(f"{path}: data precedes history header")
                values = np.fromstring(line, sep=" ")
                if values.size != len(labels):
                    raise DiagnosticsError(
                        f"{path}: history row has {values.size} values for "
                        f"{len(labels)} labels")
                records.append(dict(zip(labels, values)))
    if not records:
        raise DiagnosticsError(f"{path}: no history records")
    return records


def extract_surface_fluxes(records, source=""):
    """Convert current 17-field dynbbh surface groups into tidy records."""
    result = []
    for record in records:
        labels = list(record)
        for start, label in enumerate(labels):
            if not label.startswith("mdot_"):
                continue
            surface = label[len("mdot_"):]
            group = labels[start:start + len(SURFACE_QUANTITIES)]
            expected = [f"{quantity}_{surface}"[:10] for quantity in SURFACE_QUANTITIES]
            if group != expected:
                continue
            for quantity, field in zip(SURFACE_QUANTITIES, group):
                result.append({
                    "source": source,
                    "time": record.get("time", math.nan),
                    "dt": record.get("dt", math.nan),
                    "surface": surface,
                    "quantity": quantity,
                    "value": record[field],
                })
    return result


def _safe_label(summary):
    directory = Path(str(summary["run_directory"]))
    directory_label = directory.parent.name if directory.name == "bin" else directory.name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_",
                  f"{directory_label}_{summary['run']}_cycle{int(summary['cycle']):08d}")


def write_csv(path: Path, rows: Sequence[Mapping]):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_volume(args):
    primary_groups = group_binary_paths(expand_paths(args.primitive), "primitive")
    conserved_groups = {
        header.match_key: paths for header, paths in group_binary_paths(
            expand_paths(args.conserved), "conserved-density")}
    angular_groups = ({
        header.match_key: paths for header, paths in group_binary_paths(
            expand_paths(args.angular), "angular")}
        if args.angular else {})
    torque_groups = ({
        header.match_key: paths for header, paths in group_binary_paths(
            expand_paths(args.torque), "torque")}
        if args.torque else {})
    output_dir = Path(args.output).resolve()
    summaries = []
    for header, paths in primary_groups:
        dump = load_binary_sets(paths)[0]
        try:
            cons = load_binary_sets(conserved_groups[header.match_key])[0]
        except KeyError as exc:
            raise DiagnosticsError(
                f"{header.paths[0]}: no matching conserved-density dump") from exc
        _matching_dump(dump, {cons.match_key: cons}, "conserved-density")
        ang = None
        if angular_groups:
            try:
                ang = load_binary_sets(angular_groups[header.match_key])[0]
            except KeyError as exc:
                raise DiagnosticsError(
                    f"{header.paths[0]}: no matching angular dump") from exc
            _matching_dump(dump, {ang.match_key: ang}, "angular")
        tor = None
        if torque_groups:
            try:
                tor = load_binary_sets(torque_groups[header.match_key])[0]
            except KeyError as exc:
                raise DiagnosticsError(
                    f"{header.paths[0]}: no matching torque dump") from exc
            _matching_dump(dump, {tor.match_key: tor}, "torque")
        summary, profile = analyze_volume_dump(
            dump, cons, ang, tor, args.rmin, args.rmax, args.dr,
            args.rho_min, args.radial_coordinate)
        summaries.append(summary)
        profile_rows = [dict(zip(profile, values)) for values in zip(*profile.values())]
        write_csv(output_dir / "profiles" / f"{_safe_label(summary)}.csv", profile_rows)
    summaries.sort(key=lambda row: (str(row["run"]), float(row["time"])))
    write_csv(output_dir / "summary.csv", summaries)
    print(f"Wrote {len(summaries)} dump summaries to {output_dir}")


def run_history(args):
    raw_rows = []
    flux_rows = []
    for name in expand_paths(args.history):
        records = read_history(name)
        source = str(Path(name).resolve())
        raw_rows.extend({"source": source, **record} for record in records)
        flux_rows.extend(extract_surface_fluxes(records, str(Path(name).resolve())))
    output_dir = Path(args.output).resolve()
    write_csv(output_dir / "history.csv", raw_rows)
    if flux_rows:
        write_csv(output_dir / "surface_fluxes.csv", flux_rows)
    elif not args.raw_only_ok:
        raise DiagnosticsError(
            "no complete current-format dynbbh surface groups found; use "
            "--raw-only-ok to export only generic history columns")
    print(f"Wrote history diagnostics to {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Maintained AthenaK circumbinary-disk diagnostics",
        epilog=("Use quoted globs for repository-independent batch processing. "
                "Companion dumps are matched by run basename, cycle, and directory."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    volume = subparsers.add_parser(
        "volume", help="compute AMR-aware disk profiles and integrated budgets")
    volume.add_argument("--primitive", nargs="+", required=True,
                        help="full-volume mhd_w or mhd_w_bcc .bin files/globs")
    volume.add_argument("--conserved", nargs="+", required=True,
                        help="matching full-volume mhd_u_d .bin files/globs")
    volume.add_argument("--angular", nargs="+",
                        help="optional matching angular_momentum .bin files/globs")
    volume.add_argument("--torque", nargs="+",
                        help="optional matching torque .bin files/globs")
    volume.add_argument("--rmin", type=float, default=0.0)
    volume.add_argument("--rmax", type=float)
    volume.add_argument("--dr", type=float, default=5.0)
    volume.add_argument("--rho-min", type=float, default=0.0,
                        help="exclude atmosphere below this primitive density")
    volume.add_argument("--radial-coordinate", choices=("cylindrical", "spherical"),
                        default="cylindrical")
    volume.add_argument("--output", default="cbd_diagnostics")
    volume.set_defaults(func=run_volume)

    history = subparsers.add_parser(
        "history", help="extract radial and H1/H2 flux time series")
    history.add_argument("history", nargs="+", help="Athena .hst files/globs")
    history.add_argument("--raw-only-ok", action="store_true",
                         help="allow histories without dynbbh surface groups")
    history.add_argument("--output", default="cbd_diagnostics")
    history.set_defaults(func=run_history)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (DiagnosticsError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
