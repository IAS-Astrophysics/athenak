import argparse
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def find_athenak_root():
    candidates = [Path(__file__).resolve().parent.parent, Path.cwd(), *Path.cwd().parents]
    for base in candidates:
        direct = base / "src" / "pgen" / "dynbbh.cpp"
        nested = base / "athenak" / "src" / "pgen" / "dynbbh.cpp"
        if direct.exists():
            return base
        if nested.exists():
            return base / "athenak"
    return Path(__file__).resolve().parent.parent


ATHENAK_ROOT = find_athenak_root()
DYNBBH_CPP = ATHENAK_ROOT / "src" / "pgen" / "dynbbh.cpp"
METRIC_KERNEL_SOURCE = None
REQUIRED_H5_DATASETS = ("uov", "B", "x1v", "x2v", "x3v", "x1f", "x2f", "x3f")
SKIPPED_H5_PREFIX = "Skipped HDF5 file"


def format_h5_skip_message(fname, reason):
    return f"{SKIPPED_H5_PREFIX}: {fname}: {reason}"


def is_h5_skip_message(message):
    return str(message).startswith(f"{SKIPPED_H5_PREFIX}:")


def write_skip_log(log_path, messages):
    if not messages:
        return
    text = "\n".join(str(message) for message in messages) + "\n"
    Path(log_path).write_text(text)


def parse_bool(value, default=False):
    if value is None:
        return default
    text = str(value).strip().strip('"').strip("'").lower()
    if text in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if text in {"false", "f", "no", "n", "0", "off"}:
        return False
    return default


def get_rot_matrix(tilt_angle_y=0.0, tilt_angle_x=0.0):
    cx, sx = np.cos(tilt_angle_x), np.sin(tilt_angle_x)
    cy, sy = np.cos(tilt_angle_y), np.sin(tilt_angle_y)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ])
    ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ])
    return ry @ rx


def disk_axis_from_tilt_angle(tilt_angle_deg):
    psi = np.deg2rad(tilt_angle_deg)
    return np.array([np.sin(psi), 0.0, np.cos(psi)], dtype=np.float64)


def normalize_vector(vec, fallback=None):
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    out = np.divide(arr, norm, out=np.zeros_like(arr), where=norm > 0.0)
    bad = (~np.isfinite(out).all(axis=-1)) | (norm[..., 0] <= 0.0)
    if np.any(bad):
        if fallback is None:
            fallback_arr = np.zeros_like(arr)
            fallback_arr[..., 2] = 1.0
        else:
            fallback_arr = np.broadcast_to(np.asarray(fallback, dtype=np.float64), arr.shape)
        out = np.where(bad[..., None], fallback_arr, out)
    return out


def rotation_matrices_from_axes(axes):
    z_axis = normalize_vector(axes)
    ref = np.zeros_like(z_axis)
    ref[..., 1] = 1.0
    x_axis = np.cross(ref, z_axis)
    x_norm = np.linalg.norm(x_axis, axis=-1)

    fallback_ref = np.zeros_like(z_axis)
    fallback_ref[..., 0] = 1.0
    fallback_x = np.cross(fallback_ref, z_axis)
    x_axis = np.where(np.expand_dims(x_norm <= 1.0e-12, axis=-1), fallback_x, x_axis)
    x_axis = normalize_vector(x_axis)
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    return np.stack((x_axis, y_axis, z_axis), axis=-2)


def rotation_matrix_from_axis(axis):
    return rotation_matrices_from_axes(np.asarray(axis, dtype=np.float64))[()]


def euler_zyx_from_rotation_matrix(rot):
    rot = np.asarray(rot, dtype=np.float64)
    pitch = np.arcsin(np.clip(-rot[..., 2, 0], -1.0, 1.0))
    cp = np.cos(pitch)
    regular = np.abs(cp) > 1.0e-12

    yaw = np.empty_like(pitch)
    roll = np.empty_like(pitch)
    yaw_regular = np.arctan2(rot[..., 1, 0], rot[..., 0, 0])
    roll_regular = np.arctan2(rot[..., 2, 1], rot[..., 2, 2])
    yaw_gimbal = np.arctan2(-rot[..., 0, 1], rot[..., 1, 1])

    yaw[...] = np.where(regular, yaw_regular, yaw_gimbal)
    roll[...] = np.where(regular, roll_regular, 0.0)
    return np.stack((yaw, pitch, roll), axis=-1)


def tilt_is_dynamical(tilt_angle_deg, margin_deg=1.0):
    angle = abs(float(tilt_angle_deg)) % 360.0
    dist_0 = min(angle, 360.0 - angle)
    dist_180 = abs(angle - 180.0)
    return min(dist_0, dist_180) > margin_deg


def parse_athena_parfile(parfile_path):
    parfile_path = Path(parfile_path)
    values = {}
    raw_values = {}
    for raw_line in Path(parfile_path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("<") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        raw_values[key] = value.strip().strip('"').strip("'")
        try:
            values[key] = float(value)
        except ValueError:
            continue

    use_traj_table = parse_bool(raw_values.get("use_traj_table"), False)
    traj_file = raw_values.get("traj_file", "")
    traj_path = None
    traj_table = None
    if use_traj_table:
        if not traj_file:
            raise ValueError("problem/use_traj_table=true requires problem/traj_file")
        traj_path = Path(traj_file).expanduser()
        if not traj_path.is_absolute():
            traj_path = parfile_path.parent / traj_path
        traj_table = load_trajectory_table(traj_path)

    params = {
        "sep": values.get("sep", 25.0),
        "q": values.get("q", 1.0),
        "gamma_gas": values.get("gamma", 5.0 / 3.0),
        "a1": values.get("a1", 0.0),
        "a2": values.get("a2", 0.0),
        "th_a1": np.deg2rad(values.get("th_a1", 0.0)),
        "th_a2": np.deg2rad(values.get("th_a2", 0.0)),
        "ph_a1": np.deg2rad(values.get("ph_a1", 0.0)),
        "ph_a2": np.deg2rad(values.get("ph_a2", 0.0)),
        "a1_buffer": values.get("a1_buffer", 0.01),
        "a2_buffer": values.get("a2_buffer", 0.01),
        "cutoff_floor": values.get("cutoff_floor", 1.0e-4),
        "tilt_angle_deg": values.get("tilt_angle", 0.0),
        "use_traj_table": use_traj_table,
        "traj_file": str(traj_path) if traj_path is not None else "",
        "traj_table": traj_table,
    }
    params["om"] = params["sep"] ** (-1.5)
    return params


TRAJ_COLUMNS = (
    "time",
    "m1_t",
    "m2_t",
    "xi1x",
    "xi1y",
    "xi1z",
    "xi2x",
    "xi2y",
    "xi2z",
    "a1x",
    "a1y",
    "a1z",
    "a2x",
    "a2y",
    "a2z",
    "v1x",
    "v1y",
    "v1z",
    "v2x",
    "v2y",
    "v2z",
)


def load_trajectory_table(path):
    rows = []
    with Path(path).open() as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < len(TRAJ_COLUMNS):
                raise ValueError(f"bad trajectory row in {path} line {lineno}")
            try:
                rows.append([float(cols[i]) for i in range(len(TRAJ_COLUMNS))])
            except ValueError as exc:
                raise ValueError(f"bad trajectory value in {path} line {lineno}") from exc

    if len(rows) < 2:
        raise ValueError(f"trajectory file has fewer than 2 rows: {path}")

    arr = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(arr[:, 0]) <= 0.0):
        raise ValueError(f"trajectory times must increase: {path}")
    if np.any(arr[:, 1] <= 0.0) or np.any(arr[:, 2] <= 0.0):
        raise ValueError(f"trajectory masses must be positive: {path}")
    spin1_sq = np.sum(arr[:, 9:12] ** 2, axis=1)
    spin2_sq = np.sum(arr[:, 12:15] ** 2, axis=1)
    if np.any(spin1_sq > 1.0 + 1.0e-12) or np.any(spin2_sq > 1.0 + 1.0e-12):
        raise ValueError(f"invalid spin in trajectory file: {path}")
    return {name: arr[:, i] for i, name in enumerate(TRAJ_COLUMNS)}


def interpolate_trajectory_table(time, table):
    times = table["time"]
    trange = times[-1] - times[0]
    tol = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(times[0]), abs(times[-1]), trange)
    if time < times[0] - tol or time > times[-1] + tol:
        raise ValueError(
            f"requested time {time} is outside trajectory-table range "
            f"[{times[0]}, {times[-1]}]"
        )

    time = float(np.clip(time, times[0], times[-1]))
    i1 = int(np.searchsorted(times, time, side="right"))
    if i1 == 0:
        i0, i1 = 0, 1
    elif i1 >= len(times):
        i0, i1 = len(times) - 2, len(times) - 1
    else:
        i0 = i1 - 1

    dt = times[i1] - times[i0]
    w = (time - times[i0]) / dt

    def hermite(pname, vname):
        p0, p1 = table[pname][i0], table[pname][i1]
        v0, v1 = table[vname][i0], table[vname][i1]
        w2, w3 = w * w, w * w * w
        pos = (
            (2.0 * w3 - 3.0 * w2 + 1.0) * p0
            + (w3 - 2.0 * w2 + w) * dt * v0
            + (-2.0 * w3 + 3.0 * w2) * p1
            + (w3 - w2) * dt * v1
        )
        vel = (
            (6.0 * w2 - 6.0 * w) * p0
            + (3.0 * w2 - 4.0 * w + 1.0) * dt * v0
            + (-6.0 * w2 + 6.0 * w) * p1
            + (3.0 * w2 - 2.0 * w) * dt * v1
        ) / dt
        return pos, vel

    def linear(name):
        return (1.0 - w) * table[name][i0] + w * table[name][i1]

    x1, vx1 = hermite("xi1x", "v1x")
    y1, vy1 = hermite("xi1y", "v1y")
    z1, vz1 = hermite("xi1z", "v1z")
    x2, vx2 = hermite("xi2x", "v2x")
    y2, vy2 = hermite("xi2y", "v2y")
    z2, vz2 = hermite("xi2z", "v2z")

    return {
        "xi1x": x1,
        "xi1y": y1,
        "xi1z": z1,
        "xi2x": x2,
        "xi2y": y2,
        "xi2z": z2,
        "v1x": vx1,
        "v1y": vy1,
        "v1z": vz1,
        "v2x": vx2,
        "v2y": vy2,
        "v2z": vz2,
        "a1x": linear("a1x"),
        "a1y": linear("a1y"),
        "a1z": linear("a1z"),
        "a2x": linear("a2x"),
        "a2y": linear("a2y"),
        "a2z": linear("a2z"),
        "m1_t": linear("m1_t"),
        "m2_t": linear("m2_t"),
    }


def find_traj_t(time, params):
    if params.get("use_traj_table", False):
        return interpolate_trajectory_table(time, params["traj_table"])

    r_bh1_0 = params["q"] / (1.0 + params["q"]) * params["sep"]
    r_bh2_0 = -params["sep"] / (1.0 + params["q"])
    om_t = params["om"] * time

    return {
        "xi1x": r_bh1_0 * np.cos(om_t),
        "xi1y": r_bh1_0 * np.sin(om_t),
        "xi1z": 0.0,
        "xi2x": r_bh2_0 * np.cos(om_t),
        "xi2y": r_bh2_0 * np.sin(om_t),
        "xi2z": 0.0,
        "v1x": -r_bh1_0 * params["om"] * np.sin(om_t),
        "v1y": r_bh1_0 * params["om"] * np.cos(om_t),
        "v1z": 0.0,
        "v2x": -r_bh2_0 * params["om"] * np.sin(om_t),
        "v2y": r_bh2_0 * params["om"] * np.cos(om_t),
        "v2z": 0.0,
        "a1x": params["a1"] * np.sin(params["th_a1"]) * np.cos(params["ph_a1"]),
        "a1y": params["a1"] * np.sin(params["th_a1"]) * np.sin(params["ph_a1"]),
        "a1z": params["a1"] * np.cos(params["th_a1"]),
        "a2x": params["a2"] * np.sin(params["th_a2"]) * np.cos(params["ph_a2"]),
        "a2y": params["a2"] * np.sin(params["th_a2"]) * np.sin(params["ph_a2"]),
        "a2z": params["a2"] * np.cos(params["th_a2"]),
        "m1_t": 1.0 / (params["q"] + 1.0),
        "m2_t": 1.0 - 1.0 / (params["q"] + 1.0),
    }


def boost_gamma_minus_one_over_v2(v2, gamma):
    v2 = np.asarray(v2, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    small = v2 < 1.0e-12
    series = 0.5 + 0.375 * v2 + 0.3125 * v2 * v2
    regular = (gamma - 1.0) / np.where(small, 1.0, v2)
    return np.where(small, series, regular)


def build_boost_jacobian(vx, vy, vz, shape):
    v2 = vx * vx + vy * vy + vz * vz
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1.0e-300))
    q = boost_gamma_minus_one_over_v2(v2, gamma)

    jac = np.zeros(shape + (4, 4), dtype=np.float64)
    jac[..., 0, 0] = gamma
    jac[..., 0, 1] = -gamma * vx
    jac[..., 0, 2] = -gamma * vy
    jac[..., 0, 3] = -gamma * vz
    jac[..., 1, 0] = jac[..., 0, 1]
    jac[..., 2, 0] = jac[..., 0, 2]
    jac[..., 3, 0] = jac[..., 0, 3]
    jac[..., 1, 1] = 1.0 + q * vx * vx
    jac[..., 1, 2] = q * vx * vy
    jac[..., 1, 3] = q * vx * vz
    jac[..., 2, 1] = jac[..., 1, 2]
    jac[..., 2, 2] = 1.0 + q * vy * vy
    jac[..., 2, 3] = q * vy * vz
    jac[..., 3, 1] = jac[..., 1, 3]
    jac[..., 3, 2] = jac[..., 2, 3]
    jac[..., 3, 3] = 1.0 + q * vz * vz
    return jac


def boosted_spatial_coordinates(x, y, z, x0, y0, z0, vx, vy, vz):
    dx = x - x0
    dy = y - y0
    dz = z - z0
    v2 = vx * vx + vy * vy + vz * vz
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1.0e-300))
    q = boost_gamma_minus_one_over_v2(v2, gamma)
    vd = vx * dx + vy * dy + vz * dz
    return dx + q * vx * vd, dy + q * vy * vd, dz + q * vz * vd


def kerr_schild_perturbation(x, y, z, ax, ay, az, mass):
    shape = x.shape
    rt2 = np.sqrt(2.0)
    irt2 = 1.0 / rt2
    a2 = ax * ax + ay * ay + az * az
    x2 = x * x + y * y + z * z
    ad = ax * x + ay * y + az * z
    term = x2 - a2
    rho2 = term + np.sqrt(np.maximum(4.0 * ad * ad + term * term, 0.0))
    rho2_safe = np.maximum(rho2, 1.0e-300)
    rho = np.sqrt(rho2_safe)
    denom = np.maximum(ad * ad + 0.25 * rho2_safe * rho2_safe, 1.0e-300)
    fac = irt2 * rho2_safe * rho * mass / denom
    den = np.maximum(a2 + 0.5 * rho2_safe, 1.0e-300)
    iden = 1.0 / den

    ell = np.empty(shape + (3,), dtype=np.float64)
    ell[..., 0] = y * az - z * ay + rt2 * ad * ax / rho + rho * x * irt2
    ell[..., 1] = -x * az + z * ax + rt2 * ad * ay / rho + rho * y * irt2
    ell[..., 2] = x * ay - y * ax + rt2 * ad * az / rho + rho * z * irt2

    ks = np.zeros(shape + (4, 4), dtype=np.float64)
    ks[..., 0, 0] = fac
    ks[..., 0, 1:] = fac[..., None] * ell * iden[..., None]
    ks[..., 1:, 0] = ks[..., 0, 1:]
    ks[..., 1:, 1:] = (
        fac[..., None, None]
        * ell[..., :, None]
        * ell[..., None, :]
        * iden[..., None, None]
        * iden[..., None, None]
    )
    return ks


def superposed_bbh_metric(x, y, z, time, params, kernel_source):
    x, y, z = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    shape = x.shape

    loc = {"x": x, "y": y, "z": z}
    loc.update(find_traj_t(time, params))

    x1bh1, x2bh1, x3bh1 = boosted_spatial_coordinates(
        x, y, z, loc["xi1x"], loc["xi1y"], loc["xi1z"], loc["v1x"], loc["v1y"], loc["v1z"]
    )
    x1bh2, x2bh2, x3bh2 = boosted_spatial_coordinates(
        x, y, z, loc["xi2x"], loc["xi2y"], loc["xi2z"], loc["v2x"], loc["v2y"], loc["v2z"]
    )

    m1 = loc["m1_t"]
    m2 = loc["m2_t"]
    a1x = loc["a1x"] * m1
    a1y = loc["a1y"] * m1
    a1z = loc["a1z"] * m1
    a2x = loc["a2x"] * m2
    a2y = loc["a2y"] * m2
    a2z = loc["a2z"] * m2
    a1 = np.sqrt(a1x * a1x + a1y * a1y + a1z * a1z + 1.0e-40)
    a2 = np.sqrt(a2x * a2x + a2y * a2y + a2z * a2z + 1.0e-40)

    rbh1 = np.sqrt(x1bh1 * x1bh1 + x2bh1 * x2bh1 + x3bh1 * x3bh1)
    rbh2 = np.sqrt(x1bh2 * x1bh2 + x2bh2 * x2bh2 + x3bh2 * x3bh2)
    rbh1_cutoff = np.abs(a1) * (1.0 + params["a1_buffer"]) + params["cutoff_floor"]
    rbh2_cutoff = np.abs(a2) * (1.0 + params["a2_buffer"]) + params["cutoff_floor"]
    x3bh1 = np.where(rbh1 < rbh1_cutoff, np.where(x3bh1 > 0.0, rbh1_cutoff, -rbh1_cutoff), x3bh1)
    x3bh2 = np.where(rbh2 < rbh2_cutoff, np.where(x3bh2 > 0.0, rbh2_cutoff, -rbh2_cutoff), x3bh2)

    ks1 = kerr_schild_perturbation(x1bh1, x2bh1, x3bh1, a1x, a1y, a1z, m1)
    ks2 = kerr_schild_perturbation(x1bh2, x2bh2, x3bh2, a2x, a2y, a2z, m2)
    j1 = build_boost_jacobian(loc["v1x"], loc["v1y"], loc["v1z"], shape)
    j2 = build_boost_jacobian(loc["v2x"], loc["v2y"], loc["v2z"], shape)

    gcov = np.zeros(shape + (4, 4), dtype=np.float64)
    gcov[..., 0, 0] = -1.0
    gcov[..., 1, 1] = 1.0
    gcov[..., 2, 2] = 1.0
    gcov[..., 3, 3] = 1.0

    gcov += np.einsum("...mi,...mn,...nj->...ij", j1, ks1, j1)
    gcov += np.einsum("...mi,...mn,...nj->...ij", j2, ks2, j2)
    return gcov


def adm_from_gcov(gcov):
    gamma_dd = gcov[..., 1:, 1:]
    beta_d = gcov[..., 0, 1:]
    gamma_uu = np.linalg.inv(gamma_dd)
    beta_u = np.einsum("...ij,...j->...i", gamma_uu, beta_d)
    alpha_sq = np.einsum("...i,...i->...", beta_d, beta_u) - gcov[..., 0, 0]
    alpha = np.sqrt(np.maximum(alpha_sq, 1.0e-300))
    return gamma_dd, gamma_uu, alpha, beta_u


def gamma_law_specific_enthalpy(rho, press, gamma_gas):
    gm1 = gamma_gas - 1.0
    return 1.0 + (gamma_gas * press) / np.maximum(rho * gm1, 1.0e-300)


def rotate_spatial_tensor(r_mat, tensor):
    return np.einsum("...ia,...jb,...ab->...ij", r_mat, r_mat, tensor)


def rotate_spatial_vector(r_mat, vector):
    return np.einsum("...ia,...a->...i", r_mat, vector)


def aligned_spherical_transforms(pts_aligned):
    x = pts_aligned[..., 0]
    y = pts_aligned[..., 1]
    z = pts_aligned[..., 2]
    r = np.maximum(np.linalg.norm(pts_aligned, axis=-1), 1.0e-300)
    rho_cyl_sq = np.maximum(x * x + y * y, 1.0e-300)
    rho_cyl = np.sqrt(rho_cyl_sq)
    cos_phi = np.where(rho_cyl > 1.0e-150, x / rho_cyl, 1.0)
    sin_phi = np.where(rho_cyl > 1.0e-150, y / rho_cyl, 0.0)

    jac = np.zeros(pts_aligned.shape[:-1] + (3, 3), dtype=np.float64)
    jac[..., 0, 0] = x / r
    jac[..., 1, 0] = y / r
    jac[..., 2, 0] = z / r
    jac[..., 0, 1] = z * cos_phi
    jac[..., 1, 1] = z * sin_phi
    jac[..., 2, 1] = -rho_cyl
    jac[..., 0, 2] = -y
    jac[..., 1, 2] = x

    inv_jac = np.zeros_like(jac)
    inv_jac[..., 0, 0] = x / r
    inv_jac[..., 0, 1] = y / r
    inv_jac[..., 0, 2] = z / r
    inv_jac[..., 1, 0] = z * cos_phi / np.maximum(r * r, 1.0e-300)
    inv_jac[..., 1, 1] = z * sin_phi / np.maximum(r * r, 1.0e-300)
    inv_jac[..., 1, 2] = -rho_cyl / np.maximum(r * r, 1.0e-300)
    inv_jac[..., 2, 0] = -y / rho_cyl_sq
    inv_jac[..., 2, 1] = x / rho_cyl_sq

    return jac, inv_jac


def transform_metric_to_spherical(gcov_cart, jac):
    gcov_sph = np.zeros_like(gcov_cart)
    gcov_sph[..., 0, 0] = gcov_cart[..., 0, 0]
    gcov_sph[..., 0, 1:] = np.einsum("...i,...ia->...a", gcov_cart[..., 0, 1:], jac)
    gcov_sph[..., 1:, 0] = gcov_sph[..., 0, 1:]
    gcov_sph[..., 1:, 1:] = np.einsum("...ia,...ij,...jb->...ab", jac, gcov_cart[..., 1:, 1:], jac)
    return gcov_sph


def transform_contravariant_to_spherical(vec_con_cart, inv_jac):
    vec_con_sph = np.zeros_like(vec_con_cart)
    vec_con_sph[..., 0] = vec_con_cart[..., 0]
    vec_con_sph[..., 1:] = np.einsum("...ai,...i->...a", inv_jac, vec_con_cart[..., 1:])
    return vec_con_sph


def metric_dot(gcov, vec_a, vec_b):
    return np.einsum("...ab,...a,...b->...", gcov, vec_a, vec_b)


def covector_dot(gcon, cov_a, cov_b):
    return np.einsum("...ab,...a,...b->...", gcon, cov_a, cov_b)


def normalize_timelike(vec_con, gcov):
    norm = metric_dot(gcov, vec_con, vec_con)
    scale = np.sqrt(np.maximum(-norm, 1.0e-300))
    return vec_con / scale[..., None]


def orthonormalize_seed(seed_con, gcov, basis_vectors):
    vec = np.array(seed_con, copy=True)
    for base in basis_vectors:
        base_norm = metric_dot(gcov, base, base)
        coeff = metric_dot(gcov, vec, base)
        denom = np.where(np.abs(base_norm) > 1.0e-300, base_norm, np.nan)
        vec = vec - (coeff / denom)[..., None] * base
    norm = metric_dot(gcov, vec, vec)
    scale = np.sqrt(np.maximum(norm, 1.0e-300))
    return vec / scale[..., None]


def normalize_timelike_covector(cov, gcon):
    norm = covector_dot(gcon, cov, cov)
    scale = np.sqrt(np.maximum(-norm, 1.0e-300))
    return cov / scale[..., None]


def orthonormalize_covector(seed_cov, gcon, basis_covectors):
    cov = np.array(seed_cov, copy=True)
    for base in basis_covectors:
        base_norm = covector_dot(gcon, base, base)
        coeff = covector_dot(gcon, cov, base)
        denom = np.where(np.abs(base_norm) > 1.0e-300, base_norm, np.nan)
        cov = cov - (coeff / denom)[..., None] * base
    norm = covector_dot(gcon, cov, cov)
    scale = np.sqrt(np.maximum(norm, 1.0e-300))
    return cov / scale[..., None]


def build_shell_geometry(pts_aligned, pts_sim, coord_dA, r_mat, time, params, kernel_source):
    gcov_sim = superposed_bbh_metric(
        pts_sim[..., 0], pts_sim[..., 1], pts_sim[..., 2], time, params, kernel_source
    )
    gcov = np.zeros_like(gcov_sim)
    gcov[..., 0, 0] = gcov_sim[..., 0, 0]
    gcov[..., 0, 1:] = rotate_spatial_vector(r_mat, gcov_sim[..., 0, 1:])
    gcov[..., 1:, 0] = gcov[..., 0, 1:]
    gcov[..., 1:, 1:] = rotate_spatial_tensor(r_mat, gcov_sim[..., 1:, 1:])
    gcon = np.linalg.inv(gcov)

    gamma_dd, gamma_uu, alpha, beta_u = adm_from_gcov(gcov)
    coord_normal_cov = pts_aligned / np.maximum(
        np.linalg.norm(pts_aligned, axis=-1, keepdims=True), 1.0e-300
    )
    d_sigma_cov = coord_normal_cov * coord_dA[..., None]
    proper_dA = np.sqrt(
        np.maximum(np.einsum("...i,...ij,...j->...", d_sigma_cov, gamma_uu, d_sigma_cov), 0.0)
    )
    unit_normal_cov = np.divide(
        d_sigma_cov,
        proper_dA[..., None],
        out=np.zeros_like(d_sigma_cov),
        where=proper_dA[..., None] > 0.0,
    )

    return {
        "gcov": gcov,
        "gcon": gcon,
        "gamma_dd": gamma_dd,
        "gamma_uu": gamma_uu,
        "alpha": alpha,
        "beta_u": beta_u,
        "coord_normal_cov": coord_normal_cov,
        "d_sigma_cov": d_sigma_cov,
        "unit_normal_cov": unit_normal_cov,
        "proper_dA": proper_dA,
    }


def valencia_shell_diagnostics(
    rho,
    press,
    prim_wv_sim,
    b_tilde_sim,
    pts_aligned,
    midplane_distance,
    r_mat,
    shell_geom,
    params,
):
    gcov = shell_geom["gcov"]
    gcon = shell_geom["gcon"]
    gamma_dd = shell_geom["gamma_dd"]
    gamma_uu = shell_geom["gamma_uu"]
    alpha = shell_geom["alpha"]
    beta_u = shell_geom["beta_u"]
    d_sigma_cov = shell_geom["d_sigma_cov"]
    unit_normal_cov = shell_geom["unit_normal_cov"]
    proper_dA = shell_geom["proper_dA"]

    prim_wv = rotate_spatial_vector(r_mat, prim_wv_sim)
    b_tilde = rotate_spatial_vector(r_mat, b_tilde_sim)

    prim_lower = np.einsum("...ij,...j->...i", gamma_dd, prim_wv)
    u_sq = np.einsum("...i,...i->...", prim_wv, prim_lower)
    w_lorentz = np.sqrt(1.0 + u_sq)
    v_u = prim_wv / w_lorentz[..., None]
    v_d = np.einsum("...ij,...j->...i", gamma_dd, v_u)

    u_con = np.zeros(gcov.shape[:-1], dtype=np.float64)
    u_con[..., 0] = w_lorentz / alpha
    u_con[..., 1:] = w_lorentz[..., None] * (v_u - beta_u / alpha[..., None])
    u_cov = np.einsum("...ab,...b->...a", gcov, u_con)

    detg = np.linalg.det(gamma_dd)
    sqrt_gamma = np.sqrt(np.maximum(detg, 1.0e-300))
    b_u = b_tilde / sqrt_gamma[..., None]
    bv = np.einsum("...i,...i->...", b_u, v_d)

    b_con = np.zeros_like(u_con)
    b_con[..., 0] = w_lorentz * bv / alpha
    b_con[..., 1:] = (
        b_u / w_lorentz[..., None]
        + w_lorentz[..., None] * bv[..., None] * (v_u - beta_u / alpha[..., None])
    )
    b_cov = np.einsum("...ab,...b->...a", gcov, b_con)

    # Match flux_generalized.cpp: \dot M = - \int rho u^i dS_i, where dS_i is
    # the covariant shell element. The unit-normal projection is kept for local
    # shell diagnostics, while mdot_surface carries the fully integrated
    # covector element.
    ur_coord = np.einsum("...i,...i->...", u_con[..., 1:], unit_normal_cov)
    vr_euler = np.einsum("...i,...i->...", v_u, unit_normal_cov)
    mdot_density = -rho * ur_coord
    mdot_surface = -rho * np.einsum("...i,...i->...", u_con[..., 1:], d_sigma_cov)

    br_coord = np.einsum("...i,...i->...", b_con[..., 1:], unit_normal_cov)
    x_aligned = pts_aligned[..., 0]
    y_aligned = pts_aligned[..., 1]
    z_aligned = pts_aligned[..., 2]
    u_phi = -y_aligned * u_cov[..., 1] + x_aligned * u_cov[..., 2]
    b_phi = -y_aligned * b_cov[..., 1] + x_aligned * b_cov[..., 2]
    radius = np.linalg.norm(pts_aligned, axis=-1)

    # Build a spatial orthonormal basis from the shell radial direction and the
    # azimuthal tangent, using the spatial metric gamma_ij for normalization and
    # orthogonalization. This is the natural GR-consistent frame for local
    # r-phi stresses on the analysis shell.
    radial_seed_u = pts_aligned / np.maximum(np.linalg.norm(pts_aligned, axis=-1, keepdims=True), 1.0e-300)
    radial_norm = np.sqrt(
        np.maximum(
            np.einsum("...i,...ij,...j->...", radial_seed_u, gamma_dd, radial_seed_u),
            1.0e-300,
        )
    )
    ehat_r_u = radial_seed_u / radial_norm[..., None]

    phi_seed_u = np.stack((-y_aligned, x_aligned, np.zeros_like(x_aligned)), axis=-1)
    phi_proj_r = np.einsum("...i,...ij,...j->...", phi_seed_u, gamma_dd, ehat_r_u)
    phi_perp_u = phi_seed_u - phi_proj_r[..., None] * ehat_r_u
    phi_norm = np.sqrt(
        np.maximum(
            np.einsum("...i,...ij,...j->...", phi_perp_u, gamma_dd, phi_perp_u),
            1.0e-300,
        )
    )
    ehat_phi_u = phi_perp_u / phi_norm[..., None]

    rho_h = rho * gamma_law_specific_enthalpy(rho, press, params["gamma_gas"])
    b_sq = np.einsum("...a,...a->...", b_con, b_cov)
    total_pressure = press + 0.5 * b_sq

    b0_euler = w_lorentz * bv
    b_d_euler = (
        np.einsum("...ij,...j->...i", gamma_dd, b_u) / w_lorentz[..., None]
        + b0_euler[..., None] * v_d
    )
    w2 = w_lorentz * w_lorentz
    fluid_momentum_d = rho_h[..., None] * w2[..., None] * v_d
    em_momentum_d = (
        b_sq[..., None] * w2[..., None] * v_d - b0_euler[..., None] * b_d_euler
    )
    fluid_stress_dd = (
        rho_h[..., None, None]
        * w2[..., None, None]
        * v_d[..., :, None]
        * v_d[..., None, :]
        + press[..., None, None] * gamma_dd
    )
    em_stress_dd = (
        b_sq[..., None, None]
        * w2[..., None, None]
        * v_d[..., :, None]
        * v_d[..., None, :]
        + 0.5 * b_sq[..., None, None] * gamma_dd
        - b_d_euler[..., :, None] * b_d_euler[..., None, :]
    )
    fluid_stress_ud = np.einsum("...ik,...kj->...ij", gamma_uu, fluid_stress_dd)
    em_stress_ud = np.einsum("...ik,...kj->...ij", gamma_uu, em_stress_dd)
    fluid_flux_ud = (
        alpha[..., None, None] * fluid_stress_ud
        - beta_u[..., :, None] * fluid_momentum_d[..., None, :]
    )
    em_flux_ud = (
        alpha[..., None, None] * em_stress_ud
        - beta_u[..., :, None] * em_momentum_d[..., None, :]
    )
    pdot_fluid_density = np.einsum("...ij,...i->...j", fluid_flux_ud, unit_normal_cov)
    pdot_em_density = np.einsum("...ij,...i->...j", em_flux_ud, unit_normal_cov)
    pdot_fluid_surface = np.einsum("...ij,...i->...j", fluid_flux_ud, d_sigma_cov)
    pdot_em_surface = np.einsum("...ij,...i->...j", em_flux_ud, d_sigma_cov)

    def angular_momentum_flux(momentum_flux_d):
        return np.stack(
            (
                y_aligned * momentum_flux_d[..., 2] - z_aligned * momentum_flux_d[..., 1],
                z_aligned * momentum_flux_d[..., 0] - x_aligned * momentum_flux_d[..., 2],
                x_aligned * momentum_flux_d[..., 1] - y_aligned * momentum_flux_d[..., 0],
            ),
            axis=-1,
        )

    ldot_fluid_density = angular_momentum_flux(pdot_fluid_density)
    ldot_em_density = angular_momentum_flux(pdot_em_density)
    ldot_fluid_surface = angular_momentum_flux(pdot_fluid_surface)
    ldot_em_surface = angular_momentum_flux(pdot_em_surface)

    matter_stress = rho_h * ur_coord * u_phi
    maxwell_stress = -br_coord * b_phi
    em_exact_stress = b_sq * ur_coord * u_phi - br_coord * b_phi

    uhat_r = np.einsum("...i,...i->...", ehat_r_u, u_cov[..., 1:])
    uhat_phi = np.einsum("...i,...i->...", ehat_phi_u, u_cov[..., 1:])
    bhat_r = np.einsum("...i,...i->...", ehat_r_u, b_cov[..., 1:])
    bhat_phi = np.einsum("...i,...i->...", ehat_phi_u, b_cov[..., 1:])

    ortho_matter_stress = rho_h * uhat_r * uhat_phi
    ortho_maxwell_stress = -bhat_r * bhat_phi
    ortho_em_exact_stress = b_sq * uhat_r * uhat_phi - bhat_r * bhat_phi

    # Penna et al. (2012) alpha procedure: construct the orthonormal rest
    # frame of the instantaneous azimuthal mean flow in spherical coordinates,
    # compute local stresses there, azimuthally average, and defer the final
    # vertical/time averaging to the archive stitching step.
    jac_sph, inv_jac_sph = aligned_spherical_transforms(pts_aligned)
    gcov_sph = transform_metric_to_spherical(gcov, jac_sph)
    gcon_sph = np.linalg.inv(gcov_sph)
    u_con_sph = transform_contravariant_to_spherical(u_con, inv_jac_sph)
    b_con_sph = transform_contravariant_to_spherical(b_con, inv_jac_sph)
    u_cov_sph = np.einsum("...ab,...b->...a", gcov_sph, u_con_sph)
    b_cov_sph = np.einsum("...ab,...b->...a", gcov_sph, b_con_sph)

    sqrt_neg_g = np.sqrt(np.maximum(-np.linalg.det(gcov_sph), 1.0e-300))
    sqrt_g_thetatheta = np.sqrt(np.maximum(gcov_sph[..., 2, 2], 1.0e-300))
    mean_flow_con_sph = np.mean(u_con_sph, axis=2, keepdims=True)
    mean_flow_con_sph = np.broadcast_to(mean_flow_con_sph, u_con_sph.shape).copy()
    mean_flow_cov_sph = np.einsum("...ab,...b->...a", gcov_sph, mean_flow_con_sph)

    omega_t = normalize_timelike_covector(mean_flow_cov_sph, gcon_sph)
    seed_r_cov = np.zeros_like(u_con_sph)
    seed_r_cov[..., 1] = 1.0
    omega_r = orthonormalize_covector(seed_r_cov, gcon_sph, [omega_t])
    seed_theta_cov = np.zeros_like(u_con_sph)
    seed_theta_cov[..., 2] = 1.0
    omega_theta = orthonormalize_covector(seed_theta_cov, gcon_sph, [omega_t, omega_r])
    seed_phi_cov = np.zeros_like(u_con_sph)
    seed_phi_cov[..., 3] = 1.0
    omega_phi = orthonormalize_covector(seed_phi_cov, gcon_sph, [omega_t, omega_r, omega_theta])

    coframe = np.stack((omega_t, omega_r, omega_theta, omega_phi), axis=-2)
    frame = np.linalg.inv(coframe)
    ehat_r_mean = frame[..., :, 1]
    ehat_theta_mean = frame[..., :, 2]
    ehat_phi_mean = frame[..., :, 3]

    t_rey_cov_penna = rho_h[..., None, None] * u_cov_sph[..., :, None] * u_cov_sph[..., None, :] + press[..., None, None] * gcov_sph

    stress_rey_penna_local = np.einsum(
        "...a,...ab,...b->...", ehat_r_mean, t_rey_cov_penna, ehat_phi_mean
    )
    stress_shear_rey_penna_local = np.einsum(
        "...a,...ab,...b->...", ehat_r_mean, t_rey_cov_penna, ehat_theta_mean
    )
    uhat_r_penna = np.einsum("...a,...a->...", ehat_r_mean, u_cov_sph)
    uhat_theta_penna = np.einsum("...a,...a->...", ehat_theta_mean, u_cov_sph)
    uhat_phi_penna = np.einsum("...a,...a->...", ehat_phi_mean, u_cov_sph)
    bhat_r_penna = np.einsum("...a,...a->...", ehat_r_mean, b_cov_sph)
    bhat_theta_penna = np.einsum("...a,...a->...", ehat_theta_mean, b_cov_sph)
    bhat_phi_penna = np.einsum("...a,...a->...", ehat_phi_mean, b_cov_sph)
    stress_max_penna_local = -bhat_r_penna * bhat_phi_penna
    stress_shear_max_penna_local = -bhat_r_penna * bhat_theta_penna
    stress_em_exact_penna_local = b_sq * uhat_r_penna * uhat_phi_penna + stress_max_penna_local
    stress_shear_em_exact_penna_local = (
        b_sq * uhat_r_penna * uhat_theta_penna + stress_shear_max_penna_local
    )
    stress_tot_penna_local = stress_rey_penna_local + stress_em_exact_penna_local
    stress_shear_tot_penna_local = stress_shear_rey_penna_local + stress_shear_em_exact_penna_local
    alpha_rey_penna_local = stress_rey_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_max_penna_local = stress_max_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_em_exact_penna_local = stress_em_exact_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_penna_local = stress_tot_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_shear_rey_penna_local = stress_shear_rey_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_shear_max_penna_local = stress_shear_max_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_shear_em_exact_penna_local = (
        stress_shear_em_exact_penna_local / np.maximum(total_pressure, 1.0e-300)
    )
    alpha_shear_penna_local = stress_shear_tot_penna_local / np.maximum(total_pressure, 1.0e-300)

    penna_scale_weight = rho * u_con_sph[..., 0] * sqrt_neg_g
    penna_scale_num = np.sum((midplane_distance ** 2) * penna_scale_weight, axis=(1, 2))
    penna_scale_den = np.sum(penna_scale_weight, axis=(1, 2))
    scale_height_penna = np.where(
        penna_scale_den > 0.0,
        np.sqrt(np.maximum(penna_scale_num / penna_scale_den, 0.0)),
        np.nan,
    )
    h_over_r_penna = scale_height_penna / np.maximum(radius[:, 0, 0], 1.0e-300)

    disk_mask = midplane_distance <= scale_height_penna[:, None, None]
    disk_area = np.sum(proper_dA * disk_mask, axis=(1, 2))

    def shell_avg(quantity):
        numer = np.sum(quantity * proper_dA * disk_mask, axis=(1, 2))
        return np.where(disk_area > 0.0, numer / disk_area, np.nan)

    penna_vert_weight_theta = np.mean(rho * u_con_sph[..., 0] * sqrt_g_thetatheta, axis=2)
    penna_ptot_theta = np.mean(total_pressure, axis=2)
    penna_rey_theta = np.mean(stress_rey_penna_local, axis=2)
    penna_max_theta = np.mean(stress_max_penna_local, axis=2)
    penna_em_exact_theta = np.mean(stress_em_exact_penna_local, axis=2)
    penna_total_theta = np.mean(stress_tot_penna_local, axis=2)
    penna_shear_rey_theta = np.mean(stress_shear_rey_penna_local, axis=2)
    penna_shear_max_theta = np.mean(stress_shear_max_penna_local, axis=2)
    penna_shear_em_exact_theta = np.mean(stress_shear_em_exact_penna_local, axis=2)
    penna_shear_total_theta = np.mean(stress_shear_tot_penna_local, axis=2)
    penna_alpha_rey_theta = np.mean(alpha_rey_penna_local, axis=2)
    penna_alpha_max_theta = np.mean(alpha_max_penna_local, axis=2)
    penna_alpha_em_exact_theta = np.mean(alpha_em_exact_penna_local, axis=2)
    penna_alpha_theta = np.mean(alpha_penna_local, axis=2)
    penna_alpha_shear_rey_theta = np.mean(alpha_shear_rey_penna_local, axis=2)
    penna_alpha_shear_max_theta = np.mean(alpha_shear_max_penna_local, axis=2)
    penna_alpha_shear_em_exact_theta = np.mean(alpha_shear_em_exact_penna_local, axis=2)
    penna_alpha_shear_theta = np.mean(alpha_shear_penna_local, axis=2)
    mean_flow_con_sph_theta = np.mean(u_con_sph, axis=2)

    stress_profiles = {
        "stress_mean_press": shell_avg(press),
        "stress_mean_ptot": shell_avg(total_pressure),
        "stress_mean_rhoh": shell_avg(rho_h),
        "stress_mean_ur": shell_avg(ur_coord),
        "stress_mean_uphi": shell_avg(u_phi),
        "stress_mean_matter": shell_avg(matter_stress),
        "stress_mean_maxwell": shell_avg(maxwell_stress),
        "stress_mean_em_exact": shell_avg(em_exact_stress),
        "stress_mean_uhat_r": shell_avg(uhat_r),
        "stress_mean_uhat_phi": shell_avg(uhat_phi),
        "stress_mean_matter_ortho": shell_avg(ortho_matter_stress),
        "stress_mean_maxwell_ortho": shell_avg(ortho_maxwell_stress),
        "stress_mean_em_exact_ortho": shell_avg(ortho_em_exact_stress),
        "stress_disk_area": disk_area,
        "scale_height_penna": scale_height_penna,
        "h_over_r_penna": h_over_r_penna,
        "penna_scale_num": penna_scale_num,
        "penna_scale_den": penna_scale_den,
        "penna_vert_weight_theta": penna_vert_weight_theta,
        "penna_ptot_theta_weighted": penna_ptot_theta * penna_vert_weight_theta,
        "penna_rey_theta_weighted": penna_rey_theta * penna_vert_weight_theta,
        "penna_max_theta_weighted": penna_max_theta * penna_vert_weight_theta,
        "penna_em_exact_theta_weighted": penna_em_exact_theta * penna_vert_weight_theta,
        "penna_total_theta_weighted": penna_total_theta * penna_vert_weight_theta,
        "penna_shear_rey_theta_weighted": penna_shear_rey_theta * penna_vert_weight_theta,
        "penna_shear_max_theta_weighted": penna_shear_max_theta * penna_vert_weight_theta,
        "penna_shear_em_exact_theta_weighted": penna_shear_em_exact_theta * penna_vert_weight_theta,
        "penna_shear_total_theta_weighted": penna_shear_total_theta * penna_vert_weight_theta,
        "penna_alpha_rey_theta_weighted": penna_alpha_rey_theta * penna_vert_weight_theta,
        "penna_alpha_max_theta_weighted": penna_alpha_max_theta * penna_vert_weight_theta,
        "penna_alpha_em_exact_theta_weighted": penna_alpha_em_exact_theta * penna_vert_weight_theta,
        "penna_alpha_theta_weighted": penna_alpha_theta * penna_vert_weight_theta,
        "penna_alpha_shear_rey_theta_weighted": penna_alpha_shear_rey_theta * penna_vert_weight_theta,
        "penna_alpha_shear_max_theta_weighted": penna_alpha_shear_max_theta * penna_vert_weight_theta,
        "penna_alpha_shear_em_exact_theta_weighted": (
            penna_alpha_shear_em_exact_theta * penna_vert_weight_theta
        ),
        "penna_alpha_shear_theta_weighted": penna_alpha_shear_theta * penna_vert_weight_theta,
        "penna_mean_flow_ut": mean_flow_con_sph_theta[..., 0],
        "penna_mean_flow_ur": mean_flow_con_sph_theta[..., 1],
        "penna_mean_flow_utheta": mean_flow_con_sph_theta[..., 2],
        "penna_mean_flow_uphi": mean_flow_con_sph_theta[..., 3],
        "ldot_fluid_density": ldot_fluid_density,
        "ldot_em_density": ldot_em_density,
        "ldot_fluid_surface": ldot_fluid_surface,
        "ldot_em_surface": ldot_em_surface,
    }

    return mdot_density, mdot_surface, ur_coord, vr_euler, stress_profiles


def get_var_indices(h5_file):
    try:
        var_names = h5_file.attrs["VariableNames"]
        var_names = [v.decode("utf-8") if isinstance(v, bytes) else v for v in var_names]
    except KeyError:
        var_names = ["dens", "mom1", "mom2", "mom3", "Etot"]

    idx_map = {name: i for i, name in enumerate(var_names)}
    dens_idx = idx_map.get("dens", 0)
    press_idx = idx_map.get("press", idx_map.get("pres", idx_map.get("pgas", 4)))
    vx_idx = idx_map.get("velx", idx_map.get("mom1", 1))
    vy_idx = idx_map.get("vely", idx_map.get("mom2", 2))
    vz_idx = idx_map.get("velz", idx_map.get("mom3", 3))
    return dens_idx, press_idx, vx_idx, vy_idx, vz_idx, "mom1" in idx_map


def get_dump_time(fname):
    with h5py.File(fname, "r") as f:
        return float(f.attrs.get("Time", 0.0))


def inspect_h5_dump(fname):
    try:
        with h5py.File(fname, "r") as f:
            missing = [key for key in REQUIRED_H5_DATASETS if key not in f]
            if missing:
                reason = "missing required dataset(s): " + ", ".join(missing)
                return None, format_h5_skip_message(fname, reason)

            _dens_i, _press_i, _vx_i, _vy_i, _vz_i, is_conserved = get_var_indices(f)
            if is_conserved:
                reason = (
                    "does not contain primitive velocity output required for GR stresses"
                )
                return None, format_h5_skip_message(fname, reason)
            return float(f.attrs.get("Time", 0.0)), None
    except (OSError, KeyError) as exc:
        return None, format_h5_skip_message(fname, exc)


def make_shell_points(radii_axis, theta_vals, phi_vals, r_mats):
    r_3d, theta_3d, phi_3d = np.meshgrid(radii_axis, theta_vals, phi_vals, indexing="ij")
    x_aligned = r_3d * np.sin(theta_3d) * np.cos(phi_3d)
    y_aligned = r_3d * np.sin(theta_3d) * np.sin(phi_3d)
    z_aligned = r_3d * np.cos(theta_3d)
    pts_aligned = np.stack((x_aligned, y_aligned, z_aligned), axis=-1)

    r_mats = np.asarray(r_mats, dtype=np.float64)
    if r_mats.ndim == 2:
        pts_sim = np.einsum("ji,...j->...i", r_mats, pts_aligned)
    else:
        pts_sim = np.einsum("...ji,...j->...i", r_mats[:, None, None, :, :], pts_aligned)

    dtheta = theta_vals[1] - theta_vals[0] if theta_vals.size > 1 else np.pi
    dphi = phi_vals[1] - phi_vals[0] if phi_vals.size > 1 else 2.0 * np.pi
    coord_dA = (r_3d ** 2) * np.sin(theta_3d) * dtheta * dphi
    return r_3d, theta_3d, phi_3d, pts_aligned, pts_sim, coord_dA


def interpolate_shell_samples(
    ds_uov,
    ds_b,
    x1v,
    x2v,
    x3v,
    x1f,
    x2f,
    x3f,
    dens_i,
    press_i,
    vx_i,
    vy_i,
    vz_i,
    pts_sim,
):
    n_bins, n_theta, n_phi = pts_sim.shape[:3]
    n_tot = n_bins * n_theta * n_phi
    x_flat = pts_sim[..., 0].ravel()
    y_flat = pts_sim[..., 1].ravel()
    z_flat = pts_sim[..., 2].ravel()

    interp_dens = np.full(n_tot, np.nan, dtype=np.float64)
    interp_press = np.full(n_tot, np.nan, dtype=np.float64)
    interp_vx = np.full(n_tot, np.nan, dtype=np.float64)
    interp_vy = np.full(n_tot, np.nan, dtype=np.float64)
    interp_vz = np.full(n_tot, np.nan, dtype=np.float64)
    interp_bx = np.zeros(n_tot, dtype=np.float64)
    interp_by = np.zeros(n_tot, dtype=np.float64)
    interp_bz = np.zeros(n_tot, dtype=np.float64)
    unassigned = np.ones(n_tot, dtype=bool)
    interp_kwargs = dict(method="linear", bounds_error=False, fill_value=None)

    for b in range(ds_uov.shape[1]):
        if not np.any(unassigned):
            break
        eps = 1.0e-8
        mask = (
            (x_flat >= x1f[b, 0] - eps)
            & (x_flat <= x1f[b, -1] + eps)
            & (y_flat >= x2f[b, 0] - eps)
            & (y_flat <= x2f[b, -1] + eps)
            & (z_flat >= x3f[b, 0] - eps)
            & (z_flat <= x3f[b, -1] + eps)
            & unassigned
        )
        if not np.any(mask):
            continue

        uov_block = ds_uov[:, b, :, :, :]
        b_block = ds_b[:, b, :, :, :]
        pts_query = np.column_stack([z_flat[mask], y_flat[mask], x_flat[mask]])
        axes = (x3v[b], x2v[b], x1v[b])
        interp_dens[mask] = RegularGridInterpolator(axes, uov_block[dens_i], **interp_kwargs)(pts_query)
        interp_press[mask] = RegularGridInterpolator(axes, uov_block[press_i], **interp_kwargs)(pts_query)
        interp_vx[mask] = RegularGridInterpolator(axes, uov_block[vx_i], **interp_kwargs)(pts_query)
        interp_vy[mask] = RegularGridInterpolator(axes, uov_block[vy_i], **interp_kwargs)(pts_query)
        interp_vz[mask] = RegularGridInterpolator(axes, uov_block[vz_i], **interp_kwargs)(pts_query)
        interp_bx[mask] = RegularGridInterpolator(axes, b_block[0], **interp_kwargs)(pts_query)
        interp_by[mask] = RegularGridInterpolator(axes, b_block[1], **interp_kwargs)(pts_query)
        interp_bz[mask] = RegularGridInterpolator(axes, b_block[2], **interp_kwargs)(pts_query)
        unassigned[mask] = False

    if np.any(unassigned):
        raise ValueError(f"left {np.count_nonzero(unassigned)} shell samples unassigned")

    rho_grid = np.maximum(interp_dens, 0.0).reshape((n_bins, n_theta, n_phi))
    press_grid = np.maximum(interp_press, 0.0).reshape((n_bins, n_theta, n_phi))
    prim_wv = np.stack(
        (
            interp_vx.reshape((n_bins, n_theta, n_phi)),
            interp_vy.reshape((n_bins, n_theta, n_phi)),
            interp_vz.reshape((n_bins, n_theta, n_phi)),
        ),
        axis=-1,
    )
    b_tilde_sim = np.stack(
        (
            interp_bx.reshape((n_bins, n_theta, n_phi)),
            interp_by.reshape((n_bins, n_theta, n_phi)),
            interp_bz.reshape((n_bins, n_theta, n_phi)),
        ),
        axis=-1,
    )
    return rho_grid, press_grid, prim_wv, b_tilde_sim


def compute_mass_shell(ds_uov, x1v, x2v, x3v, dens_i, rmin, dr, n_bins, bin_edges):
    mass_shell = np.zeros(n_bins, dtype=np.float64)
    for b in range(ds_uov.shape[1]):
        uov_block = ds_uov[:, b, :, :, :]
        x = x1v[b]
        y = x2v[b]
        z = x3v[b]
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        dz = z[1] - z[0] if len(z) > 1 else 1.0
        dV = dx * dy * dz

        use_supersampling = (dx >= dr / 5.0) and (dy >= dr / 5.0) and (dz >= dr / 5.0)
        h_r = 0.5 * np.sqrt(dx * dx + dy * dy + dz * dz)

        z_grid, y_grid, x_grid = np.meshgrid(z, y, x, indexing="ij")
        r_grid = np.sqrt(x_grid ** 2 + y_grid ** 2 + z_grid ** 2)

        dens_flat_block = uov_block[dens_i].ravel()
        x_block = x_grid.ravel()
        y_block = y_grid.ravel()
        z_block = z_grid.ravel()
        r_block = r_grid.ravel()

        idx_min = np.floor((r_block - h_r - rmin) / dr).astype(int)
        idx_max = np.floor((r_block + h_r - rmin) / dr).astype(int)
        mask_global = (idx_max >= 0) & (idx_min < n_bins)

        if not np.any(mask_global):
            continue

        inds_min = idx_min[mask_global]
        inds_max = idx_max[mask_global]
        dens_active = dens_flat_block[mask_global]

        if use_supersampling:
            mask_easy = inds_min == inds_max
        else:
            mask_easy = np.ones(inds_min.shape, dtype=bool)
            r_active = r_block[mask_global]
            inds_min = np.digitize(r_active, bin_edges) - 1

        if np.any(mask_easy):
            final_bins = inds_min[mask_easy]
            valid_easy = (final_bins >= 0) & (final_bins < n_bins)
            if np.any(valid_easy):
                bins_to_count = final_bins[valid_easy]
                weights = dens_active[mask_easy][valid_easy] * dV
                mass_shell += np.bincount(bins_to_count, weights=weights, minlength=n_bins)

        mask_split = ~mask_easy
        if use_supersampling and np.any(mask_split):
            x_split = x_block[mask_global][mask_split]
            y_split = y_block[mask_global][mask_split]
            z_split = z_block[mask_global][mask_split]
            dens_split = dens_active[mask_split]

            off_x, off_y, off_z = 0.25 * dx, 0.25 * dy, 0.25 * dz
            sub_dV = dV / 8.0
            offsets = [
                (off_x, off_y, off_z),
                (off_x, off_y, -off_z),
                (off_x, -off_y, off_z),
                (off_x, -off_y, -off_z),
                (-off_x, off_y, off_z),
                (-off_x, off_y, -off_z),
                (-off_x, -off_y, off_z),
                (-off_x, -off_y, -off_z),
            ]
            for ox, oy, oz in offsets:
                r_sub = np.sqrt((x_split + ox) ** 2 + (y_split + oy) ** 2 + (z_split + oz) ** 2)
                bins_sub = np.floor((r_sub - rmin) / dr).astype(int)
                mask_valid_sub = (bins_sub >= 0) & (bins_sub < n_bins)
                if np.any(mask_valid_sub):
                    valid_bins = bins_sub[mask_valid_sub]
                    w_sub = dens_split[mask_valid_sub] * sub_dV
                    mass_shell += np.bincount(valid_bins, weights=w_sub, minlength=n_bins)
    return mass_shell


def estimate_shell_angular_momentum_axes(rho, prim_wv_sim, pts_sim, proper_dA, fallback_axis):
    weights = rho * proper_dA
    shell_l = np.sum(np.cross(pts_sim, prim_wv_sim) * weights[..., None], axis=(1, 2))
    fallback = np.broadcast_to(normalize_vector(fallback_axis), shell_l.shape)
    norms = np.linalg.norm(shell_l, axis=1)
    valid = np.isfinite(shell_l).all(axis=1) & (norms > 1.0e-300)
    axes = np.divide(shell_l, norms[:, None], out=np.zeros_like(shell_l), where=norms[:, None] > 0.0)
    axes = np.where(valid[:, None], axes, fallback)
    flip = np.sum(axes * fallback, axis=1) < 0.0
    axes = np.where(flip[:, None], -axes, axes)
    return normalize_vector(axes, fallback)


def integrate_and_save(args):
    (
        fname,
        npz_out,
        rmin,
        rmax,
        dr,
        n_theta,
        n_phi,
        tilt_y,
        tilt_x,
        delete_source,
        metric_params,
    ) = args

    reading_h5 = False
    try:
        reading_h5 = True
        with h5py.File(fname, "r") as f:
            missing = [key for key in REQUIRED_H5_DATASETS if key not in f]
            if missing:
                reason = "missing required dataset(s): " + ", ".join(missing)
                return False, format_h5_skip_message(fname, reason)

            ds_uov = f["uov"]
            ds_b = f["B"]
            x1v = f["x1v"][:]
            x2v = f["x2v"][:]
            x3v = f["x3v"][:]
            x1f = f["x1f"][:]
            x2f = f["x2f"][:]
            x3f = f["x3f"][:]
            t_val = float(f.attrs.get("Time", 0.0))

            dens_i, press_i, vx_i, vy_i, vz_i, is_conserved = get_var_indices(f)
            if is_conserved:
                reason = (
                    "does not contain primitive velocity output required for GR stresses"
                )
                return False, format_h5_skip_message(fname, reason)

            bin_edges = np.arange(rmin, rmax + dr * 1.0e-9, dr)
            n_bins = len(bin_edges) - 1
            radii_axis = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            dtheta = np.pi / n_theta
            dphi = 2.0 * np.pi / n_phi
            theta_vals = np.linspace(0.0, np.pi, n_theta, endpoint=False) + 0.5 * dtheta
            phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False) + 0.5 * dphi

            dynamic_disk_frame = tilt_is_dynamical(metric_params.get("tilt_angle_deg", 0.0))
            fixed_r_mat = get_rot_matrix(tilt_y, tilt_x)
            if dynamic_disk_frame and abs(tilt_y) < 1.0e-15 and abs(tilt_x) < 1.0e-15:
                parfile_axis = disk_axis_from_tilt_angle(metric_params.get("tilt_angle_deg", 0.0))
                initial_r_mat = rotation_matrix_from_axis(parfile_axis)
            else:
                # Preserve the original fixed-frame behavior for tilt_angle within
                # one degree of 0 or 180, and for explicit --tilt_y/--tilt_x use.
                initial_r_mat = fixed_r_mat

            r_3d, theta_3d, phi_3d, pts_aligned, pts_sim, coord_dA = make_shell_points(
                radii_axis, theta_vals, phi_vals, initial_r_mat
            )
            rho_grid, press_grid, prim_wv, b_tilde_sim = interpolate_shell_samples(
                ds_uov,
                ds_b,
                x1v,
                x2v,
                x3v,
                x1f,
                x2f,
                x3f,
                dens_i,
                press_i,
                vx_i,
                vy_i,
                vz_i,
                pts_sim,
            )

            shell_geom = build_shell_geometry(
                pts_aligned,
                pts_sim,
                coord_dA,
                initial_r_mat,
                t_val,
                metric_params,
                METRIC_KERNEL_SOURCE,
            )
            proper_dA = shell_geom["proper_dA"]

            if dynamic_disk_frame:
                fallback_axis = initial_r_mat[2]
                disk_lhat = estimate_shell_angular_momentum_axes(
                    rho_grid, prim_wv, pts_sim, proper_dA, fallback_axis
                )
                r_mats = rotation_matrices_from_axes(disk_lhat)
                r_3d, theta_3d, phi_3d, pts_aligned, pts_sim, coord_dA = make_shell_points(
                    radii_axis, theta_vals, phi_vals, r_mats
                )
                rho_grid, press_grid, prim_wv, b_tilde_sim = interpolate_shell_samples(
                    ds_uov,
                    ds_b,
                    x1v,
                    x2v,
                    x3v,
                    x1f,
                    x2f,
                    x3f,
                    dens_i,
                    press_i,
                    vx_i,
                    vy_i,
                    vz_i,
                    pts_sim,
                )
                shell_geom = build_shell_geometry(
                    pts_aligned,
                    pts_sim,
                    coord_dA,
                    r_mats[:, None, None, :, :],
                    t_val,
                    metric_params,
                    METRIC_KERNEL_SOURCE,
                )
                proper_dA = shell_geom["proper_dA"]
                frame_mode_id = 1
                analysis_r_mat = r_mats[:, None, None, :, :]
            else:
                disk_lhat = np.broadcast_to(initial_r_mat[2], (n_bins, 3)).copy()
                r_mats = np.broadcast_to(initial_r_mat, (n_bins, 3, 3)).copy()
                frame_mode_id = 0
                analysis_r_mat = initial_r_mat

            mass_shell = compute_mass_shell(ds_uov, x1v, x2v, x3v, dens_i, rmin, dr, n_bins, bin_edges)
            disk_euler_zyx = euler_zyx_from_rotation_matrix(np.swapaxes(r_mats, -1, -2))

            lambda_mass = np.sum(rho_grid * proper_dA, axis=(1, 2))
            midplane_distance = np.abs(r_3d * (theta_3d - np.pi / 2.0))
            mass_rtheta2 = np.sum(rho_grid * (midplane_distance ** 2) * proper_dA, axis=(1, 2))

            mdot_density, mdot_surface, ur_coord, vr_euler, stress_profiles = valencia_shell_diagnostics(
                rho_grid,
                press_grid,
                prim_wv,
                b_tilde_sim,
                pts_aligned,
                midplane_distance,
                analysis_r_mat,
                shell_geom,
                metric_params,
            )

            m1_real = np.sum(rho_grid * np.cos(phi_3d) * proper_dA, axis=(1, 2))
            m1_imag = np.sum(rho_grid * np.sin(phi_3d) * proper_dA, axis=(1, 2))
            m2_real = np.sum(rho_grid * np.cos(2.0 * phi_3d) * proper_dA, axis=(1, 2))
            m2_imag = np.sum(rho_grid * np.sin(2.0 * phi_3d) * proper_dA, axis=(1, 2))

            mdot_theta_phi_avg = np.nanmean(mdot_density, axis=2)
            rho_theta_phi_avg = np.nanmean(rho_grid, axis=2)
            ur_theta_phi_avg = np.nanmean(ur_coord, axis=2)
            vr_theta_phi_avg = np.nanmean(vr_euler, axis=2)
            mdot_of_radius = np.nansum(mdot_surface, axis=(1, 2))
            ldot_fluid_theta_phi_avg = np.nanmean(
                stress_profiles["ldot_fluid_density"], axis=2
            )
            ldot_em_theta_phi_avg = np.nanmean(
                stress_profiles["ldot_em_density"], axis=2
            )
            ldot_fluid_of_radius = np.nansum(
                stress_profiles["ldot_fluid_surface"], axis=(1, 2)
            )
            ldot_em_of_radius = np.nansum(
                stress_profiles["ldot_em_surface"], axis=(1, 2)
            )

        reading_h5 = False
        np.savez(
            npz_out,
            time=np.array([t_val]),
            radius=radii_axis,
            theta=theta_vals,
            disk_frame_mode_id=np.array([frame_mode_id], dtype=np.int64),
            disk_lhat=np.array([disk_lhat]),
            disk_frame_matrix=np.array([r_mats]),
            disk_euler_zyx=np.array([disk_euler_zyx]),
            disk_euler_zyx_deg=np.array([np.rad2deg(disk_euler_zyx)]),
            lambda_mass=np.array([lambda_mass]),
            mass_shell=np.array([mass_shell]),
            mass_rtheta2=np.array([mass_rtheta2]),
            m1_real=np.array([m1_real]),
            m1_imag=np.array([m1_imag]),
            m2_real=np.array([m2_real]),
            m2_imag=np.array([m2_imag]),
            rho_theta_phi_avg=np.array([rho_theta_phi_avg]),
            mdot_theta_phi_avg=np.array([mdot_theta_phi_avg]),
            ur_theta_phi_avg=np.array([ur_theta_phi_avg]),
            vr_theta_phi_avg=np.array([vr_theta_phi_avg]),
            mdot_of_radius=np.array([mdot_of_radius]),
            ldot_x_fluid_theta_phi_avg=np.array([ldot_fluid_theta_phi_avg[..., 0]]),
            ldot_y_fluid_theta_phi_avg=np.array([ldot_fluid_theta_phi_avg[..., 1]]),
            ldot_z_fluid_theta_phi_avg=np.array([ldot_fluid_theta_phi_avg[..., 2]]),
            ldot_x_em_theta_phi_avg=np.array([ldot_em_theta_phi_avg[..., 0]]),
            ldot_y_em_theta_phi_avg=np.array([ldot_em_theta_phi_avg[..., 1]]),
            ldot_z_em_theta_phi_avg=np.array([ldot_em_theta_phi_avg[..., 2]]),
            ldot_x_fluid_of_radius=np.array([ldot_fluid_of_radius[..., 0]]),
            ldot_y_fluid_of_radius=np.array([ldot_fluid_of_radius[..., 1]]),
            ldot_z_fluid_of_radius=np.array([ldot_fluid_of_radius[..., 2]]),
            ldot_x_em_of_radius=np.array([ldot_em_of_radius[..., 0]]),
            ldot_y_em_of_radius=np.array([ldot_em_of_radius[..., 1]]),
            ldot_z_em_of_radius=np.array([ldot_em_of_radius[..., 2]]),
            stress_mean_press=np.array([stress_profiles["stress_mean_press"]]),
            stress_mean_ptot=np.array([stress_profiles["stress_mean_ptot"]]),
            stress_mean_rhoh=np.array([stress_profiles["stress_mean_rhoh"]]),
            stress_mean_ur=np.array([stress_profiles["stress_mean_ur"]]),
            stress_mean_uphi=np.array([stress_profiles["stress_mean_uphi"]]),
            stress_mean_matter=np.array([stress_profiles["stress_mean_matter"]]),
            stress_mean_maxwell=np.array([stress_profiles["stress_mean_maxwell"]]),
            stress_mean_em_exact=np.array([stress_profiles["stress_mean_em_exact"]]),
            stress_mean_uhat_r=np.array([stress_profiles["stress_mean_uhat_r"]]),
            stress_mean_uhat_phi=np.array([stress_profiles["stress_mean_uhat_phi"]]),
            stress_mean_matter_ortho=np.array([stress_profiles["stress_mean_matter_ortho"]]),
            stress_mean_maxwell_ortho=np.array([stress_profiles["stress_mean_maxwell_ortho"]]),
            stress_mean_em_exact_ortho=np.array([stress_profiles["stress_mean_em_exact_ortho"]]),
            stress_disk_area=np.array([stress_profiles["stress_disk_area"]]),
            scale_height_penna=np.array([stress_profiles["scale_height_penna"]]),
            h_over_r_penna=np.array([stress_profiles["h_over_r_penna"]]),
            penna_scale_num=np.array([stress_profiles["penna_scale_num"]]),
            penna_scale_den=np.array([stress_profiles["penna_scale_den"]]),
            penna_vert_weight_theta=np.array([stress_profiles["penna_vert_weight_theta"]]),
            penna_ptot_theta_weighted=np.array([stress_profiles["penna_ptot_theta_weighted"]]),
            penna_rey_theta_weighted=np.array([stress_profiles["penna_rey_theta_weighted"]]),
            penna_max_theta_weighted=np.array([stress_profiles["penna_max_theta_weighted"]]),
            penna_em_exact_theta_weighted=np.array([stress_profiles["penna_em_exact_theta_weighted"]]),
            penna_total_theta_weighted=np.array([stress_profiles["penna_total_theta_weighted"]]),
            penna_shear_rey_theta_weighted=np.array([stress_profiles["penna_shear_rey_theta_weighted"]]),
            penna_shear_max_theta_weighted=np.array([stress_profiles["penna_shear_max_theta_weighted"]]),
            penna_shear_em_exact_theta_weighted=np.array([stress_profiles["penna_shear_em_exact_theta_weighted"]]),
            penna_shear_total_theta_weighted=np.array([stress_profiles["penna_shear_total_theta_weighted"]]),
            penna_alpha_rey_theta_weighted=np.array([stress_profiles["penna_alpha_rey_theta_weighted"]]),
            penna_alpha_max_theta_weighted=np.array([stress_profiles["penna_alpha_max_theta_weighted"]]),
            penna_alpha_em_exact_theta_weighted=np.array([stress_profiles["penna_alpha_em_exact_theta_weighted"]]),
            penna_alpha_theta_weighted=np.array([stress_profiles["penna_alpha_theta_weighted"]]),
            penna_alpha_shear_rey_theta_weighted=np.array([stress_profiles["penna_alpha_shear_rey_theta_weighted"]]),
            penna_alpha_shear_max_theta_weighted=np.array([stress_profiles["penna_alpha_shear_max_theta_weighted"]]),
            penna_alpha_shear_em_exact_theta_weighted=np.array([stress_profiles["penna_alpha_shear_em_exact_theta_weighted"]]),
            penna_alpha_shear_theta_weighted=np.array([stress_profiles["penna_alpha_shear_theta_weighted"]]),
            penna_mean_flow_ut=np.array([stress_profiles["penna_mean_flow_ut"]]),
            penna_mean_flow_ur=np.array([stress_profiles["penna_mean_flow_ur"]]),
            penna_mean_flow_utheta=np.array([stress_profiles["penna_mean_flow_utheta"]]),
            penna_mean_flow_uphi=np.array([stress_profiles["penna_mean_flow_uphi"]]),
            rmin_used=rmin,
        )

        if delete_source:
            try:
                os.remove(fname)
            except OSError:
                pass

        return True, f"Saved {os.path.basename(npz_out)}"
    except (OSError, KeyError) as exc:
        if reading_h5:
            return False, format_h5_skip_message(fname, exc)
        import traceback

        return False, f"Error in {fname}: {exc}\n{traceback.format_exc()}"
    except Exception as exc:
        import traceback

        return False, f"Error in {fname}: {exc}\n{traceback.format_exc()}"


def trapezoid_weights(coords):
    coords = np.asarray(coords, dtype=np.float64)
    npts = coords.size
    if npts == 0:
        return np.array([], dtype=np.float64)
    if npts == 1:
        return np.array([1.0], dtype=np.float64)

    diffs = np.diff(coords)
    if np.any(~np.isfinite(diffs)) or np.any(diffs <= 0.0):
        return np.ones(npts, dtype=np.float64)

    weights = np.empty(npts, dtype=np.float64)
    weights[0] = 0.5 * diffs[0]
    weights[-1] = 0.5 * diffs[-1]
    if npts > 2:
        weights[1:-1] = 0.5 * (coords[2:] - coords[:-2])
    return weights


def stitch_archives(output_filename, files=None, npz_pattern=None, omega_bin=None):
    if files is None:
        if npz_pattern is None:
            return
        files = sorted(glob.glob(npz_pattern))
    if not files:
        return

    data_keys = [
        "time",
        "disk_frame_mode_id",
        "disk_lhat",
        "disk_frame_matrix",
        "disk_euler_zyx",
        "disk_euler_zyx_deg",
        "lambda_mass",
        "mass_shell",
        "mass_rtheta2",
        "m1_real",
        "m1_imag",
        "m2_real",
        "m2_imag",
        "rho_theta_phi_avg",
        "mdot_theta_phi_avg",
        "ur_theta_phi_avg",
        "vr_theta_phi_avg",
        "mdot_of_radius",
        "ldot_x_fluid_theta_phi_avg",
        "ldot_y_fluid_theta_phi_avg",
        "ldot_z_fluid_theta_phi_avg",
        "ldot_x_em_theta_phi_avg",
        "ldot_y_em_theta_phi_avg",
        "ldot_z_em_theta_phi_avg",
        "ldot_x_fluid_of_radius",
        "ldot_y_fluid_of_radius",
        "ldot_z_fluid_of_radius",
        "ldot_x_em_of_radius",
        "ldot_y_em_of_radius",
        "ldot_z_em_of_radius",
        "penna_mean_flow_ut",
        "penna_mean_flow_ur",
        "penna_mean_flow_utheta",
        "penna_mean_flow_uphi",
    ]
    radii_axis = None
    theta_axis = None
    rmin_used = None
    stress_keys = [
        "stress_mean_press",
        "stress_mean_ptot",
        "stress_mean_rhoh",
        "stress_mean_ur",
        "stress_mean_uphi",
        "stress_mean_matter",
        "stress_mean_maxwell",
        "stress_mean_em_exact",
        "stress_mean_uhat_r",
        "stress_mean_uhat_phi",
        "stress_mean_matter_ortho",
        "stress_mean_maxwell_ortho",
        "stress_mean_em_exact_ortho",
        "stress_disk_area",
        "scale_height_penna",
        "h_over_r_penna",
        "penna_scale_num",
        "penna_scale_den",
        "penna_vert_weight_theta",
        "penna_ptot_theta_weighted",
        "penna_rey_theta_weighted",
        "penna_max_theta_weighted",
        "penna_em_exact_theta_weighted",
        "penna_total_theta_weighted",
        "penna_shear_rey_theta_weighted",
        "penna_shear_max_theta_weighted",
        "penna_shear_em_exact_theta_weighted",
        "penna_shear_total_theta_weighted",
        "penna_alpha_rey_theta_weighted",
        "penna_alpha_max_theta_weighted",
        "penna_alpha_em_exact_theta_weighted",
        "penna_alpha_theta_weighted",
        "penna_alpha_shear_rey_theta_weighted",
        "penna_alpha_shear_max_theta_weighted",
        "penna_alpha_shear_em_exact_theta_weighted",
        "penna_alpha_shear_theta_weighted",
    ]
    valid_entries = []
    shapes = {}
    dtypes = {}

    for fname in files:
        try:
            with np.load(fname, mmap_mode="r") as data:
                if any(key not in data.files for key in data_keys):
                    continue
                if radii_axis is None:
                    radii_axis = data["radius"]
                    theta_axis = data["theta"]
                    rmin_used = data["rmin_used"]
                    for key in data_keys:
                        shapes[key] = data[key][0].shape
                        dtypes[key] = data[key].dtype
                valid_entries.append((float(data["time"][0]), fname))
        except Exception:
            pass

    if not valid_entries:
        return

    valid_entries.sort(key=lambda item: item[0])
    nfiles = len(valid_entries)
    times = np.array([item[0] for item in valid_entries], dtype=np.float64)
    time_weights = trapezoid_weights(times)
    if omega_bin is not None and np.isfinite(omega_bin) and omega_bin > 0.0:
        orbit_period = 2.0 * np.pi / omega_bin
        orbit_coords = times / orbit_period
        orbit_weights = trapezoid_weights(orbit_coords)
    else:
        orbit_coords = np.full(nfiles, np.nan, dtype=np.float64)
        orbit_weights = time_weights.copy()

    stress_sums = None
    stress_count = 0
    stress_weight_total = 0.0
    stress_time_weight_total = 0.0
    out_data = {}
    for key in data_keys:
        out_shape = (nfiles,) + shapes[key]
        out_data[key] = np.empty(out_shape, dtype=dtypes[key])

    for idx, (_time, fname) in enumerate(valid_entries):
        with np.load(fname, mmap_mode="r") as data:
            for key in data_keys:
                out_data[key][idx] = data[key][0]
            if all(key in data.files for key in stress_keys):
                if stress_sums is None:
                    stress_sums = {key: np.zeros_like(data[key][0], dtype=np.float64) for key in stress_keys}
                weight = float(orbit_weights[idx])
                for key in stress_keys:
                    stress_sums[key] += weight * data[key][0]
                stress_weight_total += weight
                stress_time_weight_total += float(time_weights[idx])
                stress_count += 1

    out_data["radius"] = radii_axis
    out_data["theta"] = theta_axis
    out_data["rmin_used"] = rmin_used
    out_data["scale_height"] = np.sqrt(out_data["mass_rtheta2"] / (out_data["lambda_mass"] + 1.0e-30))
    out_data["stress_snapshot_time_weight"] = time_weights
    out_data["stress_snapshot_orbit"] = orbit_coords
    out_data["stress_snapshot_orbit_weight"] = orbit_weights
    out_data["stress_num_files"] = np.array([stress_count], dtype=np.int64)
    out_data["stress_time_weight_total"] = np.array([stress_time_weight_total], dtype=np.float64)
    out_data["stress_orbit_weight_total"] = np.array([stress_weight_total], dtype=np.float64)

    if stress_weight_total > 0.0:
        mean_press = stress_sums["stress_mean_press"] / stress_weight_total
        mean_ptot = stress_sums["stress_mean_ptot"] / stress_weight_total
        mean_rhoh = stress_sums["stress_mean_rhoh"] / stress_weight_total
        mean_ur = stress_sums["stress_mean_ur"] / stress_weight_total
        mean_uphi = stress_sums["stress_mean_uphi"] / stress_weight_total
        mean_matter = stress_sums["stress_mean_matter"] / stress_weight_total
        mean_maxwell = stress_sums["stress_mean_maxwell"] / stress_weight_total
        mean_em_exact = stress_sums["stress_mean_em_exact"] / stress_weight_total
        mean_uhat_r = stress_sums["stress_mean_uhat_r"] / stress_weight_total
        mean_uhat_phi = stress_sums["stress_mean_uhat_phi"] / stress_weight_total
        mean_matter_ortho = stress_sums["stress_mean_matter_ortho"] / stress_weight_total
        mean_maxwell_ortho = stress_sums["stress_mean_maxwell_ortho"] / stress_weight_total
        mean_em_exact_ortho = stress_sums["stress_mean_em_exact_ortho"] / stress_weight_total
        mean_disk_area = stress_sums["stress_disk_area"] / stress_weight_total

        stress_adv = mean_rhoh * mean_ur * mean_uphi
        stress_rey = mean_matter - stress_adv
        stress_adv_ortho = mean_rhoh * mean_uhat_r * mean_uhat_phi
        stress_rey_ortho = mean_matter_ortho - stress_adv_ortho
        inv_press = 1.0 / np.maximum(mean_press, 1.0e-300)
        inv_ptot = 1.0 / np.maximum(mean_ptot, 1.0e-300)

        out_data["stress_time_avg_press"] = mean_press
        out_data["stress_time_avg_ptot"] = mean_ptot
        out_data["stress_time_avg_rhoh"] = mean_rhoh
        out_data["stress_time_avg_ur"] = mean_ur
        out_data["stress_time_avg_uphi"] = mean_uphi
        out_data["stress_time_avg_matter"] = mean_matter
        out_data["stress_time_avg_maxwell"] = mean_maxwell
        out_data["stress_time_avg_em_exact"] = mean_em_exact
        out_data["stress_time_avg_uhat_r"] = mean_uhat_r
        out_data["stress_time_avg_uhat_phi"] = mean_uhat_phi
        out_data["stress_time_avg_matter_ortho"] = mean_matter_ortho
        out_data["stress_time_avg_maxwell_ortho"] = mean_maxwell_ortho
        out_data["stress_time_avg_em_exact_ortho"] = mean_em_exact_ortho
        out_data["stress_time_avg_disk_area"] = mean_disk_area
        out_data["stress_adv"] = stress_adv
        out_data["stress_rey"] = stress_rey
        out_data["stress_adv_ortho"] = stress_adv_ortho
        out_data["stress_rey_ortho"] = stress_rey_ortho
        out_data["alpha_rey_pgas"] = stress_rey * inv_press
        out_data["alpha_max_pgas"] = mean_maxwell * inv_press
        out_data["alpha_eff_pgas"] = (stress_rey + mean_maxwell) * inv_press
        out_data["alpha_em_exact_pgas"] = mean_em_exact * inv_press
        out_data["alpha_eff_gr_pgas"] = (stress_rey + mean_em_exact) * inv_press
        out_data["alpha_rey"] = stress_rey * inv_ptot
        out_data["alpha_max"] = mean_maxwell * inv_ptot
        out_data["alpha_eff"] = (stress_rey + mean_maxwell) * inv_ptot
        out_data["alpha_em_exact"] = mean_em_exact * inv_ptot
        out_data["alpha_eff_gr"] = (stress_rey + mean_em_exact) * inv_ptot
        out_data["alpha_rey_ortho_pgas"] = stress_rey_ortho * inv_press
        out_data["alpha_max_ortho_pgas"] = mean_maxwell_ortho * inv_press
        out_data["alpha_eff_ortho_pgas"] = (stress_rey_ortho + mean_maxwell_ortho) * inv_press
        out_data["alpha_em_exact_ortho_pgas"] = mean_em_exact_ortho * inv_press
        out_data["alpha_eff_gr_ortho_pgas"] = (stress_rey_ortho + mean_em_exact_ortho) * inv_press
        out_data["alpha_rey_ortho"] = stress_rey_ortho * inv_ptot
        out_data["alpha_max_ortho"] = mean_maxwell_ortho * inv_ptot
        out_data["alpha_eff_ortho"] = (stress_rey_ortho + mean_maxwell_ortho) * inv_ptot
        out_data["alpha_em_exact_ortho"] = mean_em_exact_ortho * inv_ptot
        out_data["alpha_eff_gr_ortho"] = (stress_rey_ortho + mean_em_exact_ortho) * inv_ptot

        penna_scale_height_exact = np.where(
            stress_sums["penna_scale_den"] > 0.0,
            np.sqrt(np.maximum(stress_sums["penna_scale_num"] / stress_sums["penna_scale_den"], 0.0)),
            np.nan,
        )
        penna_h_over_r_exact = penna_scale_height_exact / np.maximum(radii_axis, 1.0e-300)
        penna_mask_theta = np.abs(theta_axis[None, :] - np.pi / 2.0) <= penna_h_over_r_exact[:, None]
        penna_weight_theta = stress_sums["penna_vert_weight_theta"]
        penna_weight_masked = np.where(penna_mask_theta, penna_weight_theta, 0.0)
        penna_weight_sum = np.sum(penna_weight_masked, axis=1)

        def penna_vertical_average(weighted_theta_sum):
            numer = np.sum(np.where(penna_mask_theta, weighted_theta_sum, 0.0), axis=1)
            return np.where(penna_weight_sum > 0.0, numer / penna_weight_sum, np.nan)

        mean_ptot_penna = penna_vertical_average(stress_sums["penna_ptot_theta_weighted"])
        mean_rey_penna = penna_vertical_average(stress_sums["penna_rey_theta_weighted"])
        mean_max_penna = penna_vertical_average(stress_sums["penna_max_theta_weighted"])
        mean_em_exact_penna = penna_vertical_average(stress_sums["penna_em_exact_theta_weighted"])
        mean_total_penna = penna_vertical_average(stress_sums["penna_total_theta_weighted"])
        mean_shear_rey_penna = penna_vertical_average(stress_sums["penna_shear_rey_theta_weighted"])
        mean_shear_max_penna = penna_vertical_average(stress_sums["penna_shear_max_theta_weighted"])
        mean_shear_em_exact_penna = penna_vertical_average(
            stress_sums["penna_shear_em_exact_theta_weighted"]
        )
        mean_shear_total_penna = penna_vertical_average(stress_sums["penna_shear_total_theta_weighted"])
        mean_alpha_rey_penna = penna_vertical_average(stress_sums["penna_alpha_rey_theta_weighted"])
        mean_alpha_max_penna = penna_vertical_average(stress_sums["penna_alpha_max_theta_weighted"])
        mean_alpha_em_exact_penna = penna_vertical_average(stress_sums["penna_alpha_em_exact_theta_weighted"])
        mean_alpha_penna = penna_vertical_average(stress_sums["penna_alpha_theta_weighted"])
        mean_alpha_shear_rey_penna = penna_vertical_average(
            stress_sums["penna_alpha_shear_rey_theta_weighted"]
        )
        mean_alpha_shear_max_penna = penna_vertical_average(
            stress_sums["penna_alpha_shear_max_theta_weighted"]
        )
        mean_alpha_shear_em_exact_penna = penna_vertical_average(
            stress_sums["penna_alpha_shear_em_exact_theta_weighted"]
        )
        mean_alpha_shear_penna = penna_vertical_average(stress_sums["penna_alpha_shear_theta_weighted"])

        out_data["h_over_r_penna"] = penna_h_over_r_exact
        out_data["scale_height_penna"] = penna_scale_height_exact
        out_data["h_over_r_penna_time_avg"] = penna_h_over_r_exact
        out_data["scale_height_penna_time_avg"] = penna_scale_height_exact
        out_data["stress_time_avg_penna_weight"] = penna_weight_sum
        out_data["stress_time_avg_ptot_penna"] = mean_ptot_penna
        out_data["stress_time_avg_rey_penna"] = mean_rey_penna
        out_data["stress_time_avg_max_penna"] = mean_max_penna
        out_data["stress_time_avg_em_exact_penna"] = mean_em_exact_penna
        out_data["stress_time_avg_total_penna"] = mean_total_penna
        out_data["stress_time_avg_shear_rey_penna"] = mean_shear_rey_penna
        out_data["stress_time_avg_shear_max_penna"] = mean_shear_max_penna
        out_data["stress_time_avg_shear_em_exact_penna"] = mean_shear_em_exact_penna
        out_data["stress_time_avg_shear_total_penna"] = mean_shear_total_penna
        out_data["alpha_rey_penna"] = mean_alpha_rey_penna
        out_data["alpha_max_penna"] = mean_alpha_max_penna
        out_data["alpha_em_exact_penna"] = mean_alpha_em_exact_penna
        out_data["alpha_penna"] = mean_alpha_penna
        out_data["alpha_shear_rey_penna"] = mean_alpha_shear_rey_penna
        out_data["alpha_shear_max_penna"] = mean_alpha_shear_max_penna
        out_data["alpha_shear_em_exact_penna"] = mean_alpha_shear_em_exact_penna
        out_data["alpha_shear_penna"] = mean_alpha_shear_penna

    np.savez(output_filename, **out_data)


def process_files(
    run_dir,
    rmin,
    rmax,
    dr,
    n_theta,
    n_phi,
    tilt_y,
    tilt_x,
    output_filename,
    tstart=None,
    tend=None,
    time_units="orbit",
    delete_source=False,
    nproc=1,
):
    run_path = Path(run_dir).resolve()
    parfile = run_path / "parfile.par"
    bin_dir = run_path / "bin"
    if not parfile.exists():
        raise FileNotFoundError(f"Missing parfile: {parfile}")
    if not bin_dir.exists():
        raise FileNotFoundError(f"Missing bin directory: {bin_dir}")

    metric_params = parse_athena_parfile(parfile)
    if tilt_is_dynamical(metric_params.get("tilt_angle_deg", 0.0)):
        print(
            "Using angular-momentum-aligned disk frame "
            f"(tilt_angle={metric_params['tilt_angle_deg']:.6g} deg)."
        )
    else:
        print(
            "Using fixed disk frame "
            f"(tilt_angle={metric_params['tilt_angle_deg']:.6g} deg)."
        )
    if metric_params.get("use_traj_table", False):
        print(f"Using trajectory table: {metric_params['traj_file']}")
    orbit_period = 2.0 * np.pi / metric_params["om"]
    if time_units == "orbit":
        tstart_abs = None if tstart is None else tstart * orbit_period
        tend_abs = None if tend is None else tend * orbit_period
    elif time_units == "code":
        tstart_abs = tstart
        tend_abs = tend
    else:
        raise ValueError(f"Unsupported time_units={time_units}")

    all_files = sorted(str(path) for path in bin_dir.glob("torus.mhd_w_bcc.*.athdf"))
    if not all_files:
        return

    skip_log_path = run_path / f"{Path(output_filename).stem}_skipped_h5.log"
    skipped_messages = []
    skipped_files = set()
    selected_files = []
    tasks = []
    script_mtime = Path(__file__).stat().st_mtime
    for fname in all_files:
        t_val, skip_message = inspect_h5_dump(fname)
        if skip_message is not None:
            print(skip_message)
            skipped_messages.append(skip_message)
            skipped_files.add(fname)
            continue
        if tstart_abs is not None and t_val < tstart_abs:
            continue
        if tend_abs is not None and t_val > tend_abs:
            continue

        selected_files.append(fname)
        npz_out = fname[:-6] + ".npz"
        should_process = True
        if os.path.exists(npz_out):
            if os.path.getmtime(npz_out) > max(os.path.getmtime(fname), script_mtime):
                should_process = False
            else:
                try:
                    os.remove(npz_out)
                except OSError:
                    pass

        if should_process:
            tasks.append(
                (
                    fname,
                    npz_out,
                    rmin,
                    rmax,
                    dr,
                    n_theta,
                    n_phi,
                    tilt_y,
                    tilt_x,
                    delete_source,
                    metric_params,
                )
            )

    if not selected_files:
        write_skip_log(skip_log_path, skipped_messages)
        print("No input dumps found in the requested time range.")
        return

    if tasks:
        failures = []
        if nproc > 1:
            with Pool(nproc) as pool:
                for i, result in enumerate(pool.imap(integrate_and_save, tasks), start=1):
                    ok, message = result
                    if not ok:
                        if is_h5_skip_message(message):
                            skipped_messages.append(message)
                            skipped_files.add(tasks[i - 1][0])
                        else:
                            failures.append(message)
                    sys.stdout.write(f"\r[Proc] {i}/{len(tasks)}...")
                    sys.stdout.flush()
        else:
            for i, task in enumerate(tasks, start=1):
                ok, message = integrate_and_save(task)
                if not ok:
                    if is_h5_skip_message(message):
                        skipped_messages.append(message)
                        skipped_files.add(task[0])
                    else:
                        failures.append(message)
                sys.stdout.write(f"\r[Proc] {i}/{len(tasks)}...")
                sys.stdout.flush()
        print("\nBatch processing complete.")
        write_skip_log(skip_log_path, skipped_messages)
        if skipped_messages:
            print(f"Wrote skipped HDF5 log: {skip_log_path}")
        if failures:
            raise RuntimeError("\n\n".join(failures))
    else:
        write_skip_log(skip_log_path, skipped_messages)

    selected_npz_files = [
        fname[:-6] + ".npz"
        for fname in selected_files
        if fname not in skipped_files and os.path.exists(fname[:-6] + ".npz")
    ]
    stitch_archives(
        str(run_path / output_filename),
        files=selected_npz_files,
        omega_bin=metric_params["om"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("-n", "--nproc", type=int, default=1)
    parser.add_argument("--rmin", type=float, default=20.0)
    parser.add_argument("--rmax", type=float, default=400.0)
    parser.add_argument("--dr", type=float, default=10.0)
    parser.add_argument("--ntheta", type=int, default=96)
    parser.add_argument("--nphi", type=int, default=128)
    parser.add_argument("--tilt_y", type=float, default=0.0)
    parser.add_argument("--tilt_x", type=float, default=0.0)
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tend", type=float, default=None)
    parser.add_argument("--time-units", choices=["orbit", "code"], default="orbit")
    parser.add_argument("--delete", action="store_true", default=False)
    parser.add_argument("--output", default="analysis_disk_structure.npz")
    args = parser.parse_args()

    process_files(
        args.run_dir,
        args.rmin,
        args.rmax,
        args.dr,
        args.ntheta,
        args.nphi,
        args.tilt_y,
        args.tilt_x,
        args.output,
        args.tstart,
        args.tend,
        args.time_units,
        delete_source=args.delete,
        nproc=args.nproc,
    )
