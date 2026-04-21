import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


DYNBBH_CPP = "/home/hzhu/athenak/src/pgen/dynbbh.cpp"

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


def parse_athena_parfile(parfile_path):
    values = {}
    for raw_line in Path(parfile_path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("<") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        try:
            values[key] = float(value)
        except ValueError:
            continue

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
        "adjust_mass1": values.get("adjust_mass1", 1.0),
        "adjust_mass2": values.get("adjust_mass2", 1.0),
        "a1_buffer": values.get("a1_buffer", 0.01),
        "a2_buffer": values.get("a2_buffer", 0.01),
        "cutoff_floor": values.get("cutoff_floor", 1.0e-4),
    }
    params["om"] = params["sep"] ** (-1.5)
    return params


def load_superposed_metric_kernel(source_path):
    text = Path(source_path).read_text()
    start = text.index("  Real o1 = 1.4142135623730951;")
    end = text.index("  /* Initialize the flat part */", start)

    translated = []
    for raw in text[start:end].splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("/*"):
            continue
        line = line.rstrip(";")
        if line.startswith("Real "):
            line = line[5:]
        line = re.sub(r"\bpow\(", "np.power(", line)
        line = re.sub(r"\bsqrt\(", "np.sqrt(", line)
        line = re.sub(r"\bfabs\(", "np.abs(", line)
        line = re.sub(r"([A-Za-z_]\w*)\[(\d+)\]\[(\d+)\]", r"\1[..., \2, \3]", line)
        translated.append(line)
    return "\n".join(translated)


METRIC_KERNEL_SOURCE = load_superposed_metric_kernel(DYNBBH_CPP)


def find_traj_t(time, params):
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
        "v1x": -r_bh1_0 * params["om"] * np.sin(om_t) + 1.0e-40,
        "v1y": r_bh1_0 * params["om"] * np.cos(om_t) + 1.0e-40,
        "v1z": 1.0e-40,
        "v2x": -r_bh2_0 * params["om"] * np.sin(om_t) + 1.0e-40,
        "v2y": r_bh2_0 * params["om"] * np.cos(om_t) + 1.0e-40,
        "v2z": 1.0e-40,
        "a1x": params["a1"] * np.sin(params["th_a1"]) * np.cos(params["ph_a1"]),
        "a1y": params["a1"] * np.sin(params["th_a1"]) * np.sin(params["ph_a1"]),
        "a1z": params["a1"] * np.cos(params["th_a1"]),
        "a2x": params["a2"] * np.sin(params["th_a2"]) * np.cos(params["ph_a2"]),
        "a2y": params["a2"] * np.sin(params["th_a2"]) * np.sin(params["ph_a2"]),
        "a2z": params["a2"] * np.cos(params["th_a2"]),
        "m1_t": 1.0 / (params["q"] + 1.0),
        "m2_t": 1.0 - 1.0 / (params["q"] + 1.0),
    }


def superposed_bbh_metric(x, y, z, time, params, kernel_source):
    x, y, z = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    shape = x.shape

    loc = {"x": x, "y": y, "z": z}
    loc.update(find_traj_t(time, params))

    v1 = np.sqrt(loc["v1x"] ** 2 + loc["v1y"] ** 2 + loc["v1z"] ** 2)
    v2 = np.sqrt(loc["v2x"] ** 2 + loc["v2y"] ** 2 + loc["v2z"] ** 2)

    oo4 = np.sqrt(1.0 - v1 * v1)
    oo5 = 1.0 / oo4
    oo14 = np.sqrt(1.0 - v2 * v2)
    oo15 = 1.0 / oo14
    oo18, oo21, oo22 = -loc["xi1x"], -loc["xi1y"], -loc["xi1z"]
    oo23, oo26, oo27 = -loc["xi2x"], -loc["xi2y"], -loc["xi2z"]
    oo19 = 1.0 / (v1 * v1)
    oo24 = 1.0 / (v2 * v2)
    oo20 = -1.0 + oo4
    oo25 = -1.0 + oo14

    oo32 = (
        loc["xi1y"] * loc["v1y"]
        + loc["xi1z"] * loc["v1z"]
        + loc["v1x"] * (-x + loc["xi1x"])
        + loc["v1y"] * (-y)
        + loc["v1z"] * (-z)
    )
    oo37 = (
        loc["v2x"] * (-x + loc["xi2x"])
        + loc["xi2y"] * loc["v2y"]
        + loc["xi2z"] * loc["v2z"]
        + loc["v2y"] * (-y)
        + loc["v2z"] * (-z)
    )

    x1bh1 = (oo18 + x) - oo20 * (
        oo5
        * (
            loc["v1x"]
            * (
                (
                    (oo18 + x) * loc["v1x"]
                    + ((oo21 + y) * loc["v1y"] + (oo22 + z) * loc["v1z"])
                )
                * oo19
            )
        )
    )
    x1bh2 = (oo23 + x) - oo24 * (
        oo25
        * (
            loc["v2x"]
            * (
                (
                    (oo23 + x) * loc["v2x"]
                    + ((oo26 + y) * loc["v2y"] + (oo27 + z) * loc["v2z"])
                )
                * oo15
            )
        )
    )
    x2bh1 = oo21 + (oo20 * (oo32 * (oo5 * (loc["v1y"] * oo19))) + y)
    x2bh2 = oo26 + (oo24 * (oo25 * (oo37 * (loc["v2y"] * oo15))) + y)
    x3bh1 = oo22 + (oo20 * (oo32 * (oo5 * (loc["v1z"] * oo19))) + z)
    x3bh2 = oo27 + (oo24 * (oo25 * (oo37 * (loc["v2z"] * oo15))) + z)

    a1_t = np.sqrt(loc["a1x"] ** 2 + loc["a1y"] ** 2 + loc["a1z"] ** 2 + 1.0e-40)
    a2_t = np.sqrt(loc["a2x"] ** 2 + loc["a2y"] ** 2 + loc["a2z"] ** 2 + 1.0e-40)
    a1 = a1_t * params["adjust_mass1"]
    m1 = loc["m1_t"] * params["adjust_mass1"]
    a2 = a2_t * params["adjust_mass2"]
    m2 = loc["m2_t"] * params["adjust_mass2"]

    rbh1 = np.sqrt(x1bh1 * x1bh1 + x2bh1 * x2bh1 + x3bh1 * x3bh1)
    rbh2 = np.sqrt(x1bh2 * x1bh2 + x2bh2 * x2bh2 + x3bh2 * x3bh2)
    rbh1_cutoff = np.abs(a1) * (1.0 + params["a1_buffer"]) + params["cutoff_floor"]
    rbh2_cutoff = np.abs(a2) * (1.0 + params["a2_buffer"]) + params["cutoff_floor"]

    x3bh1 = np.where(rbh1 < rbh1_cutoff, np.where(x3bh1 > 0.0, rbh1_cutoff, -rbh1_cutoff), x3bh1)
    x3bh2 = np.where(rbh2 < rbh2_cutoff, np.where(x3bh2 > 0.0, rbh2_cutoff, -rbh2_cutoff), x3bh2)

    namespace = {
        "np": np,
        "KS1": np.zeros(shape + (4, 4), dtype=np.float64),
        "KS2": np.zeros(shape + (4, 4), dtype=np.float64),
        "J1": np.zeros(shape + (4, 4), dtype=np.float64),
        "J2": np.zeros(shape + (4, 4), dtype=np.float64),
        "x": x,
        "y": y,
        "z": z,
        "x1BH1": x1bh1,
        "x1BH2": x1bh2,
        "x2BH1": x2bh1,
        "x2BH2": x2bh2,
        "x3BH1": x3bh1,
        "x3BH2": x3bh2,
        "a1x": loc["a1x"],
        "a1y": loc["a1y"],
        "a1z": loc["a1z"],
        "a2x": loc["a2x"],
        "a2y": loc["a2y"],
        "a2z": loc["a2z"],
        "v1": v1,
        "v2": v2,
        "v1x": loc["v1x"],
        "v1y": loc["v1y"],
        "v1z": loc["v1z"],
        "v2x": loc["v2x"],
        "v2y": loc["v2y"],
        "v2z": loc["v2z"],
        "a1": a1,
        "a2": a2,
        "m1": m1,
        "m2": m2,
    }
    exec(kernel_source, namespace)

    ks1 = namespace["KS1"]
    ks2 = namespace["KS2"]
    j1 = namespace["J1"]
    j2 = namespace["J2"]

    gcov = np.zeros(shape + (4, 4), dtype=np.float64)
    gcov[..., 0, 0] = -1.0
    gcov[..., 1, 1] = 1.0
    gcov[..., 2, 2] = 1.0
    gcov[..., 3, 3] = 1.0

    for i in range(4):
        for j in range(i, 4):
            total = np.zeros(shape, dtype=np.float64)
            for m in range(4):
                for n in range(4):
                    total += j2[..., m, i] * j2[..., n, j] * ks2[..., m, n]
                    total += j1[..., m, i] * j1[..., n, j] * ks1[..., m, n]
            gcov[..., i, j] += total
            gcov[..., j, i] = gcov[..., i, j]
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
    return np.einsum("ia,jb,...ab->...ij", r_mat, r_mat, tensor)


def rotate_spatial_vector(r_mat, vector):
    return np.einsum("ia,...a->...i", r_mat, vector)


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
    ehat_phi_mean = frame[..., :, 3]

    t_rey_cov_penna = rho_h[..., None, None] * u_cov_sph[..., :, None] * u_cov_sph[..., None, :] + press[..., None, None] * gcov_sph

    stress_rey_penna_local = np.einsum(
        "...a,...ab,...b->...", ehat_r_mean, t_rey_cov_penna, ehat_phi_mean
    )
    uhat_r_penna = np.einsum("...a,...a->...", ehat_r_mean, u_cov_sph)
    uhat_phi_penna = np.einsum("...a,...a->...", ehat_phi_mean, u_cov_sph)
    bhat_r_penna = np.einsum("...a,...a->...", ehat_r_mean, b_cov_sph)
    bhat_phi_penna = np.einsum("...a,...a->...", ehat_phi_mean, b_cov_sph)
    stress_max_penna_local = -bhat_r_penna * bhat_phi_penna
    stress_em_exact_penna_local = b_sq * uhat_r_penna * uhat_phi_penna + stress_max_penna_local
    stress_tot_penna_local = stress_rey_penna_local + stress_em_exact_penna_local
    alpha_rey_penna_local = stress_rey_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_max_penna_local = stress_max_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_em_exact_penna_local = stress_em_exact_penna_local / np.maximum(total_pressure, 1.0e-300)
    alpha_penna_local = stress_tot_penna_local / np.maximum(total_pressure, 1.0e-300)

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
    penna_alpha_rey_theta = np.mean(alpha_rey_penna_local, axis=2)
    penna_alpha_max_theta = np.mean(alpha_max_penna_local, axis=2)
    penna_alpha_em_exact_theta = np.mean(alpha_em_exact_penna_local, axis=2)
    penna_alpha_theta = np.mean(alpha_penna_local, axis=2)

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
        "penna_alpha_rey_theta_weighted": penna_alpha_rey_theta * penna_vert_weight_theta,
        "penna_alpha_max_theta_weighted": penna_alpha_max_theta * penna_vert_weight_theta,
        "penna_alpha_em_exact_theta_weighted": penna_alpha_em_exact_theta * penna_vert_weight_theta,
        "penna_alpha_theta_weighted": penna_alpha_theta * penna_vert_weight_theta,
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

    try:
        with h5py.File(fname, "r") as f:
            if "uov" not in f or "x1v" not in f or "x1f" not in f:
                return False, f"Corrupted structure in {fname}"

            ds_uov = f["uov"]
            ds_b = f["B"] if "B" in f else None
            if ds_b is None:
                return False, f"{fname} is missing the B dataset required for GR stress diagnostics"
            x1v = f["x1v"][:]
            x2v = f["x2v"][:]
            x3v = f["x3v"][:]
            x1f = f["x1f"][:]
            x2f = f["x2f"][:]
            x3f = f["x3f"][:]
            t_val = float(f.attrs.get("Time", 0.0))

            dens_i, press_i, vx_i, vy_i, vz_i, is_conserved = get_var_indices(f)
            if is_conserved:
                return False, f"{fname} does not contain primitive velocity output required for GR stresses"

            bin_edges = np.arange(rmin, rmax + dr * 1.0e-9, dr)
            n_bins = len(bin_edges) - 1
            radii_axis = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            dtheta = np.pi / n_theta
            dphi = 2.0 * np.pi / n_phi
            theta_vals = np.linspace(0.0, np.pi, n_theta, endpoint=False) + 0.5 * dtheta
            phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False) + 0.5 * dphi

            r_3d, theta_3d, phi_3d = np.meshgrid(radii_axis, theta_vals, phi_vals, indexing="ij")
            x_aligned = r_3d * np.sin(theta_3d) * np.cos(phi_3d)
            y_aligned = r_3d * np.sin(theta_3d) * np.sin(phi_3d)
            z_aligned = r_3d * np.cos(theta_3d)

            r_mat = get_rot_matrix(tilt_y, tilt_x)
            pts_aligned = np.stack((x_aligned, y_aligned, z_aligned), axis=-1)
            pts_sim = np.einsum("ij,...j->...i", r_mat.T, pts_aligned)

            coord_dA = (r_3d ** 2) * np.sin(theta_3d) * dtheta * dphi

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

            mass_shell = np.zeros(n_bins, dtype=np.float64)

            for b in range(ds_uov.shape[1]):
                uov_block = ds_uov[:, b, :, :, :]
                b_block = ds_b[:, b, :, :, :] if ds_b is not None else None

                if np.any(unassigned):
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
                    if np.any(mask):
                        pts_query = np.column_stack([z_flat[mask], y_flat[mask], x_flat[mask]])
                        interp_kwargs = dict(method="linear", bounds_error=False, fill_value=None)
                        interp_dens[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), uov_block[dens_i], **interp_kwargs)(pts_query)
                        interp_press[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), uov_block[press_i], **interp_kwargs)(pts_query)
                        interp_vx[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), uov_block[vx_i], **interp_kwargs)(pts_query)
                        interp_vy[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), uov_block[vy_i], **interp_kwargs)(pts_query)
                        interp_vz[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), uov_block[vz_i], **interp_kwargs)(pts_query)
                        if b_block is not None:
                            interp_bx[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), b_block[0], **interp_kwargs)(pts_query)
                            interp_by[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), b_block[1], **interp_kwargs)(pts_query)
                            interp_bz[mask] = RegularGridInterpolator((x3v[b], x2v[b], x1v[b]), b_block[2], **interp_kwargs)(pts_query)
                        unassigned[mask] = False

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

                if np.any(mask_global):
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

            if np.any(unassigned):
                return False, f"{fname} left {np.count_nonzero(unassigned)} shell samples unassigned"

            interp_dens = np.maximum(interp_dens, 0.0)
            interp_press = np.maximum(interp_press, 0.0)

            rho_grid = interp_dens.reshape((n_bins, n_theta, n_phi))
            press_grid = interp_press.reshape((n_bins, n_theta, n_phi))
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

            shell_geom = build_shell_geometry(
                pts_aligned,
                pts_sim,
                coord_dA,
                r_mat,
                t_val,
                metric_params,
                METRIC_KERNEL_SOURCE,
            )
            proper_dA = shell_geom["proper_dA"]

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
                r_mat,
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

        np.savez(
            npz_out,
            time=np.array([t_val]),
            radius=radii_axis,
            theta=theta_vals,
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
            penna_alpha_rey_theta_weighted=np.array([stress_profiles["penna_alpha_rey_theta_weighted"]]),
            penna_alpha_max_theta_weighted=np.array([stress_profiles["penna_alpha_max_theta_weighted"]]),
            penna_alpha_em_exact_theta_weighted=np.array([stress_profiles["penna_alpha_em_exact_theta_weighted"]]),
            penna_alpha_theta_weighted=np.array([stress_profiles["penna_alpha_theta_weighted"]]),
            rmin_used=rmin,
        )

        if delete_source:
            try:
                os.remove(fname)
            except OSError:
                pass

        return True, f"Saved {os.path.basename(npz_out)}"
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
        "penna_alpha_rey_theta_weighted",
        "penna_alpha_max_theta_weighted",
        "penna_alpha_em_exact_theta_weighted",
        "penna_alpha_theta_weighted",
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
        mean_alpha_rey_penna = penna_vertical_average(stress_sums["penna_alpha_rey_theta_weighted"])
        mean_alpha_max_penna = penna_vertical_average(stress_sums["penna_alpha_max_theta_weighted"])
        mean_alpha_em_exact_penna = penna_vertical_average(stress_sums["penna_alpha_em_exact_theta_weighted"])
        mean_alpha_penna = penna_vertical_average(stress_sums["penna_alpha_theta_weighted"])

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
        out_data["alpha_rey_penna"] = mean_alpha_rey_penna
        out_data["alpha_max_penna"] = mean_alpha_max_penna
        out_data["alpha_em_exact_penna"] = mean_alpha_em_exact_penna
        out_data["alpha_penna"] = mean_alpha_penna

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

    selected_files = []
    tasks = []
    for fname in all_files:
        try:
            t_val = get_dump_time(fname)
        except OSError as e:
            print(f"Skipping unreadable file {fname}: {e}")
            continue
        #t_val = get_dump_time(fname)
        if tstart_abs is not None and t_val < tstart_abs:
            continue
        if tend_abs is not None and t_val > tend_abs:
            continue

        selected_files.append(fname)
        npz_out = fname[:-6] + ".npz"
        should_process = True
        if os.path.exists(npz_out):
            if os.path.getmtime(npz_out) > os.path.getmtime(fname):
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
        print("No input dumps found in the requested time range.")
        return

    if tasks:
        failures = []
        if nproc > 1:
            with Pool(nproc) as pool:
                for i, result in enumerate(pool.imap(integrate_and_save, tasks), start=1):
                    ok, message = result
                    if not ok:
                        failures.append(message)
                    sys.stdout.write(f"\r[Proc] {i}/{len(tasks)}...")
                    sys.stdout.flush()
        else:
            for i, task in enumerate(tasks, start=1):
                ok, message = integrate_and_save(task)
                if not ok:
                    failures.append(message)
                sys.stdout.write(f"\r[Proc] {i}/{len(tasks)}...")
                sys.stdout.flush()
        print("\nBatch processing complete.")
        if failures:
            raise RuntimeError("\n\n".join(failures))

    selected_npz_files = [fname[:-6] + ".npz" for fname in selected_files if os.path.exists(fname[:-6] + ".npz")]
    stitch_archives(str(run_path / output_filename), files=selected_npz_files, omega_bin=metric_params["om"])


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

