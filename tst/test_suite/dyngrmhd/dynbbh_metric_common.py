"""Analytic checks for the dynbbh superposed Kerr-Schild background."""

import math
import subprocess
from pathlib import Path

import numpy as np
import athena_read


INPUT_FILE = "inputs/dynbbh_metric.athinput"
SEP = 25.0
Q = 1.0
OMEGA = SEP ** -1.5
M1 = 1.0 / (Q + 1.0)
M2 = 1.0 - M1
FD_STEP = 5.0e-5
CHI1 = 0.93
CHI2 = 0.88
TH1 = math.radians(37.0)
PH1 = math.radians(123.0)
TH2 = math.radians(71.0)
PH2 = math.radians(-41.0)
CHECK_KEYS = (
    "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
    "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
    "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
)
CASES = (
    {
        "name": "far",
        "y": 0.0,
        "z": 0.0,
        "flags": [],
        "rtol": 2.0e-7,
    },
    {
        "name": "strong",
        "y": 5.0,
        "z": 3.0,
        "flags": [
            "mesh/nx1=4", "mesh/x1min=15.0", "mesh/x1max=27.0",
            "mesh/nx2=5", "mesh/x2min=2.5", "mesh/x2max=7.5",
            "mesh/nx3=5", "mesh/x3min=0.5", "mesh/x3max=5.5",
            "meshblock/nx1=4", "meshblock/nx2=5", "meshblock/nx3=5",
            "output1/slice_x2=5.0", "output1/slice_x3=3.0",
        ],
        "rtol": 5.0e-7,
    },
)


def trajectory(t):
    r1 = Q / (1.0 + Q) * SEP
    r2 = -SEP / (1.0 + Q)
    c = math.cos(OMEGA * t)
    s = math.sin(OMEGA * t)
    return {
        "x1": r1 * c,
        "y1": r1 * s,
        "z1": 0.0,
        "x2": r2 * c,
        "y2": r2 * s,
        "z2": 0.0,
        "vx1": -r1 * OMEGA * s,
        "vy1": r1 * OMEGA * c,
        "vz1": 0.0,
        "vx2": -r2 * OMEGA * s,
        "vy2": r2 * OMEGA * c,
        "vz2": 0.0,
        "ax1": CHI1 * math.sin(TH1) * math.cos(PH1),
        "ay1": CHI1 * math.sin(TH1) * math.sin(PH1),
        "az1": CHI1 * math.cos(TH1),
        "ax2": CHI2 * math.sin(TH2) * math.cos(PH2),
        "ay2": CHI2 * math.sin(TH2) * math.sin(PH2),
        "az2": CHI2 * math.cos(TH2),
        "m1": M1,
        "m2": M2,
    }


def boost_jacobian(vx, vy, vz):
    v2 = vx * vx + vy * vy + vz * vz
    gamma = 1.0 / math.sqrt(1.0 - v2)
    if v2 < 1.0e-12:
        q = 0.5 + 0.375 * v2 + 0.3125 * v2 * v2
    else:
        q = (gamma - 1.0) / v2
    return np.array(
        [
            [gamma, -gamma * vx, -gamma * vy, -gamma * vz],
            [-gamma * vx, 1.0 + q * vx * vx, q * vx * vy, q * vx * vz],
            [-gamma * vy, q * vx * vy, 1.0 + q * vy * vy, q * vy * vz],
            [-gamma * vz, q * vx * vz, q * vy * vz, 1.0 + q * vz * vz],
        ],
        dtype=float,
    )


def boosted_position(x, y, z, x0, y0, z0, vx, vy, vz):
    dx = x - x0
    dy = y - y0
    dz = z - z0
    v2 = vx * vx + vy * vy + vz * vz
    gamma = 1.0 / math.sqrt(1.0 - v2)
    if v2 < 1.0e-12:
        q = 0.5 + 0.375 * v2 + 0.3125 * v2 * v2
    else:
        q = (gamma - 1.0) / v2
    vd = vx * dx + vy * dy + vz * dz
    return dx + q * vx * vd, dy + q * vy * vd, dz + q * vz * vd


def kerr_schild_perturbation(x, y, z, ax, ay, az, mass):
    rt2 = math.sqrt(2.0)
    irt2 = 1.0 / rt2
    a2 = ax * ax + ay * ay + az * az
    x2 = x * x + y * y + z * z
    ad = ax * x + ay * y + az * z
    term = x2 - a2
    rho2 = term + math.sqrt(4.0 * ad * ad + term * term)
    rho = math.sqrt(rho2)
    fac = irt2 * rho2 * rho * mass / (ad * ad + 0.25 * rho2 * rho2)
    den = a2 + 0.5 * rho2
    ell = np.array(
        [
            1.0,
            (y * az - z * ay + rt2 * ad * ax / rho + rho * x * irt2) / den,
            (-x * az + z * ax + rt2 * ad * ay / rho + rho * y * irt2) / den,
            (x * ay - y * ax + rt2 * ad * az / rho + rho * z * irt2) / den,
        ],
        dtype=float,
    )
    return fac * np.outer(ell, ell)


def superposed_metric(t, x, y, z):
    tr = trajectory(t)
    g = np.diag([-1.0, 1.0, 1.0, 1.0])
    for suffix in ("1", "2"):
        xb, yb, zb = boosted_position(
            x,
            y,
            z,
            tr[f"x{suffix}"],
            tr[f"y{suffix}"],
            tr[f"z{suffix}"],
            tr[f"vx{suffix}"],
            tr[f"vy{suffix}"],
            tr[f"vz{suffix}"],
        )
        mass = tr[f"m{suffix}"]
        ks = kerr_schild_perturbation(
            xb,
            yb,
            zb,
            tr[f"ax{suffix}"] * mass,
            tr[f"ay{suffix}"] * mass,
            tr[f"az{suffix}"] * mass,
            mass,
        )
        jac = boost_jacobian(tr[f"vx{suffix}"], tr[f"vy{suffix}"], tr[f"vz{suffix}"])
        g += jac.T @ ks @ jac
    return g


def metric_derivatives(t, x, y, z):
    h = FD_STEP
    return (
        (superposed_metric(t + h, x, y, z) - superposed_metric(t - h, x, y, z)) / (2.0 * h),
        (superposed_metric(t, x + h, y, z) - superposed_metric(t, x - h, y, z)) / (2.0 * h),
        (superposed_metric(t, x, y + h, z) - superposed_metric(t, x, y - h, z)) / (2.0 * h),
        (superposed_metric(t, x, y, z + h) - superposed_metric(t, x, y, z - h)) / (2.0 * h),
    )


def spatial_det(g):
    return (
        -g[0, 2] * g[0, 2] * g[1, 1]
        + 2.0 * g[0, 1] * g[0, 2] * g[1, 2]
        - g[1, 2] * g[1, 2] * g[0, 0]
        - g[0, 1] * g[0, 1] * g[2, 2]
        + g[0, 0] * g[1, 1] * g[2, 2]
    )


def adm_from_metric(t, x, y, z):
    gcov = superposed_metric(t, x, y, z)
    dtg, dxg, dyg, dzg = metric_derivatives(t, x, y, z)
    gamma = gcov[1:4, 1:4]
    invgamma = np.linalg.inv(gamma)
    beta_down = gcov[0, 1:4]
    beta = invgamma @ beta_down
    alpha = math.sqrt(float(beta_down @ beta - gcov[0, 0]))

    bx, by, bz = beta
    dbetadownxx, dbetadownyx, dbetadownzx = dxg[0, 1], dxg[0, 2], dxg[0, 3]
    dbetadownxy, dbetadownyy, dbetadownzy = dyg[0, 1], dyg[0, 2], dyg[0, 3]
    dbetadownxz, dbetadownyz, dbetadownzz = dzg[0, 1], dzg[0, 2], dzg[0, 3]
    dtgxx, dtgxy, dtgxz = dtg[1, 1], dtg[1, 2], dtg[1, 3]
    dtgyy, dtgyz, dtgzz = dtg[2, 2], dtg[2, 3], dtg[3, 3]
    dgxxx, dgxyx, dgxzx = dxg[1, 1], dxg[1, 2], dxg[1, 3]
    dgyyx, dgyzx, dgzzx = dxg[2, 2], dxg[2, 3], dxg[3, 3]
    dgxxy, dgxyy, dgxzy = dyg[1, 1], dyg[1, 2], dyg[1, 3]
    dgyyy, dgyzy, dgzzy = dyg[2, 2], dyg[2, 3], dyg[3, 3]
    dgxxz, dgxyz, dgxzz = dzg[1, 1], dzg[1, 2], dzg[1, 3]
    dgyyz, dgyzz, dgzzz = dzg[2, 2], dzg[2, 3], dzg[3, 3]
    kdd = np.zeros((3, 3))
    kdd[0, 0] = -(-2 * dbetadownxx - bx * dgxxx - by * dgxxy - bz * dgxxz
                  + 2 * (bx * dgxxx + by * dgxyx + bz * dgxzx) + dtgxx) / (2 * alpha)
    kdd[0, 1] = -(-dbetadownxy - dbetadownyx + bx * dgxxy - bz * dgxyz
                  + bz * dgxzy + by * dgyyx + bz * dgyzx + dtgxy) / (2 * alpha)
    kdd[0, 2] = -(-dbetadownxz - dbetadownzx + bx * dgxxz + by * dgxyz
                  - by * dgxzy + by * dgyzx + bz * dgzzx + dtgxz) / (2 * alpha)
    kdd[1, 1] = -(-2 * dbetadownyy - bx * dgyyx - by * dgyyy - bz * dgyyz
                  + 2 * (bx * dgxyy + by * dgyyy + bz * dgyzy) + dtgyy) / (2 * alpha)
    kdd[1, 2] = -(-dbetadownyz - dbetadownzy + bx * dgxyz + bx * dgxzy
                  + by * dgyyz - bx * dgyzx + bz * dgzzy + dtgyz) / (2 * alpha)
    kdd[2, 2] = -(-2 * dbetadownzz - bx * dgzzx - by * dgzzy - bz * dgzzz
                  + 2 * (bx * dgxzz + by * dgyzz + bz * dgzzz) + dtgzz) / (2 * alpha)
    kdd[1, 0] = kdd[0, 1]
    kdd[2, 0] = kdd[0, 2]
    kdd[2, 1] = kdd[1, 2]

    return {
        "adm_gxx": gamma[0, 0],
        "adm_gxy": gamma[0, 1],
        "adm_gxz": gamma[0, 2],
        "adm_gyy": gamma[1, 1],
        "adm_gyz": gamma[1, 2],
        "adm_gzz": gamma[2, 2],
        "adm_Kxx": kdd[0, 0],
        "adm_Kxy": kdd[0, 1],
        "adm_Kxz": kdd[0, 2],
        "adm_Kyy": kdd[1, 1],
        "adm_Kyz": kdd[1, 2],
        "adm_Kzz": kdd[2, 2],
        "adm_alpha": alpha,
        "adm_betax": beta[0],
        "adm_betay": beta[1],
        "adm_betaz": beta[2],
    }


def write_trajectory_table(path):
    rows = []
    for t in (-1.0e-3, 0.0, 1.0e-3):
        tr = trajectory(t)
        rows.append([
            t, tr["m1"], tr["m2"],
            tr["x1"], tr["y1"], tr["z1"], tr["x2"], tr["y2"], tr["z2"],
            tr["ax1"], tr["ay1"], tr["az1"], tr["ax2"], tr["ay2"], tr["az2"],
            tr["vx1"], tr["vy1"], tr["vz1"], tr["vx2"], tr["vy2"], tr["vz2"],
        ])
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(" ".join(f"{x:.17e}" for x in row) + "\n")


def run_case(method, use_table=False, fd_step=FD_STEP, case=CASES[0]):
    label = f"{case['name']}_{method}_{'table' if use_table else 'analytic'}_{fd_step:.0e}"
    basename = f"dynbbh_metric_{label}"
    flags = [
        "./athena",
        "-i",
        INPUT_FILE,
        f"job/basename={basename}",
        f"problem/metric_derivative={method}",
        "time/nlim=0",
        "time/tlim=0.0",
        f"problem/metric_fd_step={fd_step:.17e}",
    ] + case["flags"]
    if use_table:
        table_path = Path(f"{basename}.traj")
        write_trajectory_table(table_path)
        flags += ["problem/use_traj_table=true", f"problem/traj_file={table_path.resolve()}"]
    subprocess.check_call(flags)
    return athena_read.tab(Path("tab") / f"{basename}.adm.00000.tab")


def check_against_reference(data, method, case=CASES[0]):
    for n, x in enumerate(data["x1v"]):
        ref = adm_from_metric(0.0, float(x), case["y"], case["z"])
        for key, val in ref.items():
            atol = 2.0e-8 if key.startswith("adm_g") or key in ("adm_alpha", "adm_betax", "adm_betay", "adm_betaz") else 4.0e-7
            if not np.isclose(data[key][n], val, rtol=case["rtol"], atol=atol):
                raise AssertionError(
                    f"{method} {key} mismatch at x={x}: code={data[key][n]} ref={val}"
                )


def compare_outputs(lhs, rhs, label):
    for key in CHECK_KEYS:
        diff = np.max(np.abs(lhs[key] - rhs[key]))
        scale = max(1.0, np.max(np.abs(lhs[key])), np.max(np.abs(rhs[key])))
        if diff > 5.0e-7 * scale:
            raise AssertionError(f"{label} {key} branch mismatch: max abs diff={diff}")


def run_and_check(method, use_table=False, case=CASES[0]):
    data = run_case(method, use_table=use_table, case=case)
    check_against_reference(data, f"{case['name']} {method} {'table' if use_table else 'analytic'}", case)
    return data


def run_stress_suite():
    for case in CASES:
        for use_table in (False, True):
            ad = run_and_check("ad", use_table=use_table, case=case)
            subprocess.call("rm -rf tab *.dat *.hst", shell=True)
            fd = run_and_check("finite_difference", use_table=use_table, case=case)
            compare_outputs(ad, fd, f"{case['name']} {'table' if use_table else 'analytic'} ad-vs-fd")
            subprocess.call("rm -rf tab *.dat *.hst", shell=True)


def run_convergence_check():
    ad = run_case("ad", use_table=False)
    subprocess.call("rm -rf tab *.dat *.hst", shell=True)
    errors = []
    for step in (1.0e-4, 5.0e-5, 2.5e-5):
        fd = run_case("finite_difference", use_table=False, fd_step=step)
        err = max(np.max(np.abs(fd[key] - ad[key])) for key in CHECK_KEYS)
        errors.append(err)
        subprocess.call("rm -rf tab *.dat *.hst", shell=True)
    if max(errors) < 1.0e-10:
        return
    if not (errors[1] < 0.55 * errors[0] and errors[2] < 0.55 * errors[1]):
        raise AssertionError(f"finite-difference derivative did not converge: {errors}")
