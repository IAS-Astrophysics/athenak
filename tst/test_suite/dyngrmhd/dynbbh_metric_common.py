"""Independent regression checks for the legacy dynbbh metric and derivatives."""

import math
import re
import struct
import subprocess
from pathlib import Path

import athena_read
import numpy as np


INPUT_FILE = "inputs/dynbbh_metric.athinput"
FD_STEP = 5.0e-5
KEYS = (
    "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
    "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
    "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
)
CASES = (
    {"name": "torus", "y": 0.0, "z": 0.0, "flags": [], "ramp": False},
    {
        "name": "strong", "y": 5.0, "z": 3.0,
        "ramp": True,
        "flags": [
            "mesh/nx1=4", "mesh/x1min=15.0", "mesh/x1max=27.0",
            "mesh/nx2=5", "mesh/x2min=2.5", "mesh/x2max=7.5",
            "mesh/nx3=5", "mesh/x3min=0.5", "mesh/x3max=5.5",
            "meshblock/nx1=4", "meshblock/nx2=5", "meshblock/nx3=5",
            "output1/slice_x2=5.0", "output1/slice_x3=3.0",
        ],
    },
)


def trajectory(t, ramp=False):
    """Independent analytic binary state using dimensionless spins."""
    sep, q = 25.0, 2.0
    omega = sep**-1.5
    r1, r2 = q/(1.0 + q)*sep, -sep/(1.0 + q)
    c, s = math.cos(omega*t), math.sin(omega*t)
    a1, a2 = 0.93, 0.88
    th1, ph1 = math.radians(37.0), math.radians(123.0)
    th2, ph2 = math.radians(71.0), math.radians(-41.0)
    spin_factor = 1.0
    if ramp:
        u = min(max((t + 50.0)/100.0, 0.0), 1.0)
        spin_factor = u*u*(3.0 - 2.0*u)
    return {
        "p1": np.array([r1*c, r1*s, 0.0]),
        "p2": np.array([r2*c, r2*s, 0.0]),
        "v1": np.array([-r1*omega*s, r1*omega*c, 0.0]),
        "v2": np.array([-r2*omega*s, r2*omega*c, 0.0]),
        "a1": spin_factor*a1*np.array([math.sin(th1)*math.cos(ph1),
                                        math.sin(th1)*math.sin(ph1), math.cos(th1)]),
        "a2": spin_factor*a2*np.array([math.sin(th2)*math.cos(ph2),
                                        math.sin(th2)*math.sin(ph2), math.cos(th2)]),
        "m1": 1.0/(q + 1.0), "m2": q/(q + 1.0),
    }


def boost_q(v2):
    gamma = 1.0/math.sqrt(1.0 - v2)
    if v2 < 1.0e-12:
        return gamma, 0.5 + 0.375*v2 + 0.3125*v2*v2
    return gamma, (gamma - 1.0)/v2


def boosted_position(point, center, velocity):
    delta = point - center
    _, q = boost_q(float(velocity @ velocity))
    return delta + q*velocity*float(velocity @ delta)


def boost_jacobian(velocity):
    gamma, q = boost_q(float(velocity @ velocity))
    jac = np.eye(4)
    jac[0, 0] = gamma
    jac[0, 1:] = -gamma*velocity
    jac[1:, 0] = -gamma*velocity
    jac[1:, 1:] += q*np.outer(velocity, velocity)
    return jac


def ks_perturbation(point, spin, mass):
    radius2 = float(point @ point)
    spin2 = float(spin @ spin)
    adotx = float(spin @ point)
    term = radius2 - spin2
    rho2 = term + math.sqrt(4.0*adotx*adotx + term*term)
    rho = math.sqrt(rho2)
    fac = rho2*rho*mass/math.sqrt(2.0)/(adotx*adotx + 0.25*rho2*rho2)
    den = spin2 + 0.5*rho2
    ell = np.cross(point, spin) + math.sqrt(2.0)*adotx*spin/rho
    ell += rho*point/math.sqrt(2.0)
    null = np.concatenate(([1.0], ell/den))
    return fac*np.outer(null, null)


def metric(t, x, y, z, ramp=False):
    tr = trajectory(t, ramp=ramp)
    point = np.array([x, y, z])
    gcov = np.diag([-1.0, 1.0, 1.0, 1.0])
    for suffix in ("1", "2"):
        velocity = tr[f"v{suffix}"]
        local = boosted_position(point, tr[f"p{suffix}"], velocity)
        mass = tr[f"m{suffix}"]
        ks = ks_perturbation(local, mass*tr[f"a{suffix}"], mass)
        jac = boost_jacobian(velocity)
        gcov += jac.T @ ks @ jac
    return gcov


def adm_reference(t, x, y, z, ramp=False):
    h = FD_STEP
    gcov = metric(t, x, y, z, ramp=ramp)
    deriv = (
        (metric(t+h, x, y, z, ramp) - metric(t-h, x, y, z, ramp))/(2*h),
        (metric(t, x+h, y, z, ramp) - metric(t, x-h, y, z, ramp))/(2*h),
        (metric(t, x, y+h, z, ramp) - metric(t, x, y-h, z, ramp))/(2*h),
        (metric(t, x, y, z+h, ramp) - metric(t, x, y, z-h, ramp))/(2*h),
    )
    gamma = gcov[1:, 1:]
    invgamma = np.linalg.inv(gamma)
    beta_down = gcov[0, 1:]
    beta = invgamma @ beta_down
    alpha = math.sqrt(float(beta_down @ beta - gcov[0, 0]))
    kdd = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            connection_beta = 0.0
            for k in range(3):
                gamma_lower = 0.5*(deriv[i+1][j+1, k+1] +
                                   deriv[j+1][i+1, k+1] -
                                   deriv[k+1][i+1, j+1])
                connection_beta += gamma_lower*beta[k]
            kdd[i, j] = (-deriv[0][i+1, j+1] + deriv[i+1][0, j+1] +
                         deriv[j+1][0, i+1] - 2.0*connection_beta)/(2.0*alpha)
    return {
        "adm_gxx": gamma[0, 0], "adm_gxy": gamma[0, 1],
        "adm_gxz": gamma[0, 2], "adm_gyy": gamma[1, 1],
        "adm_gyz": gamma[1, 2], "adm_gzz": gamma[2, 2],
        "adm_Kxx": kdd[0, 0], "adm_Kxy": kdd[0, 1],
        "adm_Kxz": kdd[0, 2], "adm_Kyy": kdd[1, 1],
        "adm_Kyz": kdd[1, 2], "adm_Kzz": kdd[2, 2],
        "adm_alpha": alpha, "adm_betax": beta[0],
        "adm_betay": beta[1], "adm_betaz": beta[2],
    }


def write_trajectory_table(path, case):
    rows = []
    for time in (-1.0e-3, 0.0, 1.0e-3):
        tr = trajectory(time, ramp=case["ramp"])
        rows.append([time, tr["m1"], tr["m2"], *tr["p1"], *tr["p2"],
                     *tr["a1"], *tr["a2"], *tr["v1"], *tr["v2"]])
    path.write_text("\n".join(" ".join(f"{v:.17e}" for v in row)
                              for row in rows) + "\n", encoding="utf-8")


def write_trajectory_rows(path, rows):
    path.write_text("\n".join(" ".join(f"{v:.17e}" for v in row)
                              for row in rows) + "\n", encoding="utf-8")


def trajectory_row(time, m1, m2, p1, p2, chi1, chi2, v1=(0.0, 0.0, 0.0),
                   v2=(0.0, 0.0, 0.0)):
    return [time, m1, m2, *p1, *p2, *chi1, *chi2, *v1, *v2]


def run_custom_table(method, basename, rows, step=FD_STEP, flags=()):
    table = Path(f"{basename}.traj").resolve()
    write_trajectory_rows(table, rows)
    args = ["./athena", "-i", INPUT_FILE, f"job/basename={basename}",
            f"problem/metric_derivative={method}",
            f"problem/metric_fd_step={step:.17e}",
            "problem/use_traj_table=true", f"problem/traj_file={table}",
            *flags]
    subprocess.check_call(args)
    return athena_read.tab(Path("tab") / f"{basename}.adm.00000.tab")


def run_case(method, case, step=FD_STEP, use_table=False):
    mode = "table" if use_table else "analytic"
    basename = f"dynbbh_metric_{case['name']}_{mode}_{method}_{step:.0e}"
    args = ["./athena", "-i", INPUT_FILE, f"job/basename={basename}",
            f"problem/metric_derivative={method}",
            f"problem/metric_fd_step={step:.17e}"] + case["flags"]
    if use_table:
        table = Path(f"{basename}.traj").resolve()
        write_trajectory_table(table, case)
        args += ["problem/use_traj_table=true", f"problem/traj_file={table}"]
    elif case["ramp"]:
        args += ["problem/spin_ramp=true", "problem/spin_ramp_start_time=-50.0",
                 "problem/spin_ramp_timescale=100.0"]
    subprocess.check_call(args)
    return athena_read.tab(Path("tab") / f"{basename}.adm.00000.tab")


def check_reference(data, case):
    for n, x in enumerate(data["x1v"]):
        reference = adm_reference(0.0, float(x), case["y"], case["z"], case["ramp"])
        for key, expected in reference.items():
            atol = 3.0e-8 if not key.startswith("adm_K") else 5.0e-7
            assert np.isclose(data[key][n], expected, rtol=8.0e-7, atol=atol), (
                key, x, data[key][n], expected)


def run_trajectory_edge_checks():
    flags = CASES[1]["flags"]
    p1, p2 = (8.0, 0.0, 0.0), (-8.0, 0.0, 0.0)
    chi1 = (0.25, -0.15, 0.35)
    chi2 = (-0.20, 0.30, 0.10)

    # An interior knot must use the segment on its right regardless of the
    # interpolation cache's previous segment.  Moving that knot infinitesimally
    # to the past leaves the state and its right derivative at t=0 unchanged.
    knot_rows = [
        trajectory_row(-1.0, 0.20, 0.80, p1, p2,
                       (0.05, -0.05, 0.10), (-0.05, 0.10, 0.05)),
        trajectory_row(0.0, 0.40, 0.60, p1, p2, chi1, chi2),
        trajectory_row(1.0, 0.40, 0.60, p1, p2, chi1, chi2),
    ]
    shifted_rows = [knot_rows[0],
                    trajectory_row(-1.0e-12, 0.40, 0.60, p1, p2, chi1, chi2),
                    knot_rows[2]]
    at_knot = run_custom_table("ad", "dynbbh_table_knot", knot_rows,
                               flags=flags)
    right_limit = run_custom_table("ad", "dynbbh_table_knot_right",
                                   shifted_rows, flags=flags)
    for key in KEYS:
        np.testing.assert_allclose(at_knot[key], right_limit[key],
                                   rtol=2.0e-12, atol=2.0e-12)

    # Exercise an asymmetrically clipped FD stencil near the first table time.
    # Rapid (but valid) mass evolution makes the first-order bias of a plain
    # secant visible while the nonuniform three-point stencil remains close to AD.
    boundary_rows = [
        trajectory_row(-2.0e-5, 0.25, 0.75, p1, p2, chi1, chi2),
        trajectory_row(1.0e-3, 0.75, 0.25, p1, p2, chi1, chi2),
    ]
    boundary_ad = run_custom_table("ad", "dynbbh_table_boundary_ad",
                                   boundary_rows, flags=flags)
    boundary_fd = run_custom_table("finite_difference",
                                   "dynbbh_table_boundary_fd", boundary_rows,
                                   flags=flags)
    for key in KEYS:
        np.testing.assert_allclose(boundary_fd[key], boundary_ad[key],
                                   rtol=5.0e-6, atol=2.0e-5)

    # Endpoint velocities alone do not bound a cubic Hermite segment's
    # interior speed.  Reject an interpolated superluminal boost explicitly.
    bad_rows = [
        trajectory_row(-1.0, 0.40, 0.60, (-10.0, 0.0, 0.0), p2,
                       chi1, chi2),
        trajectory_row(1.0, 0.40, 0.60, (10.0, 0.0, 0.0), p2,
                       chi1, chi2),
    ]
    basename = "dynbbh_table_superluminal"
    table = Path(f"{basename}.traj").resolve()
    write_trajectory_rows(table, bad_rows)
    result = subprocess.run(
        ["./athena", "-i", INPUT_FILE, f"job/basename={basename}",
         "problem/metric_derivative=ad", "problem/use_traj_table=true",
         f"problem/traj_file={table}", *flags],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        check=False)
    assert result.returncode != 0, result.stdout
    assert "interpolated trajectory velocity must be finite and subluminal" in result.stdout


def run_regression_suite():
    for case in CASES:
        for use_table in (False, True):
            fd = run_case("finite_difference", case, use_table=use_table)
            ad = run_case("ad", case, use_table=use_table)
            check_reference(fd, case)
            check_reference(ad, case)
            for key in KEYS:
                np.testing.assert_allclose(ad[key], fd[key], rtol=8.0e-7,
                                           atol=5.0e-9)
    run_trajectory_edge_checks()


def run_fd_convergence():
    case = CASES[0]
    ad = run_case("ad", case)
    errors = []
    # Use deliberately coarse steps so the expected second-order truncation
    # error remains above double-precision cancellation in this weak field.
    for step in (2.0e-2, 1.0e-2, 5.0e-3):
        fd = run_case("finite_difference", case, step)
        errors.append(max(float(np.max(np.abs(fd[k] - ad[k]))) for k in KEYS))
    assert errors[1] < 0.4*errors[0], errors
    assert errors[2] < 0.4*errors[1], errors


def run_surface_check():
    radius = 100.0
    basename = "dynbbh_flux_surface_regression"
    args = [
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        "problem/user_hist=true", f"problem/flux_rsurf_inner={radius}",
        f"problem/flux_rsurf_outer={radius}", "problem/flux_dr_surf=10.0",
        "mesh/nx1=12", "mesh/x1min=-150", "mesh/x1max=150",
        "mesh/nx2=12", "mesh/x2min=-150", "mesh/x2max=150",
        "mesh/nx3=12", "mesh/x3min=-150", "mesh/x3max=150",
        "meshblock/nx1=12", "meshblock/nx2=12", "meshblock/nx3=12",
    ]
    subprocess.check_call(args)
    history = Path(f"{basename}.user.hst")
    header = [
        line for line in history.read_text(encoding="utf-8").splitlines()
        if line.startswith("#  [1]=")
    ][-1]
    labels = [token.split("=", 1)[1] for token in header.split() if "=" in token]
    expected = (
        "time", "dt", "mdot_r100", "edot_f_r100", "edot_em_r100",
        "pxdot_f_r100", "pydot_f_r100", "pzdot_f_r100",
        "pxdot_em_r100", "pydot_em_r100", "pzdot_em_r100",
        "lxdot_f_r100", "lydot_f_r100", "lzdot_f_r100",
        "lxdot_em_r100", "lydot_em_r100", "lzdot_em_r100",
        "phiB_r100", "area_r100",
    )
    assert tuple(labels) == tuple(label[:10] for label in expected), labels
    data_line = [
        line for line in history.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ][-1]
    values = np.fromstring(data_line, sep=" ")
    assert values.shape == (19,) and np.all(np.isfinite(values)), values
    mdot, area = values[2], values[-1]
    flat_area = 4.0*math.pi*radius*radius
    assert abs(area/flat_area - 1.0) < 0.02, (area, flat_area)
    # At r=100 the moving metric's shift leaves an O(M/r) coordinate flux;
    # its closed-sphere integral must nevertheless remain correspondingly small.
    assert abs(mdot) < 0.03*area*1.0e-10, (mdot, area)
    # The regression atmosphere is unmagnetized, so all EM and magnetic-flux
    # diagnostics must vanish exactly.  Angular and linear fluid fluxes remain
    # finite; their closed-surface residuals are covered by the area/flux check.
    np.testing.assert_array_equal(values[[4, 8, 9, 10, 14, 15, 16, 17]], 0.0)


def run_volume_diagnostics_check():
    """Exercise both new derived-output kernels on the live dynbbh metric."""
    results = {}
    for variable in ("angular_momentum", "torque"):
        basename = f"dynbbh_{variable}_regression"
        subprocess.check_call([
            "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
            f"output1/variable={variable}",
        ])
        table = Path("tab") / f"{basename}.{variable}.00000.tab"
        results[variable] = np.loadtxt(table, comments="#", ndmin=2)

    angular = results["angular_momentum"][:, 3:]
    torque = results["torque"][:, 3:]
    assert angular.shape[1] == 6 and torque.shape[1] == 3
    assert np.all(np.isfinite(angular)) and np.all(np.isfinite(torque))
    # The initialized atmosphere is at rest and unmagnetized.
    np.testing.assert_array_equal(angular, 0.0)
    # The nonaxisymmetric moving-binary metric supplies a small but resolved
    # gravitational torque even for the pressure-floor atmosphere.
    assert np.max(np.abs(torque)) > 1.0e-16, torque


def run_diagnostic_failure_checks():
    """Require a clear failure for unsupported multicomponent PDF axes."""
    result = subprocess.run([
        "./athena", "-i", INPUT_FILE, "job/basename=dynbbh_bad_diagnostic_pdf",
        "output1/file_type=pdf", "output1/variable=angular_momentum",
    ], check=False, capture_output=True, text=True)
    message = result.stdout + result.stderr
    assert result.returncode != 0
    assert "multicomponent angular_momentum and torque" in message, message


def _excision_mesh_args():
    return [
        "mesh/nx1=24", "mesh/x1min=-24", "mesh/x1max=24",
        "mesh/nx2=8", "mesh/x2min=-8", "mesh/x2max=8",
        "mesh/nx3=8", "mesh/x3min=-8", "mesh/x3max=8",
        "meshblock/nx1=24", "meshblock/nx2=8", "meshblock/nx3=8",
        "time/nlim=1", "time/tlim=1", "output1/dcycle=1", "output1/dt=1",
        "coord/excision_scheme=puncture", "coord/excise_1_rad=5",
        "coord/excise_2_rad=5",
    ]


def _write_time_dependent_spin_table(path):
    rows = [
        trajectory_row(0.0, 1.0, 0.1, (0.0, 0.0, 0.0),
                       (20.0, 0.0, 0.0), (0.95, 0.0, 0.0),
                       (0.0, 0.0, 0.0)),
        trajectory_row(0.1, 1.0, 0.1, (0.0, 0.0, 0.0),
                       (20.0, 0.0, 0.0), (0.0, 0.0, 0.2),
                       (0.0, 0.0, 0.0)),
    ]
    write_trajectory_rows(path, rows)


def run_excision_checks():
    """Check two-hole mask geometry, unresolved sinks, and CT divB control."""
    common = _excision_mesh_args()

    # With RK1 the only primitive recovery in the cycle must see the new
    # stage-time mask.  This table both reduces and rotates chi, expanding the
    # x-axis horizon across the cell centered at x=1.75.
    basename = "dynbbh_excision_time_dependent_spin"
    table = Path(f"{basename}.traj").resolve()
    _write_time_dependent_spin_table(table)
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        "problem/use_traj_table=true", f"problem/traj_file={table}",
        "problem/metric_derivative=ad", "time/integrator=rk1",
        "time/nlim=1", "time/tlim=0.1", "output1/dcycle=1",
        "output1/dt=1", "output1/variable=mhd_w_d",
        "mesh/nx1=16", "mesh/x1min=-4", "mesh/x1max=4",
        "mesh/nx2=1", "mesh/x2min=-0.5", "mesh/x2max=0.5",
        "mesh/nx3=1", "mesh/x3min=-0.5", "mesh/x3max=0.5",
        "meshblock/nx1=16", "meshblock/nx2=1", "meshblock/nx3=1",
        "coord/excision_scheme=puncture", "coord/excise_to_horizon=true",
    ])
    initial = np.loadtxt(
        Path("tab") / f"{basename}.mhd_w_d.00000.tab", comments="#", ndmin=2)
    evolved = np.loadtxt(
        Path("tab") / f"{basename}.mhd_w_d.00001.tab", comments="#", ndmin=2)
    probe = np.argmin(np.abs(initial[:, 2] - 1.75))
    assert np.isclose(initial[probe, 2], 1.75)
    assert initial[probe, 3] < 5.0e-10
    assert evolved[probe, 3] > 5.0e-9

    smooth = [
        "coord/smooth_excision=true",
        "coord/smooth_excision_puncture_width_fraction=1",
        "coord/puncture_flux_excision_radius_factor=1.5",
        "coord/smooth_excision_b_damping=true",
        "coord/smooth_excision_b_damping_eta=0.1",
        "problem/test_bz_gradient=1e-8",
    ]
    basename = "dynbbh_excision_geometry"
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        "output1/variable=mhd_w_d", *common, *smooth,
    ])
    initial = np.loadtxt(
        Path("tab") / f"{basename}.mhd_w_d.00000.tab", comments="#", ndmin=2)
    evolved = np.loadtxt(
        Path("tab") / f"{basename}.mhd_w_d.00001.tab", comments="#", ndmin=2)
    for data, time in ((initial, 0.0), (evolved, 0.8)):
        tr = trajectory(time)
        x, density = data[:, 2], data[:, 3]
        for center in (tr["p1"][0], tr["p2"][0]):
            assert density[np.argmin(np.abs(x-center))] > 5.0e-9
        assert density[np.argmin(np.abs(x))] < 5.0e-10
    transition = initial[:, 3]
    assert np.count_nonzero((transition > 2.0e-10) &
                            (transition < 8.0e-9)) >= 4

    basename = "dynbbh_excision_divb"
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        "output1/variable=mhd_divb", *common, *smooth,
    ])
    divb = np.loadtxt(
        Path("tab") / f"{basename}.mhd_divb.00001.tab", comments="#", ndmin=2)
    assert np.max(np.abs(divb[:, 3])) < 1.0e-18

    basename = "dynbbh_unresolved_sink"
    sink_args = [
        arg for arg in common
        if not arg.startswith(("coord/excise_1_rad", "coord/excise_2_rad",
                               "time/tlim"))
    ]
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        "output1/variable=mhd_w_d", *sink_args,
        "coord/excise_1_rad=0.2", "coord/excise_2_rad=0.2",
        "time/tlim=0.1", "problem/dfloor=1e-8", "problem/pfloor=1e-12",
        "problem/unresolved_sink=true", "problem/sink_timescale=0.01",
        "problem/sink_density_floor=1e-10",
        "problem/sink_pressure_floor=1e-18",
        "problem/sink_cells_per_radius=2",
        "problem/sink_resolved_cells_across_horizon=8",
    ])
    sink = np.loadtxt(
        Path("tab") / f"{basename}.mhd_w_d.00001.tab", comments="#", ndmin=2)
    x, density = sink[:, 2], sink[:, 3]
    for center in (trajectory(0.1)["p1"][0], trajectory(0.1)["p2"][0]):
        assert density[np.argmin(np.abs(x-center))] < 5.0e-10
    assert density[np.argmin(np.abs(x))] > 5.0e-9

    failure = subprocess.run([
        "./athena", "-i", INPUT_FILE, "job/basename=dynbbh_unresolved_fatal",
        *_excision_mesh_args(), "coord/require_resolved_horizon=true",
    ], check=False, capture_output=True, text=True)
    assert failure.returncode != 0
    assert "puncture excision is under-resolved" in failure.stdout + failure.stderr

    # A radius transition uses absolute simulation time.  Splitting the same
    # two-cycle calculation at a restart must therefore be bitwise identical.
    restart_common = [
        arg for arg in common
        if not arg.startswith(("time/nlim", "time/tlim", "output1/"))
    ]
    shrink = [
        "coord/excise_shrink_to_horizon=true",
        "coord/excise_shrink_timescale=1",
        "coord/excise_shrink_start_time=0",
    ]
    basename = "dynbbh_excision_restart"
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        *restart_common, *shrink, "time/nlim=1", "time/tlim=0.8",
        "output1/file_type=rst", "output1/dcycle=1",
    ])
    restart_file = Path("rst") / f"{basename}.00001.rst"
    assert restart_file.is_file()
    continued = f"{basename}_continued"
    subprocess.check_call([
        "./athena", "-r", restart_file, f"job/basename={continued}",
        "time/nlim=2", "time/tlim=1.6", "output1/file_type=tab",
        "output1/variable=mhd_w_d", "output1/dcycle=1",
    ])
    uninterrupted = f"{basename}_uninterrupted"
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={uninterrupted}",
        *restart_common, *shrink, "time/nlim=2", "time/tlim=1.6",
        "output1/file_type=tab", "output1/variable=mhd_w_d",
        "output1/dcycle=1",
    ])
    continued_data = np.loadtxt(
        Path("tab") / f"{continued}.mhd_w_d.00002.tab",
        comments="#", ndmin=2)
    uninterrupted_data = np.loadtxt(
        Path("tab") / f"{uninterrupted}.mhd_w_d.00002.tab",
        comments="#", ndmin=2)
    np.testing.assert_array_equal(continued_data, uninterrupted_data)


def _read_binary_mesh(path):
    """Return (cycle, [(level, bounds), ...]) from an Athena binary dump."""
    with path.open("rb") as stream:
        metadata = {}
        while True:
            line = stream.readline()
            assert line, f"truncated binary preheader in {path}"
            text = line.decode("ascii")
            match = re.match(r"\s*([^=]+)=(.*)", text)
            if match:
                metadata[match.group(1).strip()] = match.group(2).strip()
            if text.startswith("  header offset="):
                stream.read(int(metadata["header offset"]))
                break
        nvar = int(metadata["number of variables"])
        location_size = int(metadata["size of location"])
        real_format = "d" if location_size == 8 else "f"
        blocks = []
        while True:
            raw_indices = stream.read(10*4)
            if not raw_indices:
                break
            assert len(raw_indices) == 10*4
            indices = struct.unpack("=10i", raw_indices)
            raw_bounds = stream.read(6*location_size)
            bounds = struct.unpack("=" + 6*real_format, raw_bounds)
            cells = ((indices[1] - indices[0] + 1) *
                     (indices[3] - indices[2] + 1) *
                     (indices[5] - indices[4] + 1))
            stream.seek(cells*nvar*4, 1)
            blocks.append((indices[9], bounds))
    return int(metadata["cycle"]), blocks


def _latest_binary_mesh(basename):
    dumps = list(Path("bin").glob(f"{basename}.mhd_w.*.bin"))
    assert dumps, basename
    return max((_read_binary_mesh(path) for path in dumps), key=lambda item: item[0])


def _level_at(blocks, point):
    matches = [level for level, bounds in blocks
               if bounds[0] <= point[0] <= bounds[1] and
               bounds[2] <= point[1] <= bounds[3] and
               bounds[4] <= point[2] <= bounds[5]]
    assert matches, point
    return max(matches)


def _amr_args():
    return [
        "mesh/nx1=16", "mesh/x1min=-4", "mesh/x1max=4",
        "mesh/nx2=1", "mesh/x2min=-0.5", "mesh/x2max=0.5",
        "mesh/nx3=1", "mesh/x3min=-0.5", "mesh/x3max=0.5",
        "meshblock/nx1=4", "meshblock/nx2=1", "meshblock/nx3=1",
        "mesh_refinement/refinement=adaptive",
        "mesh_refinement/num_levels=3",
        "mesh_refinement/refinement_interval=2",
        "time/nlim=8", "time/tlim=10", "output1/dcycle=-1",
        "output3/dcycle=1", "output3/dt=100",
        "problem/refinement_hysteresis=1.4",
    ]


def _write_moving_refinement_table(path):
    rows = []
    for time in (-0.1, 0.0, 1.0):
        x1 = -0.8 + 0.5*time
        x2 = 0.8 - 0.5*time
        rows.append([
            time, 0.5, 0.5, x1, 0.0, 0.0, x2, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 0.0, 0.0, -0.5, 0.0, 0.0,
        ])
    path.write_text("\n".join(" ".join(f"{value:.17e}" for value in row)
                                for row in rows) + "\n", encoding="utf-8")


def run_refinement_checks():
    """Exercise moving trackers, target overlap, COM shells, and hysteresis."""
    basename = "dynbbh_refinement_analytic"
    result = subprocess.run([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        *_amr_args(), "problem/sep=4", "problem/q=2",
        "problem/amr_condition=tracker", "problem/tracker_1_rad=0.45",
        "problem/tracker_2_rad=0.45", "problem/tracker_1_reflevel=1",
        "problem/tracker_2_reflevel=2", "problem/radius_0_rad=0.35",
        "problem/radius_0_reflevel=1",
    ], check=True, capture_output=True, text=True)
    cycle, blocks = _latest_binary_mesh(basename)
    assert cycle == 8
    phase = 4.0**-1.5*0.8
    hole1 = np.array([8.0/3.0*math.cos(phase), 0.0, 0.0])
    hole2 = np.array([-4.0/3.0*math.cos(phase), 0.0, 0.0])
    assert _level_at(blocks, hole1) >= 1
    assert _level_at(blocks, hole2) == 2
    assert _level_at(blocks, np.zeros(3)) >= 1
    amr_counts = re.search(r"(\d+) MeshBlocks created, (\d+) deleted by AMR",
                           result.stdout)
    assert amr_counts and int(amr_counts.group(1)) > 0
    assert int(amr_counts.group(2)) == 0, result.stdout

    basename = "dynbbh_refinement_moving_overlap"
    table = Path(f"{basename}.traj").resolve()
    _write_moving_refinement_table(table)
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        *_amr_args(), "time/nlim=-1", "time/tlim=0.8",
        "problem/use_traj_table=true", f"problem/traj_file={table}",
        "problem/amr_condition=tracker", "problem/tracker_1_rad=0.6",
        "problem/tracker_2_rad=0.6", "problem/tracker_1_reflevel=1",
        "problem/tracker_2_reflevel=2", "problem/radius_0_rad=0.3",
        "problem/radius_0_reflevel=1",
    ])
    cycle, blocks = _latest_binary_mesh(basename)
    assert cycle > 0
    # At t=0.8 the table places the holes at x=-0.4 and +0.4.  Both tracker
    # spheres overlap the two central blocks, so the higher target must win.
    assert _level_at(blocks, np.array([-0.4, 0.0, 0.0])) == 2
    assert _level_at(blocks, np.array([0.4, 0.0, 0.0])) == 2
    assert _level_at(blocks, np.zeros(3)) == 2

    basename = "dynbbh_refinement_com_shell"
    subprocess.check_call([
        "./athena", "-i", INPUT_FILE, f"job/basename={basename}",
        *_amr_args(), "mesh/nx1=32", "mesh/x1min=-8", "mesh/x1max=8",
        "problem/amr_condition=none", "problem/radius_0_rad=0.4",
        "problem/radius_0_reflevel=2",
    ])
    _, blocks = _latest_binary_mesh(basename)
    assert _level_at(blocks, np.zeros(3)) == 2
    assert _level_at(blocks, np.array([7.0, 0.0, 0.0])) == 0

    bad = subprocess.run([
        "./athena", "-i", INPUT_FILE, "job/basename=dynbbh_bad_refinement",
        *_amr_args(), "mesh_refinement/num_levels=2",
        "problem/amr_condition=tracker", "problem/tracker_1_reflevel=2",
    ], check=False, capture_output=True, text=True)
    assert bad.returncode != 0
    assert "valid physical AMR levels" in bad.stdout + bad.stderr
