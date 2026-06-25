"""
Test for chemistry using the H2 network with uniform initial conditions. Runs tests for
different ODE solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import pathlib
import numpy as np

ode_solvers = ["forward_euler"]
input_file = "inputs/H2_uniform_test.athinput"


def H2_uniform_analytical_solution(
    t_code,
    f_H_0=1.0,
    cs_0=0.6,
    n_H=100.0,
    xi_cr=2.0e-16,
    k_gr=3.0e-17,
):
    """theoretical abundance of atomic hydrogen over time.
    input:
        t_code: time in code units, float or array
    optional parameters:
        f_H_0: initial atomic hydrogen abundance, default 0.
        cs_0: initial sound speed in code units, default 0.6.
        n_H: density in cm-3, default 100
        xi_cr: primary cosmic-ray ionization rate in s-1H-1, default 2.0e-16
        k_gr: grain surface recombination rate of H2, default 3.0e-17.
    output:
        t: time in s, float or array
        f_H: H abundance, float or array, between 0. and 1.
        T_g: gas temperature in K, float or array"""

    const_mh = 1.6733e-24
    const_kb = 1.380658e-16

    gm1 = 1.666666666666667 - 1.0
    muH = 1.4

    # defined in the athinput file
    unit_density_in_g = 2.108884e-24
    unit_energy_in_erg = 2.016257e-14
    t_unit = 3.155760e13

    num_density_to_density = const_mh * muH / unit_density_in_g

    k_cr = xi_cr * 3.0
    a1 = k_cr + 2.0 * n_H * k_gr
    a2 = k_cr
    t = t_code * t_unit
    fH = (f_H_0 - a2 / a1) * np.exp(-t * a1) + a2 / a1

    fH2 = 0.5 * (1 - fH)

    cv = 1.65 * const_kb
    alpha_gd = 3.2e-34

    T0 = num_density_to_density * cs_0**2 / gm1 * unit_energy_in_erg / cv
    Tg = (alpha_gd * n_H / (2.0 * cv) * t + 1.0 / np.sqrt(T0)) ** (-2)

    e0 = num_density_to_density * cs_0**2 / gm1 * unit_energy_in_erg
    eg = (alpha_gd * n_H * cv ** (-3.0 / 2.0) / (2) * t + 1.0 / np.sqrt(e0)) ** (-2)

    return unit_energy_in_erg, n_H, fH, fH2, Tg, eg


def H2_uniform_l1_errors(time, fH2_test, fH_test, e_int_test):
    # First compute the analytical answers
    unit_energy_in_erg, n_H, fH_fiducial, fH2_fiducial, Tg_fiducial, e_int_fiducial = (
        H2_uniform_analytical_solution(time)
    )

    # Convert to specific internal energy
    e_int_test = (e_int_test / n_H) * unit_energy_in_erg

    # Compute the L1 norms
    l1_fH2 = np.sum(np.abs(fH2_fiducial - fH2_test)) / fH2_fiducial.size
    l1_fH = np.sum(np.abs(fH_fiducial - fH_test)) / fH_fiducial.size
    l1_e_int = np.sum(np.abs(e_int_fiducial - e_int_test)) / e_int_fiducial.size

    return l1_fH2, l1_fH, l1_e_int


def run_h2_uniform(ode_solver, mpi=False):
    """Run the H2 uniform state test and compare to the analytic results.
    Parameterized over the different ODE solvers. This function is called by both the CPU
    and GPU tests."""
    RUN = testutils.mpi_run if mpi else testutils.run
    try:
        results = RUN(input_file, [f"chemistry/{ode_solver}"])
        assert results, f"H2 uniform test run failed for {ode_solver} solver."

        # Load the data
        files = sorted(pathlib.Path("./tab").glob("*.tab"))
        time = np.empty(len(files))
        H_abundance = np.empty_like(time)
        H2_abundance = np.empty_like(time)
        e_int = np.empty_like(time)

        for i, file in enumerate(files):
            data = athena_read.tab(file)
            time[i] = data["time"]
            H2_abundance[i] = data["s_00_chem_H2"][0]
            H_abundance[i] = data["s_01_chem_H"][0]
            e_int[i] = data["eint"][0]

            # Verify that all the active fields are constant across the domain
            active_fields = ["eint", "s_00_chem_H2", "s_01_chem_H"]
            for field in active_fields:
                for val in data[field]:
                    assert val == data[field][0], (
                        f"{field} is not constant across the domain."
                    )

            # Verify that the inactive fields are unchanged
            inactive_fields = ["dens", "velx", "vely", "velz"]
            fiducial_values = {
                "dens": 1.11083e02,
                "velx": 0.0,
                "vely": 0.0,
                "velz": 0.0,
            }
            for field in inactive_fields:
                for val in data[field]:
                    assert val == fiducial_values[field], (
                        f"{field} has incorrect value of {val}."
                    )

        # Compute the L1 errors with the analytical solution
        l1_fH2, l1_fH, l1_e_int = H2_uniform_l1_errors(
            time, H2_abundance, H_abundance, e_int
        )

        # Check the correct number of time steps
        fiducial_n_steps = 429
        test_n_steps = len(files)
        assert test_n_steps == fiducial_n_steps, (
            f"The number of time steps is not correct. Expected {fiducial_n_steps} but "
            "found {test_n_steps}."
        )

        # Check the L1 errors of active fields, all thresholds are the measured error
        # times a safety factor to reduce test brittleness
        l1_fH2_fiducial = 1.1 * 0.001144128932064307
        l1_fH_fiducial = 1.1 * 0.002288236484818268
        l1_e_int_fiducial = 1.1 * 1.65265104635921e-17
        assert l1_fH2 < l1_fH2_fiducial, (
            f"The L1 error for the XYZ of {l1_fH2} is greater than the allowed value"
            " of {l1_fH2_fiducial}."
        )
        assert l1_fH < l1_fH_fiducial, (
            f"The L1 error for the XYZ of {l1_fH} is greater than the allowed value"
            " of {l1_fH_fiducial}."
        )
        assert l1_e_int < l1_e_int_fiducial, (
            f"The L1 error for the XYZ of {l1_e_int} is greater than the allowed value"
            " of {l1_e_int_fiducial}."
        )

    finally:
        testutils.cleanup()


@pytest.mark.parametrize("ode_solver", ode_solvers)
def test_h2_uniform_gpu(ode_solver):
    """GPU Test for H2 uniform test problem."""
    run_h2_uniform(ode_solver)
