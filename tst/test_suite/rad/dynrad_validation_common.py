"""Cross-backend validation for ADM dynamic radiation."""

from pathlib import Path
import subprocess

import athena_read
import numpy as np


LWAVE_INPUT = "inputs/dynrad_lwave_adm.athinput"
EQUIL_INPUT = "inputs/dynrad_equilibration_adm.athinput"
BEAM_INPUT = "inputs/dynrad_beam_adm.athinput"


def _run(input_file, args):
    subprocess.check_call(["./athena", "-i", input_file, *args])


def _remove_outputs(*basenames):
    for basename in basenames:
        for path in Path("tab").glob(f"{basename}.*") if Path("tab").is_dir() else ():
            path.unlink()


def _latest_table(basename, file_id):
    paths = sorted(Path("tab").glob(f"{basename}.{file_id}.*.tab"))
    assert paths, (basename, file_id)
    data = athena_read.tab(paths[-1])
    for key, values in data.items():
        if isinstance(values, np.ndarray):
            assert np.all(np.isfinite(values)), (paths[-1], key)
    return data


def _equilibrium_gas_energy(t_final, dt=1.0e-5):
    """Integrate dEgas/dt = -(Tgas^4-Erad) for the test's unit constants."""
    egas = 3.0
    total_energy = 4.0
    gm1 = 2.0/3.0

    def rhs(value):
        tgas = gm1*value
        return -(tgas**4 - (total_energy - value))

    nstep = int(round(t_final/dt))
    step = t_final/nstep
    for _ in range(nstep):
        k1 = rhs(egas)
        k2 = rhs(egas + 0.5*step*k1)
        k3 = rhs(egas + 0.5*step*k2)
        k4 = rhs(egas + step*k3)
        egas += step*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    return egas


def run_linear_wave_convergence():
    """Require monotone convergence of the coupled radiation-fluid fast wave."""
    error_file = Path("dynrad_lwave-errs.dat")
    error_file.unlink(missing_ok=True)
    for resolution in (16, 32, 64):
        _run(LWAVE_INPUT, [
            f"mesh/nx1={resolution}", f"meshblock/nx1={resolution}",
        ])

    errors = athena_read.error_dat(error_file)[:, 4]
    assert errors.shape == (3,), errors
    assert np.all(np.isfinite(errors)) and np.all(errors > 0.0), errors
    ratios = errors[1:]/errors[:-1]
    assert np.all(ratios < 0.65), (errors, ratios)
    assert errors[-1] < 2.0e-8, errors
    print("dynrad ADM linear-wave RMS errors:", errors, "ratios:", ratios)


def run_equilibration_convergence():
    """Check homogeneous coupling against the independent thermal ODE."""
    basenames = tuple(f"dynrad_equil_cfl{index}" for index in range(3))
    _remove_outputs(*basenames)
    exact = _equilibrium_gas_energy(1.0)
    errors = []
    totals = []
    try:
        for basename, cfl in zip(basenames, (0.2, 0.1, 0.05)):
            _run(EQUIL_INPUT, [
                f"job/basename={basename}", f"time/cfl_number={cfl}",
            ])
            hydro = _latest_table(basename, "mhd_w")
            radiation = _latest_table(basename, "rad_coord")
            egas = float(np.mean(hydro["press"]))/(2.0/3.0)
            erad = float(np.mean(radiation["r00"]))
            assert np.std(hydro["press"]) < 1.0e-12
            assert np.std(radiation["r00"]) < 1.0e-12
            assert egas > 0.0 and erad > 0.0
            errors.append(abs(egas - exact))
            totals.append(egas + erad)
    finally:
        _remove_outputs(*basenames)

    errors = np.asarray(errors)
    totals = np.asarray(totals)
    assert np.all(np.diff(errors) < 0.0), errors
    assert errors[-1] < 0.35*errors[0], errors
    assert np.max(np.abs(totals - 4.0)) < 2.0e-10, totals
    print("dynrad ADM equilibration errors:", errors, "total energies:", totals)


def run_beam_regression():
    """Check ADM beam injection, direction, localization, and resolution stability."""
    basenames = ("dynrad_beam_low", "dynrad_beam_high")
    _remove_outputs(*basenames)
    moments = []
    try:
        for basename, nx1, nx2 in zip(basenames, (32, 64), (16, 32)):
            _run(BEAM_INPUT, [
                f"job/basename={basename}", f"mesh/nx1={nx1}",
                f"mesh/nx2={nx2}",
            ])
            data = _latest_table(basename, "rad_coord")
            energy = data["r00"]
            flux_x = data["r01"]
            center = np.abs(data["x2v"]) < 0.5
            outside = np.abs(data["x2v"]) > 0.8
            assert np.min(energy) > -1.0e-13
            assert np.max(energy[center]) > 1.0e-3
            assert np.max(energy[outside]) < 0.1*np.max(energy[center])
            assert np.mean(flux_x[center]) > 0.0
            positive = np.maximum(energy, 0.0)
            total = float(np.sum(positive))
            centroid = float(np.sum(positive*data["x2v"])/total)
            width = float(np.sqrt(np.sum(positive*(data["x2v"] - centroid)**2)/total))
            moments.append((total/nx2, centroid, width))
    finally:
        _remove_outputs(*basenames)

    low, high = np.asarray(moments)
    assert abs(high[0]/low[0] - 1.0) < 0.35, moments
    assert abs(low[1]) < 0.05 and abs(high[1]) < 0.05, moments
    assert abs(high[2] - low[2]) < 0.08, moments
    print("dynrad ADM beam moments (mean energy, centroid, width):", moments)


def run_all():
    run_linear_wave_convergence()
    run_equilibration_convergence()
    run_beam_regression()
