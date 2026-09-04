"""Regression checks for dynbbh radiation initialization and transport."""

import shutil
import subprocess
from pathlib import Path

import numpy as np

import athena_read


INPUT_FILE = "inputs/dynbbh_radiation.athinput"
CBD_INPUT_FILE = "inputs/dynbbh_radiation_cbd.athinput"


def _run(args):
    subprocess.check_call(["./athena", *args])


def _read_table(path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    assert data.shape[1] == 13, (path, data.shape)
    assert np.all(np.isfinite(data)), path
    return data


def _latest_table(basename):
    paths = sorted(Path("tab").glob(f"{basename}.rad_coord.*.tab"))
    assert paths, basename
    return _read_table(paths[-1])


def _clean_outputs(*basenames):
    for directory in (Path("tab"), Path("rst")):
        if not directory.is_dir():
            continue
        for basename in basenames:
            for path in directory.glob(f"{basename}.*"):
                path.unlink()
        if not any(directory.iterdir()):
            directory.rmdir()


def run_radiation_regression():
    """Exercise LTE initialization, source coupling, restart, and AMR."""
    fresh = "dynbbh_radiation_regression"
    continued = "dynbbh_radiation_restart_regression"
    legacy = "dynbbh_radiation_legacy_regression"
    activated = "dynbbh_radiation_activated_regression"
    adaptive = "dynbbh_radiation_amr_regression"
    legacy_input = Path("dynbbh_radiation_legacy.athinput")
    _clean_outputs(fresh, continued, legacy, activated, adaptive)

    try:
        _run([
            "-i", INPUT_FILE, f"job/basename={fresh}",
            "time/nlim=1", "time/tlim=0.01",
        ])
        initial = _read_table(Path("tab") / f"{fresh}.rad_coord.00000.tab")
        evolved = _read_table(Path("tab") / f"{fresh}.rad_coord.00001.tab")
        assert np.max(initial[:, 3]) > 0.0
        assert np.max(evolved[:, 3]) > 0.0
        assert not np.array_equal(initial[:, 3:], evolved[:, 3:])

        restart = Path("rst") / f"{fresh}.00001.rst"
        assert restart.is_file(), restart
        _run([
            "-r", str(restart), f"job/basename={continued}",
            "time/nlim=2", "time/tlim=0.02", "output1/dcycle=1",
        ])
        restarted = _latest_table(continued)
        assert np.max(restarted[:, 3]) > 0.0

        # Exercise opt-in activation from a pre-radiation GRMHD restart.
        source = Path(INPUT_FILE).read_text(encoding="utf-8")
        block_start = source.index("<dyn_radiation>")
        block_end = source.index("<problem>", block_start)
        legacy_input.write_text(
            source[:block_start] + source[block_end:], encoding="utf-8"
        )
        _run([
            "-i", str(legacy_input), f"job/basename={legacy}",
            "time/nlim=1", "time/tlim=0.01",
            "output1/variable=mhd_w", "output1/dt=-1", "output1/dcycle=-1",
        ])
        legacy_restart = Path("rst") / f"{legacy}.00001.rst"
        assert legacy_restart.is_file(), legacy_restart
        _run([
            "-r", str(legacy_restart), f"job/basename={activated}",
            "dyn_radiation/geometry=adm", "dyn_radiation/nlevel=1",
            "dyn_radiation/rotate_geo=false", "dyn_radiation/angular_fluxes=true",
            "dyn_radiation/reconstruct=plm", "dyn_radiation/rad_source=true",
            "dyn_radiation/kappa_a=0.01", "dyn_radiation/kappa_s=0.01",
            "dyn_radiation/kappa_p=0.0", "dyn_radiation/arad=1.0",
            "dyn_radiation/allow_missing_restart_i0=true",
            "time/nlim=2", "time/tlim=0.02",
            "output1/variable=rad_coord", "output1/id=rad_coord",
            "output1/dcycle=1",
            "output2/dt=-1",
        ])
        activated_table = _latest_table(activated)
        assert np.max(activated_table[:, 3]) > 0.0

        _run([
            "-i", INPUT_FILE, f"job/basename={adaptive}",
            "mesh_refinement/refinement=adaptive",
            "mesh_refinement/num_levels=2",
            "mesh_refinement/max_nmb_per_rank=128",
            "problem/amr_condition=tracker",
            "problem/tracker_1_rad=12", "problem/tracker_2_rad=12",
            "problem/tracker_1_reflevel=1", "problem/tracker_2_reflevel=1",
            "time/nlim=2", "time/tlim=0.02", "output1/dcycle=1",
            "output2/dt=-1",
        ])
        refined = _latest_table(adaptive)
        assert np.max(refined[:, 3]) > 0.0
        # A refined slice has more live cells than the four-block root mesh.
        assert refined.shape[0] > initial.shape[0]
    finally:
        _clean_outputs(fresh, continued, legacy, activated, adaptive)
        legacy_input.unlink(missing_ok=True)
        for directory in (Path("tab"), Path("rst")):
            if directory.is_dir() and not any(directory.iterdir()):
                shutil.rmtree(directory)


def run_cbd_regression():
    """Evolve a low-resolution radiative CBD and check coupled physical invariants."""
    basename = "dynbbh_radiation_cbd_regression"
    _clean_outputs(basename)
    try:
        _run(["-i", CBD_INPUT_FILE, f"job/basename={basename}"])
        rad_paths = sorted(Path("tab").glob(f"{basename}.rad_coord.*.tab"))
        mhd_paths = sorted(Path("tab").glob(f"{basename}.mhd_w_bcc.*.tab"))
        assert len(rad_paths) >= 3, rad_paths
        assert len(mhd_paths) >= 3, mhd_paths
        rad_initial = athena_read.tab(rad_paths[0])
        rad_final = athena_read.tab(rad_paths[-1])
        mhd_initial = athena_read.tab(mhd_paths[0])
        mhd_final = athena_read.tab(mhd_paths[-1])

        for data in (rad_initial, rad_final, mhd_initial, mhd_final):
            for key, values in data.items():
                if isinstance(values, np.ndarray):
                    assert np.all(np.isfinite(values)), key

        assert np.min(rad_final["r00"]) >= -1.0e-12
        assert np.max(rad_initial["r00"]) > 0.0
        assert not np.array_equal(rad_initial["r00"], rad_final["r00"])
        assert np.min(mhd_final["dens"]) > 0.0
        assert np.min(mhd_final["press"]) > 0.0
        assert not np.array_equal(mhd_initial["dens"], mhd_final["dens"])

        density_sum_ratio = np.sum(mhd_final["dens"])/np.sum(mhd_initial["dens"])
        assert 0.8 < density_sum_ratio < 1.2, density_sum_ratio

        def peak_radius(data):
            index = int(np.argmax(data["dens"]))
            return float(abs(data["x1v"][index]))

        r_initial = peak_radius(mhd_initial)
        r_final = peak_radius(mhd_final)
        assert abs(r_final - r_initial) < 8.0, (r_initial, r_final)

        radial_velocity = np.sign(mhd_final["x1v"])*mhd_final["velx"]
        mean_radial_velocity = np.average(radial_velocity, weights=mhd_final["dens"])
        assert abs(mean_radial_velocity) < 0.5, mean_radial_velocity

        magnetic_initial = sum(np.mean(mhd_initial[name]**2)
                               for name in ("bcc1", "bcc2", "bcc3"))
        magnetic_final = sum(np.mean(mhd_final[name]**2)
                             for name in ("bcc1", "bcc2", "bcc3"))
        if magnetic_initial > 0.0:
            assert magnetic_final/magnetic_initial < 10.0, (
                magnetic_initial, magnetic_final,
            )
        print(
            "short radiative CBD: equatorial density-sum ratio", density_sum_ratio,
            "density-peak radii", (r_initial, r_final),
            "density-weighted radial velocity", mean_radial_velocity,
            "mean B^2", (magnetic_initial, magnetic_final),
        )
    finally:
        _clean_outputs(basename)
