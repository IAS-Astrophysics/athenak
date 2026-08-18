"""Focused MPI, mesh-refinement, and restart checks for RKL2 diffusion."""

import math
from pathlib import Path
import shutil

import numpy as np

import athena_read
import test_suite.testutils as testutils


def _mhd_input(tmp_path):
    """Add test-only static-refinement and restart blocks to the shared MHD input."""
    path = tmp_path / "sts_diffusion_mhd.athinput"
    path.write_text(
        Path("inputs/diffusion_mhd.athinput").read_text()
        + """

<mesh_refinement>
refinement = none

<refined_region1>
level = 1
x1min = -1.0
x1max = 1.0
x2min = -1.0
x2max = 1.0

<output1>
file_type = rst
dcycle = 0
"""
    )
    return str(path)


def test_sts_conduction_2d_mpi():
    """Exercise a nonzero cell-centered STS flux across MPI MeshBlocks."""
    basename = "sts_conduction_2d_mpi"
    try:
        results = testutils.mpi_run(
            "inputs/diffusion.athinput",
            [
                f"job/basename={basename}",
                "time/tlim=100.0",
                "time/nlim=2",
                "time/sts_integrator=rkl2",
                "time/sts_max_dt_ratio=8.0",
                "mesh/nx1=32",
                "mesh/nx2=32",
                "mesh/nx3=1",
                "meshblock/nx1=8",
                "meshblock/nx2=8",
                "meshblock/nx3=1",
                "problem/conduction_test=true",
                "problem/viscosity_test=false",
                "problem/spread_x1=true",
                "problem/spread_x2=true",
                "problem/spread_x3=false",
                "hydro/alpha_iso=2.0",
                "hydro/conductivity_integrator=sts",
            ],
            threads=4,
        )
        assert results, "2D MPI RKL2 conduction run failed"
        rms_l1 = athena_read.error_dat(f"{basename}-errs.dat")[0][4]
        assert math.isfinite(rms_l1) and rms_l1 < 1.0e-7
    finally:
        testutils.cleanup()


def test_sts_ohmic_2d_static_amr_mpi(tmp_path):
    """Exercise RKL2 Ohmic CT, MPI boundaries, and static refinement together."""
    basename = "sts_ohmic_2d_smr_mpi"
    try:
        results = testutils.mpi_run(
            _mhd_input(tmp_path),
            [
                f"job/basename={basename}",
                "time/tlim=100.0",
                "time/nlim=2",
                "time/sts_integrator=rkl2",
                "time/sts_max_dt_ratio=8.0",
                "mesh/nghost=2",
                "mesh/nx1=32",
                "mesh/nx2=32",
                "mesh/nx3=1",
                "meshblock/nx1=8",
                "meshblock/nx2=8",
                "meshblock/nx3=1",
                "mesh_refinement/refinement=static",
                "mhd/reconstruct=plm",
                "mhd/eta_ohm=2.0",
                "mhd/resistivity_integrator=sts",
                "problem/spread_x1=true",
                "problem/spread_x2=true",
                "problem/spread_x3=false",
                "problem/vel_comp=3",
            ],
            threads=4,
        )
        assert results, "2D static-AMR MPI RKL2 Ohmic run failed"
        row = athena_read.error_dat(f"{basename}-errs.dat")[0]
        assert row[3] == 2
        assert math.isfinite(row[4]) and row[4] < 1.0e-7
    finally:
        testutils.cleanup()


def test_sts_cycle_boundary_restart(tmp_path):
    """A restart between complete RKL2 sweeps must reproduce an uninterrupted run."""
    common = [
        "time/tlim=100.0",
        "time/sts_integrator=rkl2",
        "time/sts_max_dt_ratio=8.0",
        "mesh/nx1=32",
        "mesh/nx2=1",
        "mesh/nx3=1",
        "meshblock/nx1=16",
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        "mhd/eta_ohm=2.0",
        "mhd/resistivity_integrator=sts",
        "problem/spread_x1=true",
        "problem/spread_x2=false",
        "problem/spread_x3=false",
        "problem/vel_comp=2",
    ]
    shutil.rmtree("rst", ignore_errors=True)
    try:
        direct = "sts_restart_direct"
        assert testutils.mpi_run(
            _mhd_input(tmp_path),
            [f"job/basename={direct}", "time/nlim=4"] + common,
            threads=1,
        )
        direct_error = np.array(
            athena_read.error_dat(f"{direct}-errs.dat")[0], copy=True
        )
        testutils.cleanup()

        split = "sts_restart_split"
        assert testutils.mpi_run(
            _mhd_input(tmp_path),
            [
                f"job/basename={split}",
                "time/nlim=2",
                "output1/dcycle=2",
            ]
            + common,
            threads=1,
        )
        restart_file = Path("rst") / f"{split}.00001.rst"
        assert restart_file.exists()
        testutils.cleanup()

        resumed = "sts_restart_resumed"
        results = testutils.run_command(
            [
                "mpirun",
                "-np",
                "1",
                "./athena",
                "-r",
                str(restart_file),
                f"job/basename={resumed}",
                "time/nlim=4",
                "output1/dcycle=0",
            ]
        )
        assert results, "RKL2 restart run failed"
        resumed_error = athena_read.error_dat(f"{resumed}-errs.dat")[0]
        np.testing.assert_allclose(resumed_error, direct_error, rtol=0.0, atol=0.0)
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        testutils.cleanup()
