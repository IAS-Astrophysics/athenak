"""
Linear wave convergence test for special-relativistic hydro/MHD in 2D with AMR and MPI.
Runs tests in both hydro and MHD for RK2+PLM and RK3+WENOZ using HLLE Riemann solver.
Only tests "0" wave
"""

# Modules
import sys
sys.path.insert(0, '../vis/python')
sys.path.insert(0, '../test_suite')
import pytest
import test_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
maxerrors={
    ('hydro', 'rk2', 'plm', '0'): (3.8e-09,0.25),
    ('hydro', 'rk3', 'wenoz', '0'): (3.5e-10,0.22),
    ('mhd', 'rk2', 'plm', '0'): (1.2e-08,0.24),
    ('mhd', 'rk3', 'wenoz', '0'): (9.2e-10,0.24),}

_wave = ['0']        # do not change order
_res = [64, 128]     # resolutions to test
_soe = ['hydro', 'mhd']  # system of equations to test

def arguments(iv, rv, fv, wv, res,soe,name):
    """Assemble arguments for run command"""
    return  [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=' + repr(2 if rv=='plm' else 4),
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=1',
            'meshblock/nx1=' + repr(res//16),
            'meshblock/nx2=' + repr(res//16),
            'meshblock/nx3=1',
            'time/cfl_number=0.4',
            'coord/special_rel=true',
            'coord/general_rel=false',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/amp=1.0e-3',
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , ['rk2'])
@pytest.mark.parametrize("rv" , ['plm'])
@pytest.mark.parametrize("fv" , ['hlle'])
@pytest.mark.parametrize("soe" , _soe)
def test_run_plm(iv, fv, rv, soe):
    """Run a single test with given parameters."""
    # Ignore return arguments
    _,_ = testutils.test_error_convergence(
            f"inputs/lwave_rel{soe}.athinput",
            f"sr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,
            mpi=True)

@pytest.mark.parametrize("iv" , ['rk3'])
@pytest.mark.parametrize("rv" , ['wenoz'])
@pytest.mark.parametrize("fv" , ['hlle'])
@pytest.mark.parametrize("soe" , _soe)
def test_run_wenoz(iv, fv, rv, soe):
    """Run a single test with given parameters."""
    # Ignore return arguments
    _,_ = testutils.test_error_convergence(
            f"inputs/lwave_rel{soe}.athinput",
            f"sr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,
            mpi=True)
