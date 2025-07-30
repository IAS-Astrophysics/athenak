"""
Linear wave convergence test for general-relativistic hydro/MHD in 2D with AMR.
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
    ('hydro', 'rk2', 'plm', '0'): (3.7e-09,0.25),
    ('hydro', 'rk3', 'wenoz', '0'): (3.3e-10,0.24),
    ('mhd', 'rk2', 'plm', '0'): (1.1e-08,0.24),
    ('mhd', 'rk3', 'wenoz', '0'): (9.1e-10,0.24),}

_wave = ['0']       # do not change order
_res = [64, 128]     # resolutions to test
_soe = ['hydro', 'mhd']  # system of equations to test

def arguments(iv, rv, fv, wv, res,soe,name):
    """Assemble arguments for run command"""
    return  [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=4',
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=1',
            'meshblock/nx1=' + repr(res//16),
            'meshblock/nx2=' + repr(res//16),
            'meshblock/nx3=1',
            'time/cfl_number=0.4',
            'coord/special_rel=false',
            'coord/general_rel=true',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
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
            f"gr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,)

@pytest.mark.parametrize("iv" , ['rk3'])
@pytest.mark.parametrize("rv" , ['wenoz'])
@pytest.mark.parametrize("fv" , ['hlle'])
@pytest.mark.parametrize("soe" , _soe)
def test_run_wenoz(iv, fv, rv, soe):
    """Run a single test with given parameters."""
    # Ignore return arguments
    _,_ = testutils.test_error_convergence(
            f"inputs/lwave_rel{soe}.athinput",
            f"gr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,)
