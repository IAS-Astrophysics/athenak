"""
Linear wave convergence test for general-relativistic MHD in dynamical spacetime (dyngr)
in 3D and with AMR.
Runs tests in MHD for RK2+PLM and RK3+WENOZ using HLLE Riemann solver
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
errors={
    ('mhd', 'rk2', 'plm', '0'): (4.6e-08,0.26),
    ('mhd', 'rk3', 'wenoz', '0'): (3.9e-09,0.23),}

_wave = ['0']       # do not change order
_res = [128, 256]   # resolutions to test, user higher res on GPU
_soe = ['mhd']  # system of equations to test

def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return  [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=' + repr(2 if rv=='plm' else 4),
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=' + repr(res//4),
            'meshblock/nx1=' + repr(res//8),
            'meshblock/nx2=' + repr(res//8),
            'meshblock/nx3=' + repr(res//8),
            'time/cfl_number=0.4',
            'coord/special_rel=false',
            'coord/general_rel=true',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , ['rk2'])
@pytest.mark.parametrize("rv" , ['plm'])
@pytest.mark.parametrize("fv" , ['hlle'])
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run_plm(iv, rv, fv, soe):
    """Run a single test with given parameters."""
    # Ignore return arguments
    _,_ = testutils.test_error_convergence(
        f"inputs/lwave_dyngrmhd.athinput",
        f"lwave3d_amr_{soe}",
        arguments,
        errors,
        _wave,
        _res,
        iv,
        rv,
        fv,
        soe)

@pytest.mark.parametrize("iv" , ['rk3'])
@pytest.mark.parametrize("rv" , ['wenoz'])
@pytest.mark.parametrize("fv" , ['hlle'])
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run_wenoz(iv, rv, fv, soe):
    """Run a single test with given parameters."""
    # Ignore return arguments
    _,_ = testutils.test_error_convergence(
        f"inputs/lwave_{soe}.athinput",
        f"lwave3d_amr_{soe}",
        arguments,
        errors,
        _wave,
        _res,
        iv,
        rv,
        fv,
        soe)

