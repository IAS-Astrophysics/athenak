"""
Linear wave convergence test for general-relativistic hydro/MHD in 3D with AMR.
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
    ('hydro', 'rk2', 'plm', '0'): (1.5e-08,0.26),
    ('hydro', 'rk3', 'wenoz', '0'): (1.5e-09,0.23),
    ('mhd', 'rk2', 'plm', '0'): (4.6e-08,0.26),
    ('mhd', 'rk3', 'wenoz', '0'): (3.9e-09,0.23),}

_wave = ['0']
_res = [128, 256]        # resolutions to test.  Runs on GPU, so use higher res
_soe = ['hydro', 'mhd']  # system of equations to test

def arguments(iv, rv, fv, wv, res, soe, name, dim):
    """Assemble arguments for run command"""
    return  [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=4',
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=' + repr(res//2),
            'meshblock/nx1=' + repr(res//16),
            'meshblock/nx2=' + repr(res//16),
            'meshblock/nx3=' + repr(res//16),
            'time/cfl_number=0.4',
            'coord/special_rel=false',
            'coord/general_rel=true',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("dim" , [1,2,3])
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
@pytest.mark.parametrize("soe" , _soe)
def test_run(iv, fv, rv, soe, dim):
    """Run a single test with given parameters."""
    # Ignore return arguments
    _,_ = testutils.test_error_convergence(
        f"inputs/lwave_rel{soe}.athinput",
        f'gr_{soe}_linwave{dim}d_amr',
        lambda iv,rv,fv,wv,res,soe,name : arguments(iv,rv,fv,wv,res,soe,name,dim=dim),
        maxerrors,
        _wave,
        _res,
        iv,
        rv,
        fv,
        soe,)
