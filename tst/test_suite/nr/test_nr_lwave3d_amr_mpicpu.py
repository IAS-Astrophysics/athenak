"""
Linear wave convergence test for non-relativistic hydro/MHD in 3D with AMR and MPI.
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
errors={
    ('hydro', 'rk2', 'plm', '0'): (1.2e-05,0.29),
    ('hydro', 'rk3', 'ppm4', '0'): (7.6e-06,0.4),
    ('hydro', 'rk3', 'ppmx', '0'): (6.2e-06,0.55),
    ('hydro', 'rk3', 'wenoz', '0'): (5.7e-06,0.67),
    ('mhd', 'rk2', 'plm', '0'): (1.3e-05,0.28),
    ('mhd', 'rk3', 'ppm4', '0'): (8.1e-06,0.3),
    ('mhd', 'rk3', 'ppmx', '0'): (7e-06,0.37),
    ('mhd', 'rk3', 'wenoz', '0'): (3.9e-06,0.39)
}

_wave = ['0']                      
_recon = ['plm','wenoz']
_res = [32, 64]                            

def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=' + repr(2 if rv=='plm' else 4),
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=' + repr(res//2),
            'meshblock/nx1=' + repr(res//8),
            'meshblock/nx2=' + repr(res//8),
            'meshblock/nx3=' + repr(res//8),
            'time/cfl_number=0.3',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/amp=1.0e-3',
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , ['hlle'])
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run(fv, rv, soe):
    """run test with given integrator/resolution/physics."""
    iv = 'rk2' if rv=="plm" else 'rk3'
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
        soe,
        mpi=True)
