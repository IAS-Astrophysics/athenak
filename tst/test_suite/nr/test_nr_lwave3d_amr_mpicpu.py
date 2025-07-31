"""
Linear wave convergence test for non-relativistic hydro/MHD in 3D with AMR and MPI.
Runs tests in both hydro and MHD for different
  - time integrators
  - reconstruction algorithms
Only tests "0" wave and HLLC/HLLD Riemann solvers
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
    ('hydro', 'rk2', 'plm', '0'): (8.6e-08,0.39),
    ('hydro', 'rk3', 'ppm4', '0'): (2.2e-08,0.23),
    ('hydro', 'rk3', 'ppmx', '0'): (6.4e-10,0.097),
    ('hydro', 'rk3', 'wenoz', '0'): (6.2e-10,0.11),
    ('mhd', 'rk2', 'plm', '0'): (5.7e-05,197),
    ('mhd', 'rk3', 'ppm4', '0'): (2.5e-08,0.3),
    ('mhd', 'rk3', 'ppmx', '0'): (3.2e-09,0.24),
    ('mhd', 'rk3', 'wenoz', '0'): (3.1e-09,0.25),
    }

_recon = ['ppm4','ppmx','wenoz']
_wave = ['0']                      
_flux = {'hydro': ['hllc'], 'mhd': ['hlld']}
_res = [64, 128]                            

def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=' + repr(2 if rv=='plm' else 4),
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=' + repr(res//2),
            'meshblock/nx1=' + repr(res//16),
            'meshblock/nx2=' + repr(res//16),
            'meshblock/nx3=' + repr(res//16),
            'time/cfl_number=0.4',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/amp=1.0e-3',
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , ['rk2'])
@pytest.mark.parametrize("rv" , ['plm'])
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run_plm(iv, rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    for fv in _flux[soe]:
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

@pytest.mark.parametrize("iv" , ['rk3'])
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run_other_recon(iv, rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    for fv in _flux[soe]:
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
