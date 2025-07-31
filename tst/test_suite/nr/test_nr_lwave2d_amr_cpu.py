"""
Linear wave convergence test for non-relativistic hydro/MHD in 2D with AMR.
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
errors = {
    ('hydro', 'rk2', 'plm', '0'): (6.9e-08,0.39),
    ('hydro', 'rk3', 'ppm4', '0'): (1.9e-08,0.22),
    ('hydro', 'rk3', 'ppmx', '0'): (4e-10,0.075),
    ('hydro', 'rk3', 'wenoz', '0'): (2.7e-10,0.093),
    ('mhd', 'rk2', 'plm', '0'): (7.6e-08,0.38),
    ('mhd', 'rk3', 'ppm4', '0'): (2.1e-08,0.22),
    ('mhd', 'rk3', 'ppmx', '0'): (2.9e-09,0.2),
    ('mhd', 'rk3', 'wenoz', '0'): (2.8e-09,0.23),
}

_recon = ['ppm4','ppmx','wenoz']  # do not change order
_wave = ['0']                      # do not change order
_flux = {'hydro': ['hllc'], 'mhd': ['hlld']}
_res = [64, 128]                            # resolutions to test

# Important amp=1.0e-3 so that it is large enough to trigger AMR
# Important to make sure there are enough MeshBlocks in root grid to enable AMR
def arguments(iv,rv,fv,wv,res,soe,name):
    """Assemble arguments for run command"""
    return [f'job/basename={name}',
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
            f"lwave2d_amr_{soe}",
            arguments,
            errors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,)

@pytest.mark.parametrize("iv" , ['rk3'])
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run_other_recon(iv, rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    for fv in _flux[soe]:
        # Ignore return arguments
        _,_ = testutils.test_error_convergence(
            f"inputs/lwave_{soe}.athinput",
            f"lwave2d_amr_{soe}",
            arguments,
            errors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,)
