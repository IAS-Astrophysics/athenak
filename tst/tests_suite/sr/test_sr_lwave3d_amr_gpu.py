
# Automatic test based on linear wave convergence in 1D
# In hydro, both L-/R-going sound waves and the entropy wave are tested.
# Note errors are very sensitive to the exact parameters (e.g. cfl_number, time limit)
# used. For the hard-coded error limits to apply, run parameters must not be changed.

# Modules

import sys
sys.path.insert(0, '../vis/python')
sys.path.insert(0, '../tests_suite')
import pytest
import tests_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Threshold errors and convergence rates
# for different integrators, reconstructions, and wave types
maxerrors={
    ('hydro', 'rk3', 'ppm4', '0'): (8.3e-09,0.24),
    ('hydro', 'rk3', 'ppmx', '0'): (5.2e-09,0.27),
    ('hydro', 'rk3', 'wenoz', '0'): (1.6e-09,0.23),
    ('mhd', 'rk3', 'ppm4', '0'): (2.3e-08,0.23),
    ('mhd', 'rk3', 'ppmx', '0'): (1.3e-08,0.26),
    ('mhd', 'rk3', 'wenoz', '0'): (4.2e-09,0.23),}

_int = ['rk3']
_recon = ['wenoz']  # do not change order
_wave = ['0']                      # do not change order
_flux = ['hlle']
_res = [16, 32]                            # resolutions to test
_soe = ['hydro', 'mhd']  # system of equations to test

def arguments(iv, rv, fv, wv, res, soe, name, dim):
    return  [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=4',
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2 if dim > 1 else 1),
            'mesh/nx3=' + repr(res//4 if dim > 2 else 1),
            'meshblock/nx1=4',
            'meshblock/nx2='+ repr(4 if dim > 1 else 1),
            'meshblock/nx3='+ repr(4 if dim > 2 else 1),
            'time/cfl_number=0.4',
            'coord/special_rel=true',
            'coord/general_rel=false',
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
    _,_ = testutils.test_error_convergence(
        f"inputs/lwave_rel{soe}.athinput",
        f'SR_{soe}_linwave{dim}d_amr',
        lambda iv, rv, fv, wv, res, soe, name : arguments(iv, rv, fv, wv, res, soe, name, dim=dim),
        maxerrors,
        _wave,
        _res,
        iv,
        rv,
        fv,
        soe,)
