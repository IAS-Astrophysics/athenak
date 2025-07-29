
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
    ('hydro', 'rk3', 'plm', '0'): (3.7e-09,0.25),
    ('hydro', 'rk3', 'ppm4', '0'): (2e-09,0.25),
    ('hydro', 'rk3', 'ppmx', '0'): (1.8e-09,0.39),
    ('hydro', 'rk3', 'wenoz', '0'): (3.3e-10,0.24),
    ('mhd', 'rk3', 'plm', '0'): (1.1e-08,0.24),
    ('mhd', 'rk3', 'ppm4', '0'): (5e-09,0.22),
    ('mhd', 'rk3', 'ppmx', '0'): (3.7e-09,0.3),
    ('mhd', 'rk3', 'wenoz', '0'): (9.1e-10,0.24),}

_int = ['rk3']
_recon = ['wenoz']  # do not change order
_wave = ['0']                      # do not change order
_flux = ['hlle']
_res = [32, 64]                            # resolutions to test
_soe = ['hydro', 'mhd']  # system of equations to test

def arguments(iv, rv, fv, wv, res,soe,name):
    vflow = 1.0 if wv=='3' else 0.0
    return  [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=4',
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=1',
            'meshblock/nx1=16',
            'meshblock/nx2=16',
            'meshblock/nx3=1',
            'time/cfl_number=0.4',
            'coord/special_rel=false',
            'coord/general_rel=true',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/wave_flag=' + wv]


@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
@pytest.mark.parametrize("soe" , _soe)
def test_run(iv, fv, rv, soe):
    """Run a single test with given parameters."""
    l1_rms_l,l1_rms_r = testutils.test_error_convergence(
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
            )
