
# Automatic test based on linear wave convergence in 2D
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
errors = {
    ('hydro', 'rk2', 'plm', '0'): (6.9e-08,0.39),
    ('hydro', 'rk2', 'ppm4', '0'): (5.5e-08,0.37),
    ('hydro', 'rk2', 'ppmx', '0'): (1.1e-08,0.28),
    ('hydro', 'rk2', 'wenoz', '0'): (1.1e-08,0.26),
    ('hydro', 'rk3', 'plm', '0'): (5.9e-08,0.38),
    ('hydro', 'rk3', 'ppm4', '0'): (1.9e-08,0.22),
    ('hydro', 'rk3', 'ppmx', '0'): (4e-10,0.075),
    ('hydro', 'rk3', 'wenoz', '0'): (2.7e-10,0.093),
    ('mhd', 'rk2', 'plm', '0'): (7.6e-08,0.38),
    ('mhd', 'rk2', 'ppm4', '0'): (5.4e-08,0.36),
    ('mhd', 'rk2', 'ppmx', '0'): (1.1e-08,0.28),
    ('mhd', 'rk2', 'wenoz', '0'): (1.2e-08,0.26),
    ('mhd', 'rk3', 'plm', '0'): (6.4e-08,0.37),
    ('mhd', 'rk3', 'ppm4', '0'): (2.1e-08,0.22),
    ('mhd', 'rk3', 'ppmx', '0'): (2.9e-09,0.2),
    ('mhd', 'rk3', 'wenoz', '0'): (2.8e-09,0.23),
}

_int = ['rk3']
_recon = ['plm','ppm4','ppmx','wenoz']  # do not change order
_wave = ['0']                      # do not change order
_flux = {'hydro': ['hllc'], 'mhd': ['hlld']}
_res = [32, 64]                            # resolutions to test

def arguments(iv,rv,fv,wv,res,soe,name):
    """Run the Athena++ test with given parameters."""
    vflow = 1.0 if wv=='3' else 0.0
    return [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=4',
            'mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res//2),
            'mesh/nx3=1',
            'meshblock/nx1=16',
            'meshblock/nx2=4',
            'meshblock/nx3=1',
            'time/cfl_number=0.4',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/amp=1.0e-6',
            'problem/wave_flag=' + wv,
            'problem/vflow=' + repr(vflow)]

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run(iv, rv, soe):
     for fv in _flux[soe]:
        """Run a single test with given parameters."""
        _,_ = testutils.test_error_convergence(
            f"inputs/linear_wave_{soe}.athinput",
            f"lwave2d_amr_{soe}",
            arguments,
            errors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,)
