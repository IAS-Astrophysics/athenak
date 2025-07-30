# Automatic test based on linear wave convergence in 1D
# In hydro, both L-/R-going sound waves and the entropy wave are tested.
# Note errors are very sensitive to the exact parameters (e.g. cfl_number, time limit)
# used. For the hard-coded error limits to apply, run parameters must not be changed.

# Modules

import sys
sys.path.insert(0, '../vis/python')
sys.path.insert(0, '../test_suite')
import pytest
import test_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Threshold errors and convergence rates
# for different integrators, reconstructions, and wave types
errors={
    ('hydro', 'rk2', 'plm', '0'): (8.6e-08,0.39),
    ('hydro', 'rk2', 'ppm4', '0'): (1.1e-07,0.46),
    ('hydro', 'rk2', 'ppmx', '0'): (1.9e-08,0.28),
    ('hydro', 'rk2', 'wenoz', '0'): (1.9e-08,0.26),
    ('hydro', 'rk3', 'plm', '0'): (6.9e-08,0.38),
    ('hydro', 'rk3', 'ppm4', '0'): (2.2e-08,0.23),
    ('hydro', 'rk3', 'ppmx', '0'): (6.4e-10,0.097),
    ('hydro', 'rk3', 'wenoz', '0'): (6.2e-10,0.11),
    ('mhd', 'rk2', 'plm', '0'): (5.7e-05,197),
    ('mhd', 'rk2', 'ppm4', '0'): (1.1e-07,0.45),
    ('mhd', 'rk2', 'ppmx', '0'): (2.2e-08,0.28),
    ('mhd', 'rk2', 'wenoz', '0'): (2.2e-08,0.26),
    ('mhd', 'rk3', 'plm', '0'): (7.9e-08,0.37),
    ('mhd', 'rk3', 'ppm4', '0'): (2.5e-08,0.3),
    ('mhd', 'rk3', 'ppmx', '0'): (3.2e-09,0.24),
    ('mhd', 'rk3', 'wenoz', '0'): (3.1e-09,0.25),
    }

_recon = ['plm','ppm4','ppmx','wenoz']
_wave = ['0']                      
_flux = {'hydro': ['hllc'], 'mhd': ['hlld']}
_res = [32, 64]                            

def arguments(iv, rv, fv, wv, res, soe, name, dim=3):
    """Run the Athena++ test with given parameters."""
    vflow = 1.0 if wv=='3' else 0.0
    return [f'job/basename={name}',
                            'time/tlim=1.0',
                            'time/integrator=' + iv,
                            'mesh/nghost=' + repr(2 if rv=='plm' else 4),
                            'mesh/nx1=' + repr(res),
                            'mesh/nx2=' + repr(res//2 if dim>1 else 1),
                            'mesh/nx3=' + repr(res//2 if dim>2 else 1),
                            'meshblock/nx1=8',
                            'meshblock/nx2=' + "8" if dim>1 else "1",
                            'meshblock/nx3=' + "8" if dim>1 else "1",
                            'time/cfl_number=0.4',
                            f'{soe}/reconstruct=' + rv,
                            f'{soe}/rsolver=' + fv,
                            'problem/amp=1.0e-6',
                            'problem/wave_flag=' + wv,
                            'problem/vflow=' + repr(vflow)]

@pytest.mark.parametrize("dim," , np.arange(1,4))
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run(rv, soe, dim):
    """Run a single test with given parameters."""
    iv ="rk2" if rv=='plm' else "rk3"
    for fv in _flux[soe]:
        _,_ = testutils.test_error_convergence(
            f"inputs/linear_wave_{soe}.athinput",
            'soe_linwave{dim}d_amr',
            lambda iv, rv, fv, wv, res, soe, name : arguments(iv, rv, fv, wv, res, soe, name, dim=dim),
            errors,
            _wave,
            _res,
            iv,
            rv,
            fv,
            soe,
            mpi=True)
