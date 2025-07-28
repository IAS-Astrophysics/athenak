
# Automatic test based on linear wave convergence in 1D
# In hydro, both L-/R-going sound waves and the entropy wave are tested.
# Note errors are very sensitive to the exact parameters (e.g. cfl_number, time limit)
# used. For the hard-coded error limits to apply, run parameters must not be changed.

# Modules

import sys
sys.path.append('../vis/python')
sys.path.append('../tst/tests_suite')
import pytest
import tests_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np



_int = ['rk2', 'rk3']
_recon = ['plm', 'ppm4', 'ppmx', 'wenoz']  # do not change order
_wave = ['0','4','3']                      # do not change order
_flux = ['llf', 'hlle', 'hllc', 'roe']
_res = [32, 64]                            # resolutions to test
soe="hydro"

def arguments(iv,rv,fv,wv,res,name):
    vflow = 1.0 if wv=='3' else 0.0
    return [f'job/basename={name}',
            'time/tlim=1.0',
            'time/integrator=' + iv,
            'mesh/nghost=3',
            'mesh/nx1=' + repr(res),
            'mesh/nx2=1',
            'mesh/nx3=1',
            'meshblock/nx1=16',
            'meshblock/nx2=1',
            'meshblock/nx3=1',
            'mesh_refinement/refinement=none',
            'time/cfl_number=0.4',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/along_x1=true',
            'problem/amp=1.0e-6',
            'problem/wave_flag=' + wv,
            'problem/vflow=' + repr(vflow)]

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
def test_run(iv, fv, rv):
    """Run a single test with given parameters."""
    l1_rms_l,l1_rms_r = testutils.test_error_convergence("inputs/linear_wave_hydro.athinput",
                                    "lwave1d_hydro",
                                    arguments,
                                    _wave,
                                    _res,
                                    iv,
                                    rv,
                                    fv,
                                    soe=soe,
                                    left_wave='0',
                                    right_wave='4',
                                    )
                                    
    if l1_rms_l != l1_rms_r and rv != 'ppmx':
        # PPMX is known to have different errors for L/R-going waves
        # so we skip the check
        pytest.fail(f"Errors in L/R-going sound waves not equal for {iv}+{rv}+{fv} configuration, "
                f"L: {l1_rms_l:g} R: {l1_rms_r:g}")
