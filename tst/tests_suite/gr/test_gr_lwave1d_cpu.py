
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

# Threshold errors and convergence rates
# for different integrators, reconstructions, and wave types
maxerrors={
    ('hydro', 'rk3', 'plm', '0'): (1.7e-08,0.28),
    ('hydro', 'rk3', 'ppm4', '0'): (6.8e-09,0.28),
    ('hydro', 'rk3', 'ppmx', '0'): (1.1e-11,0.035),
    ('hydro', 'rk3', 'wenoz', '0'): (9.5e-12,0.27),
    ('hydro', 'rk3', 'plm', '4'): (1.8e-08,0.28),
    ('hydro', 'rk3', 'ppm4', '4'): (5.5e-09,0.26),
    ('hydro', 'rk3', 'ppmx', '4'): (3.8e-11,0.092),
    ('hydro', 'rk3', 'wenoz', '4'): (1.3e-11,0.23),
    ('hydro', 'rk3', 'plm', '3'): (1.8e-07,0.33),
    ('hydro', 'rk3', 'ppm4', '3'): (3.9e-08,0.24),
    ('hydro', 'rk3', 'ppmx', '3'): (1.2e-10,0.064),
    ('hydro', 'rk3', 'wenoz', '3'): (2.6e-11,0.032),
    ('mhd', 'rk3', 'plm', '0'): (4.1e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '0'): (1.5e-08,0.28),
    ('mhd', 'rk3', 'ppmx', '0'): (7.8e-11,0.086),
    ('mhd', 'rk3', 'wenoz', '0'): (2.8e-11,0.13),
    ('mhd', 'rk3', 'plm', '6'): (2.8e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '6'): (7.8e-09,0.25),
    ('mhd', 'rk3', 'ppmx', '6'): (2.9e-11,0.047),
    ('mhd', 'rk3', 'wenoz', '6'): (2.5e-11,0.13),
    ('mhd', 'rk3', 'plm', '5'): (5.7e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '5'): (1.9e-08,0.25),
    ('mhd', 'rk3', 'ppmx', '5'): (9.8e-11,0.086),
    ('mhd', 'rk3', 'wenoz', '5'): (3.1e-11,0.14),
    ('mhd', 'rk3', 'plm', '1'): (3.9e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '1'): (1.3e-08,0.25),
    ('mhd', 'rk3', 'ppmx', '1'): (1.7e-11,0.024),
    ('mhd', 'rk3', 'wenoz', '1'): (3.4e-11,0.22),
    ('mhd', 'rk3', 'plm', '4'): (3e-08,0.29),
    ('mhd', 'rk3', 'ppm4', '4'): (9.7e-09,0.25),
    ('mhd', 'rk3', 'ppmx', '4'): (1.9e-11,0.042),
    ('mhd', 'rk3', 'wenoz', '4'): (1.2e-11,0.1),
    ('mhd', 'rk3', 'plm', '2'): (1.9e-08,0.32),
    ('mhd', 'rk3', 'ppm4', '2'): (5e-09,0.25),
    ('mhd', 'rk3', 'ppmx', '2'): (1.5e-11,0.07),
    ('mhd', 'rk3', 'wenoz', '2'): (3.2e-12,0.039),
    ('mhd', 'rk3', 'plm', '3'): (3.3e-08,0.37),
    ('mhd', 'rk3', 'ppm4', '3'): (4.9e-09,0.24),
    ('mhd', 'rk3', 'ppmx', '3'): (1.4e-11,0.064),
    ('mhd', 'rk3', 'wenoz', '3'): (5.7e-12,0.033),}


_soe = ['hydro', 'mhd']  # system of equations to test
_int   = ['rk3']
_recon = ['plm', 'ppm4', 'ppmx', 'wenoz']
_wave={}
_wave['mhd']  = ['0','6','5','1','4','2','3']
_wave['hydro'] = ['0','4','3']
_flux={}
_flux['mhd'] = ['llf', 'hlle']
_flux['hydro'] = ['llf', 'hlle']
_res = [32, 64] # resolutions to test

def arguments(iv, rv, fv, wv, res, soe, name):
    return  [f'job/basename={name}',
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
            'coord/special_rel=false',
            'coord/general_rel=true',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("soe" , _soe)
def test_run(iv, rv, soe):
    """Run tests of GR hydro and mhd."""
    for fv in _flux[soe]:
        l1_rms_l,l1_rms_r = testutils.test_error_convergence(
            f"inputs/lwave_rel{soe}.athinput",
            f"gr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave[soe],
            _res,
            iv,
            rv,
            fv,
            soe,
            left_wave='0',
            right_wave='4' if soe == "hydro" else "6",
            )
