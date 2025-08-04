"""
Linear wave convergence test for special-relativistic hydro/MHD in 1D.
Runs tests in both hydro and MHD using RK3 for different
  - reconstruction algorithms
  - Riemann solvers
"""

# Modules
import sys
sys.path.append('../vis/python')
sys.path.append('../tst/test_suite')
import pytest
import test_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and wave types
maxerrors={
    ('hydro', 'rk3', 'plm', '0'): (1.7e-08,0.28),
    ('hydro', 'rk3', 'ppm4', '0'): (6.4e-09,0.31),
    ('hydro', 'rk3', 'ppmx', '0'): (1.2e-11,0.037),
    ('hydro', 'rk3', 'wenoz', '0'): (1.1e-11,0.17),
    ('hydro', 'rk3', 'plm', '4'): (1.8e-08,0.28),
    ('hydro', 'rk3', 'ppm4', '4'): (4.6e-09,0.23),
    ('hydro', 'rk3', 'ppmx', '4'): (4.3e-11,0.097),
    ('hydro', 'rk3', 'wenoz', '4'): (2.5e-11,0.13),
    ('hydro', 'rk3', 'plm', '3'): (1.8e-07,0.33),
    ('hydro', 'rk3', 'ppm4', '3'): (3.8e-08,0.26),
    ('hydro', 'rk3', 'ppmx', '3'): (1.2e-10,0.063),
    ('hydro', 'rk3', 'wenoz', '3'): (2.7e-11,0.036),
    ('mhd', 'rk3', 'plm', '0'): (4.1e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '0'): (1.4e-08,0.29),
    ('mhd', 'rk3', 'ppmx', '0'): (8.1e-11,0.088),
    ('mhd', 'rk3', 'wenoz', '0'): (3.8e-11,0.12),
    ('mhd', 'rk3', 'plm', '6'): (2.8e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '6'): (7.1e-09,0.23),
    ('mhd', 'rk3', 'ppmx', '6'): (3.8e-11,0.06),
    ('mhd', 'rk3', 'wenoz', '6'): (3.7e-11,0.12),
    ('mhd', 'rk3', 'plm', '5'): (5.7e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '5'): (1.8e-08,0.25),
    ('mhd', 'rk3', 'ppmx', '5'): (9.8e-11,0.088),
    ('mhd', 'rk3', 'wenoz', '5'): (3.5e-11,0.12),
    ('mhd', 'rk3', 'plm', '1'): (3.9e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '1'): (1.3e-08,0.26),
    ('mhd', 'rk3', 'ppmx', '1'): (1.9e-11,0.027),
    ('mhd', 'rk3', 'wenoz', '1'): (3.6e-11,0.19),
    ('mhd', 'rk3', 'plm', '4'): (3e-08,0.29),
    ('mhd', 'rk3', 'ppm4', '4'): (9.6e-09,0.26),
    ('mhd', 'rk3', 'ppmx', '4'): (1.9e-11,0.042),
    ('mhd', 'rk3', 'wenoz', '4'): (1.2e-11,0.098),
    ('mhd', 'rk3', 'plm', '2'): (1.9e-08,0.32),
    ('mhd', 'rk3', 'ppm4', '2'): (5e-09,0.26),
    ('mhd', 'rk3', 'ppmx', '2'): (1.5e-11,0.069),
    ('mhd', 'rk3', 'wenoz', '2'): (3.2e-12,0.039),
    ('mhd', 'rk3', 'plm', '3'): (3.3e-08,0.37),
    ('mhd', 'rk3', 'ppm4', '3'): (4.9e-09,0.24),
    ('mhd', 'rk3', 'ppmx', '3'): (1.4e-11,0.063),
    ('mhd', 'rk3', 'wenoz', '3'): (5.7e-12,0.032),}

_int   = ['rk3']
_recon = ['plm', 'ppm4', 'ppmx', 'wenoz']
_wave={}
_wave['mhd']  = ['0','6','5','1','4','2','3']
_wave['hydro'] = ['0','4','3']
_flux={}
_flux['mhd'] = ['llf', 'hlle']
_flux['hydro'] = ['llf', 'hlle', 'hllc']
_res = [32, 64] # resolutions to test

def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
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
            'coord/special_rel=true',
            'coord/general_rel=false',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/along_x1=true',
            'problem/amp=1.0e-6',
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("soe" , ["hydro","mhd"])  # system of eqns to test
def test_run(iv, rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    for fv in _flux[soe]:
        # Ignore return arguments
        _,_ = testutils.test_error_convergence(
            f"inputs/lwave_rel{soe}.athinput",
            f"sr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave[soe],
            _res,
            iv,
            rv,
            fv,
            soe,
            )
