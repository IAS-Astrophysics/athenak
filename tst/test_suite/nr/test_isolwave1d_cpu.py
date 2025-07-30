
# Automatic test using linear wave convergence for ISOTHERMAL HYDRO in 1D
# In isothermal hydro, only L-/R-going sound waves are tested.
# Note errors are very sensitive to the exact parameters (e.g. cfl_number, time limit)
# used. For the hard-coded error limits to apply, run parameters must not be changed.

# Modules
import sys
sys.path.append('../vis/python')
sys.path.append('../tst/test_suite')
import pytest
import test_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Threshold errors and convergence rates
# for different integrators, reconstructions, and wave types
errors = {
    ('hydro', 'rk3', 'plm', '0'): (1.3e-08,0.28),
    ('hydro', 'rk3', 'ppm4', '0'): (3.2e-09,0.23),
    ('hydro', 'rk3', 'ppmx', '0'): (2.3e-11,0.075),
    ('hydro', 'rk3', 'wenoz', '0'): (1.6e-11,0.11),
    ('hydro', 'rk3', 'plm', '3'): (1.3e-08,0.28),
    ('hydro', 'rk3', 'ppm4', '3'): (3.2e-09,0.23),
    ('hydro', 'rk3', 'ppmx', '3'): (2.3e-11,0.075),
    ('hydro', 'rk3', 'wenoz', '3'): (1.6e-11,0.11),
    ('mhd', 'rk3', 'plm', '0'): (1.2e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '0'): (4.2e-09,0.3),
    ('mhd', 'rk3', 'ppmx', '0'): (1.5e-10,0.23),
    ('mhd', 'rk3', 'wenoz', '0'): (1.5e-10,0.25),
    ('mhd', 'rk3', 'plm', '5'): (1.2e-08,0.28),
    ('mhd', 'rk3', 'ppm4', '5'): (4.2e-09,0.3),
    ('mhd', 'rk3', 'ppmx', '5'): (1.5e-10,0.23),
    ('mhd', 'rk3', 'wenoz', '5'): (1.5e-10,0.25),
    ('mhd', 'rk3', 'plm', '1'): (1.3e-08,0.29),
    ('mhd', 'rk3', 'ppm4', '1'): (3.8e-09,0.26),
    ('mhd', 'rk3', 'ppmx', '1'): (1.4e-11,0.064),
    ('mhd', 'rk3', 'wenoz', '1'): (2.7e-12,0.064),
    ('mhd', 'rk3', 'plm', '4'): (1.3e-08,0.29),
    ('mhd', 'rk3', 'ppm4', '4'): (3.8e-09,0.26),
    ('mhd', 'rk3', 'ppmx', '4'): (1.4e-11,0.064),
    ('mhd', 'rk3', 'wenoz', '4'): (2.7e-12,0.064),
    ('mhd', 'rk3', 'plm', '2'): (2.5e-08,0.32),
    ('mhd', 'rk3', 'ppm4', '2'): (7.3e-09,0.28),
    ('mhd', 'rk3', 'ppmx', '2'): (1.8e-11,0.064),
    ('mhd', 'rk3', 'wenoz', '2'): (4e-12,0.056),
    ('mhd', 'rk3', 'plm', '3'): (2.5e-08,0.32),
    ('mhd', 'rk3', 'ppm4', '3'): (7.3e-09,0.28),
    ('mhd', 'rk3', 'ppmx', '3'): (1.8e-11,0.064),
    ('mhd', 'rk3', 'wenoz', '3'): (4e-12,0.056),
}


_int = ['rk3']
_recon = ['plm', 'ppm4', 'ppmx', 'wenoz']
_wave={}
_wave['hydro'] = ['0','3']
_wave['mhd']  = ['0','5','1','4','2','3']
_flux={}
_flux['mhd'] = ['llf', 'hlle', 'hlld']
_flux['hydro'] = ['llf', 'hlle', 'roe'] 
_res = [32, 64]                          # resolutions to test

def arguments(iv,rv,fv,wv,res,soe,name):
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
            f'{soe}/eos=isothermal',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv,
            'problem/along_x1=true',
            'problem/amp=1.0e-6',
            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("soe",["hydro","mhd"])
def test_run(iv, rv, soe):
    for fv in _flux[soe]:
        """Run a single test with given parameters."""
        l1_rms_l,l1_rms_r = testutils.test_error_convergence(
            f"inputs/linear_wave_{soe}.athinput",
            f"lwave1d_iso{soe}",
            arguments,
            errors,
            _wave[soe],
            _res,
            iv,
            rv,
            fv,
            soe,
            left_wave='0',
            right_wave='3' if soe=="hydro" else "5",
            )
        if l1_rms_l != l1_rms_r and rv == 'plm':
            pytest.fail(f"Errors in L/R-going sound waves not equal for {iv}+{rv}+{fv} configuration, "
                    f"L: {l1_rms_l:g} R: {l1_rms_r:g}")
