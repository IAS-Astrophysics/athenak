# Automatic test based on linear wave convergence in 1D
# In hydro, both L-/R-going sound waves and the entropy wave are tested.
# Note errors are very sensitive to the exact parameters (e.g. cfl_number, time limit)
# used. For the hard-coded error limits to apply, run parameters must not be changed.

# Modules

import sys
import os
sys.path.append('../vis/python')
sys.path.append('../tst/test_suite')
import pytest
import test_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Ensure the Athena executable is built
if not(os.path.isfile("build/src/athena")):
    testutils.clean_make()

_recon = ['plm','ppm4','ppmx','wenoz']  # do not change order
_flux = ['llf', 'hlle', 'hllc']
_res  = [256,512]                      # resolutions to test
_soe = ['hydro', 'mhd']  # system of equations to test
name = {'hydro': 'mb2', 'mhd': 'mub1'}  # names of the tests
# Reference keys for convergence tests
ref_key = {
    'hydro': ('hllc', 'wenoz'),
    'mhd': ('hlle', 'wenoz')}
# convergence ratio threshold for failure
ratio_threshold = {'hydro' : 0.6, 'mhd' : 0.8}  

def arguments(iv, rv, fv, res, name, soe):
    return [f'job/basename={name}_{iv}_{rv}_{fv}_{res}',
            'mesh/nx1='+repr(res),
            'meshblock/nx1=' + repr(128),
            'mesh/nghost=' + repr(2 if rv=='plm' else 3),
            'time/integrator=' + iv,
            'time/cfl_number=0.2',
            f'{soe}/reconstruct=' + rv,
            f'{soe}/rsolver=' + fv]

def run_test(iv, rv, fv, res, name, soe):
    """Run a single test with given parameters."""
    input_file = f"inputs/{name}.athinput" 
    testutils.athenak_run(input_file, arguments(iv, rv, fv, res, name, soe))
    data = athena_read.tab(f'tab/{name}_{iv}_{rv}_{fv}_{res}.{soe}_w.00001.tab')
    return data['dens']

# Dictionary to store density data for convergence tests
results={}
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
@pytest.mark.parametrize("soe" , _soe)
def test_run(fv, rv, soe):
    """Run a single test with given parameters."""
    iv ="rk2" if rv=='plm' else "rk3"
    if fv == 'hllc' and soe == 'mhd':
        pytest.skip("HLLC reconstruction is not available for MHD tests.")
    try:
        for res in _res:
            results[(soe,fv,rv,res)] = run_test(iv, rv, fv, res, name[soe], soe)
    finally:
        print("Cleaning up test files...")
        testutils.cleanup()

@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
@pytest.mark.parametrize("soe" , _soe)
def test_convergence(fv, rv, soe):
    iv ="rk2" if rv=='plm' else "rk3"
    if fv == 'hllc' and soe == 'mhd':
        pytest.skip("HLLC reconstruction is not available for MHD tests.")
    if ref_key[soe] == (fv, rv):
        pytest.skip(f"Can't compare reference against reference")
    
    error = {}
    # Calculate errors for the different resolutions
    for res in _res:
        ref = results[(soe,)+ref_key[soe]+(res,)]
        error[res] = (np.abs(results[(soe,fv,rv,res)]-ref)).mean()
        if error[res] > 3e-2:
            pytest.fail(f"Error for {(soe,fv,rv)} at resolution {res}: {error[res]}")
    # Check convergence
    ratio = error[_res[1]] / error[_res[0]]
    if ratio > ratio_threshold[soe]:
        pytest.fail(f"Convergence ratio for {(soe,fv,rv)} between {_res[1]} and {_res[0]}: {ratio} with threshold {ratio_threshold[soe]}")
