
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
maxerrors={}
maxerrors['rk3'] = np.array([[1e-08,],])
convrates={}
convrates['rk3'] = np.array([[0.3,],])

_int = ['rk3']
_recon = ['wenoz']  # do not change order
_wave = ['0']                      # do not change order
_flux = ['llf']
_res = [32, 64]                            # resolutions to test
_soe = ['hydro', 'mhd']  # system of equations to test

test_name = 'sr_linwave3d'  # Name of the test

def arguments(iv, rv, fv, wv, res,soe):
    """Run the Athena++ test with given parameters."""
    vflow = 1.0 if wv=='3' else 0.0
    return  [f'job/basename={test_name}',
                            'time/tlim=1.0',
                            'time/integrator=' + iv,
                            'mesh/nghost=4',
                            'mesh/nx1=' + repr(res),
                            'mesh/nx2=' + repr(res//2),
                            'mesh/nx3=' + repr(res//2),
                            'meshblock/nx1=16',
                            'meshblock/nx2=16',
                            'meshblock/nx3=16',
                            'time/cfl_number=0.4',
                            'coord/special_rel=true',
                            'coord/general_rel=false',
                            f'{soe}/reconstruct=' + rv,
                            f'{soe}/rsolver=' + fv,
                            'problem/wave_flag=' + wv]


@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
@pytest.mark.parametrize("soe" , _soe)
def test_run(iv, fv, rv, soe):
    """Run a single test with given parameters."""
    for wv in _wave:
        try:
            for res in _res:
                results = testutils.mpi_run(f"inputs/lwave_rel{soe}.athinput", arguments(iv, rv, fv, wv, res, soe))
                assert results, f"Test failed for iv={iv}, res={res}, fv={fv}, rv={rv}, wv={wv}./AthenaK run did not complete successfully."
            # Check the errors in the output
            ri = _recon.index(rv)
            wi = _wave.index(wv)
            data = athena_read.error_dat(f'{test_name}-errs.dat')
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_lr = data[0][L1_RMS_INDEX]
            l1_rms_hr = data[1][L1_RMS_INDEX]
            maxerror = maxerrors[iv][ri][wi]
            convrate = convrates[iv][ri][wi]

            if l1_rms_hr > maxerror:
                pytest.fail(f"{soe} {wv} wave error too large for {iv}+{rv}+{fv} configuration, "
                        f"error: {l1_rms_hr:g} threshold: {maxerror:g}")
            if l1_rms_hr / l1_rms_lr > convrate:
                pytest.fail(f"{soe} {wv} wave not converging for {iv}+{rv}+{fv} configuration, "
                        f"conv: {l1_rms_hr / l1_rms_lr:g} threshold: {convrate:g}")
        finally:
            testutils.cleanup()
