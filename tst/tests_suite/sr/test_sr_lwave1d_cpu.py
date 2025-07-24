
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
maxerrors={}
maxerrors['rk3'] = np.array([[1e-07,1e-07,1e-06,],
                            [1e-07,1e-07,1e-06,],
                            [1e-09,1e-09,1e-08,],
                            [1e-10,1e-09,1e-09,],])
convrates={}
convrates['rk3'] = np.array([[0.33,0.32,0.37,],
                            [0.3,0.3,0.3,],
                            [0.1,0.1,0.1,],
                            [0.3,0.3,0.04,],])

_int = ['rk3']
_recon = ['plm', 'ppm4', 'ppmx', 'wenoz']  # do not change order
_wave = ['0','4','3']                      # do not change order
_flux = ['llf', 'hlle', 'hllc']
_res = [32, 64]                            # resolutions to test
_soe = ['hydro', 'mhd']  # system of equations to test

test_name = 'sr_linwave1d'  # Name of the test

def arguments(iv, rv, fv, wv, res,soe):
    """Run the Athena++ test with given parameters."""
    vflow = 1.0 if wv=='3' else 0.0
    return  [f'job/basename={test_name}',
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
                            'problem/wave_flag=' + wv]

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
@pytest.mark.parametrize("soe" , _soe)
def test_run(iv, fv, rv, soe):
    """Run tests of SR hydro."""
    if soe == 'mhd' and fv == 'hllc':
        pytest.skip("HLLC flux is not implemented for MHD in SR hydro.")
    for wv in _wave:
        try:
            for res in _res:
                results = testutils.athenak_run(f"inputs/lwave_rel{soe}.athinput", arguments(iv, rv, fv, wv, res, soe))
                # Check if the AthenaK run was successful
                assert results, f"SR hydro failed for iv={iv}, res={res}, fv={fv}, rv={rv}, wv={wv}./AthenaK run did not complete successfully."

            ri = _recon.index(rv)
            wi = _wave.index(wv)
            data = athena_read.error_dat(f'{test_name}-errs.dat')
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_n32 = data[0][L1_RMS_INDEX]
            l1_rms_n64 = data[1][L1_RMS_INDEX]
            maxerror = maxerrors[iv][ri][wi]
            convrate = convrates[iv][ri][wi]

            if l1_rms_n64 > maxerror:
                pytest.fail(f"{wv} wave error in SR hydro too large for {iv}+{rv}+{fv} configuration, "
                        f"error: {l1_rms_n64:g} threshold: {maxerror:g}")
            if l1_rms_n64 / l1_rms_n32 > convrate:
                pytest.fail(f"{wv} wave in SR hydro not converging for {iv}+{rv}+{fv} configuration, "
                        f"conv: {l1_rms_n64 / l1_rms_n32:g} threshold: {convrate:g}")
            if wv == '0':  # Left wave
                l1_rms_l = l1_rms_n64
            if wv == '4':  # Right wave
                l1_rms_r = l1_rms_n64
        finally:
            testutils.cleanup()
