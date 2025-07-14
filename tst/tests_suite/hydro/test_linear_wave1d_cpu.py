
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
maxerrors['rk2'] = np.array([[2.5e-8,  2.5e-8,  2.0e-8],
                            [2.0e-8,  2.0e-8,  7.0e-9],
                            [2.5e-9,  2.5e-9,  4.0e-10],
                            [2.5e-9,  2.5e-9,  4.0e-10]])

maxerrors['rk3'] = np.array([[2.0e-8,  2.0e-8,  2.0e-8],
                            [5.0e-9,  5.0e-9,  6.0e-9],
                            [4.0e-11, 4.0e-11, 2.0e-11],
                            [4.0e-11, 4.0e-11, 5.0e-12]])
convrates={}
convrates['rk2'] = np.array([[0.30, 0.30, 0.30],
                            [0.35, 0.35, 0.30],
                            [0.27, 0.27, 0.30],
                            [0.27, 0.27, 0.27]])

convrates['rk3'] = np.array([[0.30, 0.30, 0.30],
                            [0.25, 0.25, 0.25],
                            [0.08, 0.08, 0.07],
                            [0.11, 0.11, 0.07]])

_int = ['rk2', 'rk3']
_recon = ['plm', 'ppm4', 'ppmx', 'wenoz']  # do not change order
_wave = ['0','4','3']                      # do not change order
_flux = ['llf', 'hlle', 'hllc', 'roe']
_res = [32, 64]                            # resolutions to test

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
def test_run(iv, fv, rv):
    """Run a single test with given parameters."""
    for wv in _wave:
        vflow = 1.0 if wv=='3' else 0.0
        try:
            for res in _res:
                arguments = ['job/basename=hydro_linwave1d',
                            'time/tlim=1.0',
                            'time/integrator=' + iv,
                            'mesh/nghost=3',
                            'mesh/nx1=' + repr(res),
                            'mesh/nx2=1',
                            'mesh/nx3=1',
                            'meshblock/nx1=16',
                            'meshblock/nx2=1',
                            'meshblock/nx3=1',
                            'time/cfl_number=0.4',
                            'hydro/reconstruct=' + rv,
                            'hydro/rsolver=' + fv,
                            'problem/along_x1=true',
                            'problem/amp=1.0e-6',
                            'problem/wave_flag=' + wv,
                            'problem/vflow=' + repr(vflow)]
                results = testutils.athenak_run("inputs/linear_wave_hydro.athinput", arguments)
                assert results, f"Test failed for iv={iv}, res={res}, fv={fv}, rv={rv}, wv={wv}./AthenaK run did not complete successfully."

            ri = _recon.index(rv)
            wi = _wave.index(wv)
            data = athena_read.error_dat('hydro_linwave1d-errs.dat')
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_n32 = data[0][L1_RMS_INDEX]
            l1_rms_n64 = data[1][L1_RMS_INDEX]
            maxerror = maxerrors[iv][ri][wi]
            convrate = convrates[iv][ri][wi]

            if l1_rms_n64 > maxerror:
                pytest.fail(f"{wv} wave error too large for {iv}+{rv}+{fv} configuration, "
                        f"error: {l1_rms_n64:g} threshold: {maxerror:g}")
            if l1_rms_n64 / l1_rms_n32 > convrate:
                pytest.fail(f"{wv} wave not converging for {iv}+{rv}+{fv} configuration, "
                        f"conv: {l1_rms_n64 / l1_rms_n32:g} threshold: {convrate:g}")
            if wv == '0':  # Left wave
                l1_rms_l = l1_rms_n64
            if wv == '4':  # Right wave
                l1_rms_r = l1_rms_n64
        finally:
            testutils.cleanup()
    if l1_rms_l != l1_rms_r and rv != 'ppmx':
        # PPMX is known to have different errors for L/R-going waves
        # so we skip the check
        pytest.fail(f"Errors in L/R-going sound waves not equal for {iv}+{rv}+{fv} configuration, "
                f"L: {l1_rms_l:g} R: {l1_rms_r:g}")
