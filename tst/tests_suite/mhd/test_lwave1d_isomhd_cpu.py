
# Automatic test using linear wave convergence for ISOTHERMAL MHD in 1D
# In isothermal MHD, only L-/R-going fast/slow/Alfven waves are tested.
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
maxerrors['rk3'] = np.array([[8e-08, 8e-08, 6e-08, 6e-08, 9e-08, 9e-08],
                            [3e-08, 3e-08, 3e-08, 3e-08, 4e-08, 4e-08],
                            [3e-09, 3e-09, 3e-10, 3e-10, 4e-10, 4e-10],
                            [3e-09, 3e-09, 7e-11, 7e-11, 2e-10, 2e-10],])
convrates={}
convrates['rk3'] = np.array([[0.3, 0.3, 0.3, 0.3, 0.4, 0.4],
                            [0.3, 0.3, 0.3, 0.3, 0.4, 0.4],
                            [0.3, 0.3, 0.07, 0.07, 0.07, 0.07],
                            [0.3, 0.3, 0.07, 0.07, 0.1, 0.1],])
_int = ['rk3']
_recon = ['plm', 'ppm4', 'ppmx', 'wenoz']  # do not change order
_wave = ['0','5','1','4','2','3']          # do not chnage order
_flux = ['llf', 'hlle', 'hlld']
_res = [32, 64]                            # resolutions to test

@pytest.mark.parametrize("iv" , _int)
@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
def test_run(iv, fv, rv):
    """Run a single test with given parameters."""
    try:
        testutils.cleanup()
    except:
        next
    for wv in _wave:
        try:
            for res in _res:
                arguments = ['job/basename=mhd_linwave1d',
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
                            'mhd/eos=isothermal',
                            'mhd/reconstruct=' + rv,
                            'mhd/rsolver=' + fv,
                            'problem/along_x1=true',
                            'problem/amp=1.0e-6',
                            'problem/wave_flag=' + wv]
                results = testutils.athenak_run("inputs/linear_wave_mhd.athinput", arguments)
                assert results, f"Test failed for iv={iv}, res={res}, fv={fv}, rv={rv}, wv={wv}./AthenaK run did not complete successfully."

            ri = _recon.index(rv)
            wi = _wave.index(wv)
            data = athena_read.error_dat('mhd_linwave1d-errs.dat')
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_n32 = data[0][L1_RMS_INDEX]
            l1_rms_n64 = data[1][L1_RMS_INDEX]
            maxerror = maxerrors[iv][ri][wi]
            convrate = convrates[iv][ri][wi]

            if l1_rms_n64 > maxerror and not(rv=="ppmx" and iv=="rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                pytest.fail(f"{wv} wave error too large for {iv}+{rv}+{fv} configuration, "
                        f"error: {l1_rms_n64:g} threshold: {maxerror:g}")
            if l1_rms_n64 / l1_rms_n32 > convrate and not(rv=="ppmx" and iv=="rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                # Note that the convergence rate is defined as the ratio of errors at different resolutions
                pytest.fail(f"{wv} wave not converging for {iv}+{rv}+{fv}, "
                        f"conv: {l1_rms_n64 / l1_rms_n32:g} threshold: {convrate:g}")
            if wv == '0' and res==64:  # Left wave
                l1_rms_l = l1_rms_n64
            if wv == '5' and res==64:  # Right wave
                l1_rms_r = l1_rms_n64
        finally:
            testutils.cleanup()
    if l1_rms_l != l1_rms_r and rv == 'plm':
        # PPMX is known to have different errors for L/R-going waves
        # so we skip the check
        pytest.fail(f"Errors in L/R-going fast waves not equal for {iv}+{rv}+{fv} configuration, "
                f"L: {l1_rms_l:g} R: {l1_rms_r:g}")
