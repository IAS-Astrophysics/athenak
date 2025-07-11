
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
maxerrors = np.array([[1e-06, ],
                        [1e-06, ],
                        [1e-07, ],
                        [1e-07, ]])
convrates={}
convrates = np.array([[0.4],
                        [0.3],
                        [0.2],
                        [0.3]])

_recon = ['plm','ppm4','ppmx','wenoz']  # do not change order
_wave = ['0']                      # do not change order
_flux = ['hllc']
_res = [32, 64]                            # resolutions to test
input_file = "inputs/linear_wave_hydro_amr.athinput"

def arguments(iv, rv, fv, wv, res):
    """Run the Athena++ test with given parameters."""
    vflow = 1.0 if wv=='3' else 0.0
    return ['job/basename=hydro_linwave3d_amr',
                            'time/tlim=1.0',
                            'time/integrator=' + iv,
                            'mesh/nghost=' + repr(2 if rv=='plm' else 4),
                            'mesh/nx1=' + repr(res),
                            'mesh/nx2=' + repr(res//2),
                            'mesh/nx3=' + repr(res//2),
                            'meshblock/nx1=8',
                            'meshblock/nx2=8',
                            'meshblock/nx3=8',
                            'time/cfl_number=0.4',
                            'hydro/reconstruct=' + rv,
                            'hydro/rsolver=' + fv,
                            'problem/amp=1.0e-6',
                            'problem/wave_flag=' + wv,
                            'problem/vflow=' + repr(vflow)]

@pytest.mark.parametrize("rv" , _recon)
@pytest.mark.parametrize("fv" , _flux)
def test_run(fv, rv):
    """Run a single test with given parameters."""
    iv ="rk2" if rv=='plm' else "rk3"
    for wv in _wave:
        try:
            for res in _res:
                results = testutils.mpi_run(input_file,
                                    arguments(iv, rv, fv, wv, res),
                                    threads=4,
                                        )
                assert results, f"Test failed for iv={iv}, res={res}, fv={fv}, rv={rv}, wv={wv}./AthenaK run did not complete successfully."
            # Check the errors in the output
            ri = _recon.index(rv)
            wi = _wave.index(wv)
            data = athena_read.error_dat('hydro_linwave3d_amr-errs.dat')
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_lr = data[0][L1_RMS_INDEX]
            l1_rms_hr = data[1][L1_RMS_INDEX]
            maxerror = maxerrors[ri][wi]
            convrate = convrates[ri][wi]

            if l1_rms_hr > maxerror:
                pytest.fail(f"{wv} wave error too large for  {iv}+{rv}+{fv} configuration, "
                        f"error: {l1_rms_hr:g} threshold: {maxerror:g}")
            if l1_rms_hr / l1_rms_lr > convrate:
                pytest.fail(f"{wv} wave not converging for  {iv}+{rv}+{fv} configuration, "
                        f"conv: {l1_rms_hr / l1_rms_lr:g} threshold: {convrate:g}")
        finally:
            testutils.cleanup()