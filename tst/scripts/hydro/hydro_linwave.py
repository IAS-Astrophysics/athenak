# Regression test based on Newtonian hydro linear wave convergence problem
#
# Runs a linear wave convergence test in 3D and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary file
# hydro_lin_wave-errs.dat).  We test both L-/R-going sound waves and
# the entropy wave; shear waves are not tested in this regression script.

# Modules
import logging
import scripts.utils.athena as athena
import sys
sys.path.insert(0, '../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name
_int = ['rk2', 'rk3']
_recon = ['plm', 'ppmx', 'wenoz']
_flux = ['llf', 'hlle', 'hllc', 'roe']
_wave = ['L-sound', 'R-sound', 'entropy']


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    # L-going sound wave
    for iv in _int:
        for rv in _recon:
            for fv in _flux:
                for res in (16, 32):
                    arguments = ['job/basename=hydro_lin_wave',
                                 'time/tlim=1.0',
                                 'time/nlim=1000',
                                 'time/integrator=' + iv,
                                 'mesh/nghost=3',
                                 'mesh/nx1=' + repr(res),
                                 'mesh/nx2=' + repr(res/2),
                                 'mesh/nx3=' + repr(res/2),
                                 'meshblock/nx1=' + repr(res/4),
                                 'meshblock/nx2=' + repr(res/4),
                                 'meshblock/nx3=' + repr(res/4),
                                 'hydro/reconstruct=' + rv,
                                 'hydro/rsolver=' + fv,
                                 'problem/amp=1.0e-6',
                                 'output1/dt=-1.0',
                                 'output2/dt=-1.0',
                                 'output3/dt=-1.0']
                    # L-going sound wave
                    args_l = arguments + ['problem/wave_flag=0',
                                          'problem/vflow=0.0']
                    athena.run('tests/linear_wave_hydro.athinput', args_l)
                    # R-going sound wave
                    args_r = arguments + ['problem/wave_flag=4',
                                          'problem/vflow=0.0']
                    athena.run('tests/linear_wave_hydro.athinput', args_r)
                    # entropy wave
                    args_entr = arguments + ['problem/wave_flag=3',
                                             'problem/vflow=1.0']
                    athena.run('tests/linear_wave_hydro.athinput', args_entr)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the magnitude of the error,
    # error convergence rates, and error identicality between L- and R-going
    # sound waves.  We enforce stricter error and convergence thresholds for
    # high-order integration and reconstruction combintations.  Some waves
    # evolved with RK2 with high-order recon also pass the stricter threshold,
    # however, we do not check those here.  When checking for identical errors
    # in L- and R- sound waves, we only consider PLM, as higher-order recon
    # algorithms do show small differences in this test.
    logger.debug('Analyzing test ' + __name__)
    data = athena_read.error_dat('build/src/hydro_lin_wave-errs.dat')
    data = data.reshape([len(_int), len(_recon), len(_flux), 2,
                         len(_wave), data.shape[-1]])
    analyze_status = True
    for ii, iv in enumerate(_int):
        for ri, rv in enumerate(_recon):
            error_threshold = [0.0]*len(_wave)
            conv_threshold = [0.0]*len(_wave)
            if (iv == 'rk3' and (rv == 'ppmx' or rv == 'wenoz')):
                error_threshold[0] = error_threshold[1] = 6.0e-9  # sound
                error_threshold[2] = 4.5e-9  # entropy
                conv_threshold[0] = conv_threshold[1] = 0.07  # sound
                conv_threshold[2] = 0.08  # entropy
            else:
                error_threshold[0] = error_threshold[1] = 2.5e-7  # sound
                error_threshold[2] = 2.5e-7  # entropy
                conv_threshold[0] = conv_threshold[1] = 0.29  # sound
                conv_threshold[2] = 0.3  # entropy
            for fi, fv in enumerate(_flux):
                for wi, wv in enumerate(_wave):
                    l1_rms_n16 = (data[ii][ri][fi][0][wi][4])
                    l1_rms_n32 = (data[ii][ri][fi][1][wi][4])
                    if l1_rms_n32 > error_threshold[wi]:
                        logger.warning("{0} wave error too large for {1}+"
                                       "{2}+{3} configuration, "
                                       "error: {4:g} threshold: {5:g}".
                                       format(wv, iv, rv, fv,
                                              l1_rms_n32,
                                              error_threshold[wi]))
                        analyze_status = False
                    if l1_rms_n32/l1_rms_n16 > conv_threshold[wi]:
                        logger.warning("{0} wave not converging for {1}+"
                                       "{2}+{3} configuration, "
                                       "conv: {4:g} threshold: {5:g}".
                                       format(wv, iv, rv, fv,
                                              l1_rms_n32/l1_rms_n16,
                                              conv_threshold[wi]))
                        analyze_status = False
                if rv == 'plm':
                    l1_rms_l = data[ii][ri][fi][1][_wave.index('L-sound')][4]
                    l1_rms_r = data[ii][ri][fi][1][_wave.index('R-sound')][4]
                    if l1_rms_l != l1_rms_r:
                        logger.warning("Errors in L/R-going sound waves not "
                                       "equal for {0}+{1}+{2} configuration, "
                                       "{3:g} {4:g}".
                                       format(iv, rv, fv,
                                              l1_rms_l, l1_rms_r))
                        analyze_status = False

    return analyze_status
