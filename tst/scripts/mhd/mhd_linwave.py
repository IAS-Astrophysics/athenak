# Regression test based on Newtonian MHD linear wave convergence problem
#
# Runs a linear wave convergence test in 3D and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary file
# mhd_lin_wave-errs.dat).  We test L-/R- fast, L-/R-Alfven, L-/R- slow waves
# and the entropy wave.

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
_flux = ['llf', 'hlle', 'hlld']
_wave = ['L-fast', 'R-fast', 'L-Alfven', 'R-Alfven',
         'L-slow', 'R-slow', 'entropy']


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    # L-going sound wave
    for iv in _int:
        for rv in _recon:
            for fv in _flux:
                for res in (16, 32):
                    arguments = ['job/basename=mhd_lin_wave',
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
                                 'mhd/reconstruct=' + rv,
                                 'mhd/rsolver=' + fv,
                                 'problem/amp=1.0e-6',
                                 'output1/dt=-1.0',
                                 'output2/dt=-1.0',
                                 'output3/dt=-1.0',
                                 'output4/dt=-1.0',
                                 'output5/dt=-1.0']
                    # L-going fast wave
                    args_lf = arguments + ['problem/wave_flag=0',
                                           'problem/vflow=0.0']
                    athena.run('tests/linear_wave_mhd.athinput', args_lf)
                    # R-going fast wave
                    args_rf = arguments + ['problem/wave_flag=6',
                                           'problem/vflow=0.0']
                    athena.run('tests/linear_wave_mhd.athinput', args_rf)
                    # L-going alfven wave
                    args_la = arguments + ['problem/wave_flag=1',
                                           'problem/vflow=0.0']
                    athena.run('tests/linear_wave_mhd.athinput', args_la)
                    # R-going alfven wave
                    args_ra = arguments + ['problem/wave_flag=5',
                                           'problem/vflow=0.0']
                    athena.run('tests/linear_wave_mhd.athinput', args_ra)
                    # L-going slow wave
                    args_ls = arguments + ['problem/wave_flag=2',
                                           'problem/vflow=0.0']
                    athena.run('tests/linear_wave_mhd.athinput', args_ls)
                    # R-going slow wave
                    args_rs = arguments + ['problem/wave_flag=4',
                                           'problem/vflow=0.0']
                    athena.run('tests/linear_wave_mhd.athinput', args_rs)
                    # entropy wave
                    args_entr = arguments + ['problem/wave_flag=3',
                                             'problem/vflow=1.0']
                    athena.run('tests/linear_wave_mhd.athinput', args_entr)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the magnitude of the error,
    # error convergence rates, and error identicality between L- and R-going
    # waves.  We enforce stricter error and convergence thresholds for
    # high-order integration and reconstruction combintations.  Some waves
    # evolved with RK2 with high-order recon also pass the stricter threshold,
    # however, we do not check those here.  When checking for identical errors
    # in L- and R- MHD waves, we only consider fast waves with PLM, as
    # other waves and higher-order recon algorithms do show small differences
    # in this test.
    logger.debug('Analyzing test ' + __name__)
    analyze_status = True
    data = athena_read.error_dat('build/src/mhd_lin_wave-errs.dat')
    data = data.reshape([len(_int), len(_recon), len(_flux), 2,
                         len(_wave), data.shape[-1]])
    analyze_status = True
    for ii, iv in enumerate(_int):
        for ri, rv in enumerate(_recon):
            error_threshold = [0.0]*len(_wave)
            conv_threshold = [0.0]*len(_wave)
            if (iv == 'rk3' and (rv == 'ppmx' or rv == 'wenoz')):
                error_threshold[0] = error_threshold[1] = 2.0e-8  # fast
                error_threshold[2] = error_threshold[3] = 5.0e-8  # Alfven
                error_threshold[4] = error_threshold[5] = 2.0e-7  # slow
                error_threshold[6] = 5.0e-9  # entropy
                conv_threshold[0] = conv_threshold[1] = 0.15  # fast
                conv_threshold[2] = conv_threshold[3] = 0.23  # Alfven
                conv_threshold[4] = conv_threshold[5] = 0.15  # slow
                conv_threshold[6] = 0.08  # entropy
            else:
                error_threshold[0] = error_threshold[1] = 3.0e-7  # fast
                error_threshold[2] = error_threshold[3] = 2.5e-7  # Alfven
                error_threshold[4] = error_threshold[5] = 5.0e-7  # slow
                error_threshold[6] = 3.5e-7  # entropy
                conv_threshold[0] = conv_threshold[1] = 0.29  # fast
                conv_threshold[2] = conv_threshold[3] = 0.34  # Alfven
                conv_threshold[4] = conv_threshold[5] = 0.50  # slow
                conv_threshold[6] = 0.38  # entropy
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
                    l1_rms_lf = data[ii][ri][fi][1][_wave.index('L-fast')][4]
                    l1_rms_rf = data[ii][ri][fi][1][_wave.index('R-fast')][4]
                    if l1_rms_lf != l1_rms_rf:
                        logger.warning("Errors in L/R-going fast waves not "
                                       "equal for {0}+{1}+{2} configuration, "
                                       "{3:g} {4:g}".
                                       format(iv, rv, fv,
                                              l1_rms_lf, l1_rms_rf))
                        analyze_status = False

    return analyze_status
