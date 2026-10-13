# Regression test based on GR Bondi Accretion
#
# Runs a convergence test in 3D and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary file
# gr_bondi-errs.dat).

# Modules
import logging
import scripts.utils.athena as athena
import sys
sys.path.insert(0, '../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    for res in (64, 128):
        arguments = ['job/basename=gr_bondi',
                     'time/tlim=100.0',
                     'time/nlim=4000',
                     'time/integrator=rk2',
                     'mesh/nghost=2',
                     'mesh/nx1=' + repr(res),
                     'mesh/nx2=' + repr(res),
                     'mesh/nx3=' + repr(res),
                     'meshblock/nx1=' + repr(res/4),
                     'meshblock/nx2=' + repr(res/4),
                     'meshblock/nx3=' + repr(res/4),
                     'hydro/reconstruct=plm',
                     'hydro/rsolver=hlle',
                     'output1/dt=-1.0']
        athena.run('tests/bondi.athinput', arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the magnitude of the error and
    # error convergence rates.
    logger.debug('Analyzing test ' + __name__)
    data = athena_read.error_dat('build/src/gr_bondi-errs.dat')
    analyze_status = True
    error_threshold = 8.5e-7
    conv_threshold = 0.31
    l1_rms_n64 = data[0][4]
    l1_rms_n128 = data[1][4]
    if (l1_rms_n128 > error_threshold):
        logger.warning("RMS-L1-err too large, "
                       "error: {0:g} threshold: {1:g}".
                       format(l1_rms_n128,
                              error_threshold))
        analyze_status = False
    if (l1_rms_n128/l1_rms_n64 > conv_threshold):
        logger.warning("RMS-L1-err not converging, "
                       "conv: {0:g} threshold: {1:g}".
                       format(l1_rms_n128/l1_rms_n64,
                              conv_threshold))
        analyze_status = False
    return analyze_status
