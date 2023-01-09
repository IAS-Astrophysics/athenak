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


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    for res in (64, 128):
        arguments = ['job/basename=rad_linwave',
                     'time/tlim=1.0',
                     'mesh/nx1=' + repr(res),
                     'mesh/nx2=1',
                     'mesh/nx3=1',
                     'output1/dt=-1.0',
                     'output2/dt=-1.0']
        athena.run('tests/rad_linwave.athinput', arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the magnitude of the error and
    # error convergence rates.
    logger.debug('Analyzing test ' + __name__)
    data = athena_read.error_dat('build/src/rad_linwave-errs.dat')
    analyze_status = True
    error_threshold = 1.0e-8
    conv_threshold = 0.3
    l1_rms_n64 = data[0][4]
    l1_rms_n128 = data[1][4]
    if l1_rms_n128 > error_threshold:
        logger.warning("wave error too large, error: {0:g} threshold: {1:g}".
                       format(l1_rms_n128, error_threshold))
        analyze_status = False
    if l1_rms_n128/l1_rms_n64 > conv_threshold:
        logger.warning("wave not converging, conv: {0:g} threshold: {1:g}".
                       format(l1_rms_n128/l1_rms_n64, conv_threshold))
        analyze_status = False

    return analyze_status
