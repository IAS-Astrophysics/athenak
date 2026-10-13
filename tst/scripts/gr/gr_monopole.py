# Regression test based on BZ field rotation rates
#
# Runs a 3D GRMHD simulation for a spinning black hole and measures the field rotation
# rate measured at the black hole event horizon (exploiting the SphericalGrid
# infrastructure).

# Modules
import logging
import numpy as np
import scripts.utils.athena as athena
import sys
sys.path.insert(0, '../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    arguments = ['output1/dt=-1.0']
    athena.run('tests/monopole.athinput', arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the mean and standard deviation of the
    # field rotation rates (relative to the horizon rotation rate) computed at every
    # theta/phi location in the SphericalGrid analysis
    logger.debug('Analyzing test ' + __name__)
    data = athena_read.error_dat('build/src/monopole-diag.dat')
    omega = list(zip(*data))[2]
    omega_error = np.abs(np.average(omega) - 0.5)/0.5
    omega_std = np.std(omega)
    analyze_status = True
    error_threshold = 0.04
    std_threshold = 0.0525
    if (omega_error > error_threshold):
        logger.warning("Rotation rate error too large, "
                       "error: {0:g} threshold: {1:g}".
                       format(omega_error, error_threshold))
        analyze_status = False
    if (omega_std > std_threshold):
        logger.warning("Rotation rate standard deviation too large, "
                       "std: {0:g} threshold: {1:g}".
                       format(omega_std, std_threshold))
        analyze_status = False
    return analyze_status
