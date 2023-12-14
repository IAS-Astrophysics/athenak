# Regression test based on the 1D/2D Hohlraum test
#
# Runs a convergence test in 1D and 2D and checks L1 errors

# Modules
import logging
import numpy as np
import scripts.utils.athena as athena
import sys
sys.path.insert(0, '../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name
_res_survey = [3, 5]
_tf = 0.75


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    for res in _res_survey:
        arguments = ['job/basename=hohlraum_1d_res' + repr(res),
                     'time/tlim=0.75',
                     'time/nlim=500',
                     'radiation/nlevel=' + repr(res)]
        athena.run('tests/hohlraum_1d.athinput', arguments)


# Analyze outputs
def analyze():
    # NOTE(@pdmullen):  In the below, we check the magnitude of the error and
    # error convergence rates.
    logger.debug('Analyzing test ' + __name__)
    analyze_status = True

    nangles = np.array([10*(i**2)+2 for i in _res_survey])

    runs = []
    for res in _res_survey:
        runs.append(np.loadtxt('build/src/tab/'
                               'hohlraum_1d_res' + str(res) + '.rad_coord.00001.tab',
                               dtype=float, unpack=True, usecols=[2, 3, 4, 7]))

    def rtt_analytic(x, t):
        rtt = 0.5 * (1.0 - x / t)
        rtt[x > t] = 0.0
        return rtt

    def rtx_analytic(x, t):
        rtx = 0.25 * (1.0 - x**2 / t**2)
        rtx[x > t] = 0.0
        return rtx

    def rxx_analytic(x, t):
        rtx = (1.0/6.0) * (1.0 - x**3 / t**3)
        rtx[x > t] = 0.0
        return rtx

    rtt_l1 = np.array([np.sum(np.abs(runs[i][1] -
                                     rtt_analytic(runs[i][0], _tf)) * (1.0/128.0))
                       for i in range(0, len(nangles))])
    rtx_l1 = np.array([np.sum(np.abs(runs[i][2] -
                                     rtx_analytic(runs[i][0], _tf)) * (1.0/128.0))
                       for i in range(0, len(nangles))])
    rxx_l1 = np.array([np.sum(np.abs(runs[i][3] -
                                     rxx_analytic(runs[i][0], _tf)) * (1.0/128.0))
                       for i in range(0, len(nangles))])

    l1_errs = (1.0/np.sqrt(3.0))*np.sqrt(rtt_l1**2.0 + rtx_l1**2.0 + rxx_l1**2.0)

    error_threshold = 1.0e-3
    conv_threshold = 0.365

    if l1_errs[1] > error_threshold:
        logger.warning("Hohlraum 1D error too large, "
                       "error: {0:g} threshold: {1:g}".
                       format(l1_errs[1], error_threshold))
        analyze_status = False
    if l1_errs[1]/l1_errs[0] > conv_threshold:
        logger.warning("Hohlraum 1D not converging, "
                       "conv: {0:g} threshold: {1:g}".
                       format(l1_errs[1]/l1_errs[0], conv_threshold))
        analyze_status = False

    return analyze_status
