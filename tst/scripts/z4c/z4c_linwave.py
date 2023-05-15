# Regression test based on Newtonian hydro linear wave convergence problem
#
# Runs a linear wave convergence test in 3D and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary
# file z4c_lin_wave-errs.dat). We only test physical gw wave. Gauge waves
# are not tested in this regression script.

# Modules
import logging
import scripts.utils.athena as athena
import sys
sys.path.insert(0, '../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name
_int = ['rk2', 'rk3', 'rk4']
_ng = ['2', '3']


# Run AthenaK
def run(**kwargs):
    logger.debug('Runnning test ' + __name__)
    for iv in _int:
        for ng in _ng:
            for res in (16, 32):
                arguments = ['job/basename=z4c_lin_wave',
                             'time/tlim=1.0',
                             'time/nlim=-1',
                             'time/integrator=' + iv,
                             'mesh/nghost=' + ng,
                             'mesh/nx1=' + repr(res),
                             'mesh/nx2=' + repr(res),
                             'mesh/nx3=' + repr(res),
                             'meshblock/nx1=' + repr(res),
                             'meshblock/nx2=' + repr(res),
                             'meshblock/nx3=' + repr(res),
                             'z4c/diss=1.0',
                             'problem/amp=1.0e-6',
                             'pgen_name=z4c_linear_wave',
                             'output1/dt=-1.0',
                             'output2/dt=-1.0',
                             'output3/dt=-1.0']
                # run test
                athena.run('tests/linear_wave_z4c.athinput', arguments)


# Analyze outputs
def analyze():
    # NOTE(@hengrui.zhu):  In the below, we check the magnitude of the error
    # and error convergence rates.  We enforce stricter error and convergence
    # thresholds for high-order spatial differencing schemes.
    logger.debug('Analyzing test ' + __name__)
    data = athena_read.error_dat('build/src/z4c_lin_wave-errs.dat')
    data = data.reshape([len(_int), len(_ng), 2, data.shape[-1]])
    analyze_status = True
    for ii, iv in enumerate(_int):
        for ni, ng in enumerate(_ng):
            error_threshold = 0.0
            conv_threshold = 0.0
            if ((iv == 'rk3' or iv == 'rk4') and (ng == '3')):
                error_threshold = 8.0e-10
                conv_threshold = 0.09
            else:
                error_threshold = 2.5e-8
                conv_threshold = 0.3

            l1_rms_n16 = (data[ii][ni][0][4])
            l1_rms_n32 = (data[ii][ni][1][4])
            if l1_rms_n32 > error_threshold:
                logger.warning("z4c wave error too large for {0}+"
                               "ng={1} configuration, "
                               "error: {2:g} threshold: {3:g}".
                               format(iv, ng,
                                      l1_rms_n32,
                                      error_threshold))
                analyze_status = False
            if l1_rms_n32/l1_rms_n16 > conv_threshold:
                logger.warning("z4c wave not converging for {0}+"
                               "ng={1} configuration, "
                               "conv: {2:g} threshold: {3:g}".
                               format(iv, ng,
                                      l1_rms_n32/l1_rms_n16,
                                      conv_threshold))
                analyze_status = False

    return analyze_status
