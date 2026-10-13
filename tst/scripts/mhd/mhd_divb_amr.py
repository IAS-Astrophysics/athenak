# Regression test for face-centered div(B) preservation under moving AMR.
#
# The divb_amr pgen initializes B from a discrete vector potential, then forces a moving
# AMR refinement pattern. This script checks direct face-centered divergence histories in
# 1D, 2D, and 3D with one through five physical AMR refinement levels.

import logging
import math
import os
import sys

import scripts.utils.athena as athena

sys.path.insert(0, '../vis/python')
import athena_read  # noqa

athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])

_CASES = [
    ('1D', 'tests/divb_amr_1d.athinput', 'DivBAMR1D', 64, 24, []),
    ('2D', 'tests/divb_amr_2d.athinput', 'DivBAMR2D', 48*48, 20, []),
]
_CASES += [
    ('3D-L{0}'.format(level), 'tests/divb_amr_3d.athinput',
     'DivBAMR3DL{0}'.format(level), 32*32*32, 8,
     ['mesh_refinement/num_levels={0}'.format(level + 1),
      'problem/refine_levels={0}'.format(level),
      'mesh_refinement/max_nmb_per_rank=16384',
      'time/nlim=8'])
    for level in range(1, 6)
]

_MAX_NDIV_TOL = 2.0e-11
_L1_NDIV_TOL = 2.0e-12
_L2_NDIV_TOL = 5.0e-12


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    for _, input_file, basename, _, _, arguments in _CASES:
        hst_file = os.path.join('build', 'src', basename + '.user.hst')
        if os.path.exists(hst_file):
            os.remove(hst_file)
        athena.run(input_file, [
            'job/basename=' + basename,
            'output1/dcycle=1',
        ] + arguments)


def analyze():
    logger.debug('Analyzing test ' + __name__)
    analyze_status = True

    for label, _, basename, root_ncell, min_rows, _ in _CASES:
        hst_file = os.path.join('build', 'src', basename + '.user.hst')
        data = athena_read.hst(hst_file)

        max_ndiv = max(data['max_ndiv'])
        l1_ndiv = max(s/v for s, v in zip(data['sum_ndiv'], data['vol']))
        l2_ndiv = max(math.sqrt(s/v) for s, v in zip(data['sum_n2'], data['vol']))
        max_ncell = max(data['ncell'])

        if len(data['time']) < min_rows:
            logger.warning('{0} AMR div(B) history is too short: {1} rows, expected {2}'.
                           format(label, len(data['time']), min_rows))
            analyze_status = False
        if max_ncell <= root_ncell:
            logger.warning('{0} AMR div(B) test did not refine: max_ncell={1:g}, '
                           'root_ncell={2:g}'.
                           format(label, max_ncell, root_ncell))
            analyze_status = False
        if max_ndiv > _MAX_NDIV_TOL:
            logger.warning('{0} max normalized div(B) too large: {1:g} threshold {2:g}'.
                           format(label, max_ndiv, _MAX_NDIV_TOL))
            analyze_status = False
        if l1_ndiv > _L1_NDIV_TOL:
            logger.warning('{0} L1 normalized div(B) too large: {1:g} threshold {2:g}'.
                           format(label, l1_ndiv, _L1_NDIV_TOL))
            analyze_status = False
        if l2_ndiv > _L2_NDIV_TOL:
            logger.warning('{0} L2 normalized div(B) too large: {1:g} threshold {2:g}'.
                           format(label, l2_ndiv, _L2_NDIV_TOL))
            analyze_status = False

    return analyze_status
