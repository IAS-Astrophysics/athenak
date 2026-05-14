# Regression test for importing SpECK GH Cartesian output into AthenaK Z4c.

import logging
import math
import os
import shutil
import subprocess

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])
_resolutions = (16, 32, 64)
_nghost = 3


def _compile_generator():
    compiler = shutil.which(os.environ.get('H5CXX', 'h5c++'))
    if compiler is None:
        raise RuntimeError('z4c_speck_cart_reader HDF5 regression requires h5c++')
    source = os.path.join(os.path.dirname(__file__),
                          'speck_cart_hdf5_generator.cpp')
    executable = os.path.join('build', 'src', 'speck_cart_hdf5_generator')
    needs_build = not os.path.exists(executable)
    if not needs_build:
        needs_build = os.path.getmtime(executable) < os.path.getmtime(source)
    if needs_build:
        subprocess.check_call([compiler, '-O2', '-std=c++17', source,
                               '-o', executable])
    return executable


def _write_cart_file(generator, filename, resolution):
    subprocess.check_call([generator, filename, str(resolution), str(_nghost)])


def _read_constraint_row(filename):
    with open(filename, 'r') as data:
        rows = [line.split() for line in data.readlines()
                if line.strip() and not line.startswith('#')]
    row = rows[-1]
    return {
        'nx': int(row[0]),
        'c': float(row[4]),
        'h': float(row[5]),
        'm': float(row[6]),
        'z': float(row[7]),
    }


def run(**kwargs):
    logger.debug('Running test ' + __name__)
    os.makedirs('build/src', exist_ok=True)
    generator = _compile_generator()
    for res in _resolutions:
        basename = f'z4c_speck_cart_reader_{res}'
        cart_file = f'speck_cart_ks_{res}_g{_nghost}.h5'
        _write_cart_file(generator, os.path.join('build/src', cart_file), res)
        arguments = [
            f'job/basename={basename}',
            f'problem/speck_cart_file={cart_file}',
            f'mesh/nghost={_nghost}',
            f'mesh/nx1={res}',
            f'mesh/nx2={res}',
            f'mesh/nx3={res}',
            f'meshblock/nx1={res}',
            f'meshblock/nx2={res}',
            f'meshblock/nx3={res}',
            'time/tlim=1.0e-12',
        ]
        athena.run('tests/z4c_speck_cart_reader.athinput', arguments)


def analyze():
    logger.debug('Analyzing test ' + __name__)
    rows = [_read_constraint_row(
        f'build/src/z4c_speck_cart_reader_{res}-speck-cart-constraints.dat')
        for res in _resolutions]
    ok = True
    for low, high in zip(rows, rows[1:]):
        if high['h'] >= low['h'] or high['m'] >= low['m'] or high['z'] >= low['z']:
            logger.warning('SpECK cart constraints are not monotonically convergent: '
                           f'{low} -> {high}')
            ok = False
        h_rate = math.log(low['h'] / high['h'], 2.0)
        m_rate = math.log(low['m'] / high['m'], 2.0)
        z_rate = math.log(low['z'] / high['z'], 2.0)
        if h_rate < 3.5 or m_rate < 3.5 or z_rate < 3.5:
            logger.warning('SpECK HDF5 cart constraint rate is too weak: '
                           f'H={h_rate:g}, M={m_rate:g}, Z={z_rate:g}')
            ok = False
    if rows[-1]['h'] > 5.0e-8:
        logger.warning('Hamiltonian constraint convergence is too weak: '
                       f'{rows[-1]["h"]:g}')
        ok = False
    if rows[-1]['m'] > 1.0e-8:
        logger.warning('Momentum constraint convergence is too weak: '
                       f'{rows[-1]["m"]:g}')
        ok = False
    if rows[-1]['z'] > 2.0e-9:
        logger.warning('Z constraint convergence is too weak: '
                       f'{rows[-1]["z"]:g}')
        ok = False
    return ok
