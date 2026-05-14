# Regression test for importing SpECK GH Cartesian output into AthenaK Z4c.

import logging
import math
import os
import struct
import sys

import scripts.utils.athena as athena

logger = logging.getLogger('athena' + __name__[7:])
_resolutions = (6, 8, 10, 12)
_nghost = 3


def _sym_pairs():
    return [(a, b) for a in range(4) for b in range(a, 4)]


def _labels():
    result = []
    for prefix in ('psi', 'pi'):
        for a, b in _sym_pairs():
            result.append(f'{prefix}{a}{b}')
    for d in range(3):
        for a, b in _sym_pairs():
            result.append(f'phi{d}_{a}{b}')
    return result


def _ks_gh_values(x, y, z, mass=1.0):
    radius = math.sqrt(x * x + y * y + z * z)
    normal = [x / radius, y / radius, z / radius]
    h = 2.0 * mass / radius

    psi = [[0.0] * 4 for _ in range(4)]
    psi[0][0] = -1.0 + h
    for a in range(3):
        psi[0][a + 1] = psi[a + 1][0] = h * normal[a]
    for a in range(3):
        for b in range(3):
            psi[a + 1][b + 1] = (1.0 if a == b else 0.0) + h * normal[a] * normal[b]

    phi = [[[0.0] * 4 for _ in range(4)] for __ in range(3)]
    for d in range(3):
        dh = -h * normal[d] / radius
        dn = [((1.0 if d == a else 0.0) - normal[d] * normal[a]) / radius
              for a in range(3)]
        phi[d][0][0] = dh
        for a in range(3):
            value = dh * normal[a] + h * dn[a]
            phi[d][0][a + 1] = phi[d][a + 1][0] = value
        for a in range(3):
            for b in range(3):
                phi[d][a + 1][b + 1] = (
                    dh * normal[a] * normal[b]
                    + h * (dn[a] * normal[b] + normal[a] * dn[b]))

    alpha = 1.0 / math.sqrt(1.0 + h)
    beta = [h / (1.0 + h) * n for n in normal]
    pi = [[sum(beta[d] * phi[d][a][b] for d in range(3)) / alpha
           for b in range(4)] for a in range(4)]

    values = []
    for a, b in _sym_pairs():
        values.append(psi[a][b])
    for a, b in _sym_pairs():
        values.append(pi[a][b])
    for d in range(3):
        for a, b in _sym_pairs():
            values.append(phi[d][a][b])
    return values


def _write_cart_file(filename, resolution):
    axes = []
    for xmin, xmax in ((3.0, 5.0), (-1.0, 1.0), (-1.0, 1.0)):
        dx = (xmax - xmin) / resolution
        axes.append([xmin + ((q - _nghost) + 0.5) * dx
                     for q in range(resolution + 2 * _nghost)])
    center = [0.5 * (axis[0] + axis[-1]) for axis in axes]
    extent = [0.5 * (axis[-1] - axis[0]) for axis in axes]
    labels = _labels()
    payload = [[] for _ in labels]
    for z in axes[2]:
        for y in axes[1]:
            for x in axes[0]:
                values = _ks_gh_values(x, y, z)
                for var, value in enumerate(values):
                    payload[var].append(value)

    with open(filename, 'wb') as output:
        output.write(struct.pack('@if3f3f3i?i', 0, 0.0, *center, *extent,
                                 len(axes[0]), len(axes[1]), len(axes[2]),
                                 False, len(labels)))
        label_text = ' '.join(labels).encode()
        output.write(struct.pack('@i', len(label_text)))
        output.write(label_text)
        for values in payload:
            for value in values:
                output.write(struct.pack('@f', float(value)))


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
    for res in _resolutions:
        basename = f'z4c_speck_cart_reader_{res}'
        cart_file = f'speck_cart_ks_{res}.bin'
        _write_cart_file(os.path.join('build/src', cart_file), res)
        arguments = [
            f'job/basename={basename}',
            f'problem/speck_cart_file={cart_file}',
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
    if rows[-1]['h'] / rows[0]['h'] > 0.12:
        logger.warning('Hamiltonian constraint convergence is too weak: '
                       f'{rows[-1]["h"] / rows[0]["h"]:g}')
        ok = False
    if rows[-1]['m'] / rows[0]['m'] > 0.08:
        logger.warning('Momentum constraint convergence is too weak: '
                       f'{rows[-1]["m"] / rows[0]["m"]:g}')
        ok = False
    if rows[-1]['z'] / rows[0]['z'] > 0.08:
        logger.warning('Z constraint convergence is too weak: '
                       f'{rows[-1]["z"] / rows[0]["z"]:g}')
        ok = False
    return ok
