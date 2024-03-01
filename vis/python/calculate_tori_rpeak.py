#! /usr/bin/env python3

"""
Script for calculating radius of pressure maximum given inner and outer edges
of equilibrium torus (and optionally, the power law scaling of angular momentum,
n, for the chakrabarti case).

Usage:
calculate_tori_rpeak.py <fm or c> <spin> <r_in> <r_out> <--n>

This covers both the Fishbone-Moncrief (FM: 1976 ApJ 207 962) and Chakrabarti
(C: 1985 ApJ 288 1) tori. The latter is generally thinner for the same inputs,
assuming n is not set. Otherwise, you can specify n to control the thickness,
which will also shift the location of the pressure maximum.
Formulas also reference Abramowicz, Jaroszynski, & Sikora (AJS: 1978 AA 63 21)
and Penna, Kulkarni, & Narayan (PKN: 2013 AA 559 A116).
"""

# Python standard modules
import argparse
import warnings

# Numerical modules
import numpy as np
from scipy.optimize import brentq


# Main function
def main(**kwargs):

    # Numerical parameter
    r_in_factor = 1.01

    # Calculate peak radius for Fishbone-Moncrief torus
    if kwargs['torus_type'] == 'fm':
        def res(r):
            f_in = fm_f(kwargs['spin'], kwargs['r_in'], np.pi / 2.0, r)
            f_out = fm_f(kwargs['spin'], kwargs['r_out'], np.pi / 2.0, r)
            return f_in - f_out
        r_peak = brentq(res, kwargs['r_in'], kwargs['r_out'])

    # Calculate peak radius for Chakrabarti torus
    if kwargs['torus_type'] == 'c':
        def res(r):
            c, n = c_cn(kwargs['spin'], kwargs['r_in'], r, kwargs['n'])
            l_in = c_l(kwargs['spin'], kwargs['r_in'], np.pi / 2.0, c, n)
            l_out = c_l(kwargs['spin'], kwargs['r_out'], np.pi / 2.0, c, n)
            h_in = c_h(kwargs['spin'], kwargs['r_in'], np.pi / 2.0, l_in, c, n,
                       kwargs['r_in'])
            h_out = c_h(kwargs['spin'], kwargs['r_out'], np.pi / 2.0, l_out, c, n,
                        kwargs['r_in'])
            return h_out - h_in
        r_peak = brentq(res, kwargs['r_in'] * r_in_factor, kwargs['r_out'])

    # Report results
    print('r_peak: {0:24.16e}'.format(r_peak))


# Geometric factors
def geometry(spin, r, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    delta = r ** 2 - 2.0 * r + spin ** 2
    sigma = r ** 2 + spin ** 2 * c ** 2
    aa = (r ** 2 + spin ** 2) ** 2 - spin ** 2 * delta * s ** 2
    return s, delta, sigma, aa


# Metric factors
def metric(spin, r, theta):
    s, _, sigma, _ = geometry(spin, r, theta)
    g_tt = -1.0 + 2.0 * r / sigma
    g_tphi = -2.0 * spin * r / sigma * s ** 2
    g_phiphi = (r ** 2 + spin ** 2 + 2.0 * spin ** 2 * r / sigma * s ** 2) * s ** 2
    return g_tt, g_tphi, g_phiphi


# Fishbone-Moncrief l_* (FM 3.8)
def fm_ls(spin, r_peak):
    numerator = r_peak ** 4 + spin ** 2 * r_peak ** 2 - 2.0 * spin ** 2 * r_peak \
        - spin * r_peak ** 0.5 * (r_peak ** 2 - spin ** 2)
    denominator = r_peak ** 2 - 3.0 * r_peak + 2.0 * spin * r_peak ** 0.5
    return r_peak ** -1.5 * numerator / denominator


# Fishbone-Moncrief f (cf. FM 3.6)
def fm_f(spin, r, theta, r_peak):
    s, delta, sigma, aa = geometry(spin, r, theta)
    ls = fm_ls(spin, r_peak)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in double_scalars',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value encountered in log',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'divide by zero encountered in double_scalars',
                                RuntimeWarning)
        term_1 = 0.5 * np.log(aa / (delta * sigma)
                              + ((aa / (delta * sigma)) ** 2
                              + 4.0 * ls ** 2 / (delta * s ** 2)) ** 0.5)
        term_2 = \
            -0.5 * (1.0 + 4.0 * ls ** 2 * delta * sigma ** 2 / (aa ** 2 * s ** 2)) ** 0.5
        term_3 = -2.0 * spin * r * ls / aa
        f = term_1 + term_2 + term_3
    return f


# Abramowicz-Jaroszynski-Sikora u_t (AJS 5)
def ajs_u_t(spin, r, theta, ll):
    g_tt, g_tphi, g_phiphi = metric(spin, r, theta)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt',
                                RuntimeWarning)
        u_t = -((g_tphi ** 2 - g_tt * g_phiphi)
                / (g_phiphi + 2.0 * ll * g_tphi + ll ** 2 * g_tt)) ** 0.5
    return u_t


# Keplerian l (PKN 4)
def l_k(spin, r):
    numerator = 1.0 - 2.0 * spin / r ** 1.5 + spin ** 2 / r ** 2
    denominator = 1.0 - 2.0 / r + spin / r ** 1.5
    return r ** 0.5 * numerator / denominator


# Von Zeipel parameter (C 3.7, 3.8)
def vz(spin, r, ll):
    numerator = r ** 3 + spin ** 2 * r + 2.0 * spin * (spin - ll)
    denominator = r + 2.0 * spin / ll - 2.0
    return (numerator / denominator) ** 0.5


# Chakrabarti c and n (C 2.14a)
def c_cn(spin, r_in, r_peak, n_input):
    l_in = l_k(spin, r_in)
    l_peak = l_k(spin, r_peak)
    lambda_in = vz(spin, r_in, l_in)
    lambda_peak = vz(spin, r_peak, l_peak)
    if n_input == 0.0:
        n = np.log(l_peak / l_in) / np.log(lambda_peak / lambda_in)
        c = l_in / lambda_in ** n
    else:
        c = l_peak / lambda_peak ** n_input
        n = n_input
    return c, n


# Chakrabarti ll (C 2.5)
def c_l(spin, r, theta, c, n):
    variable_l_min = 1.0
    variable_l_max = 100.0
    g_tt, g_tphi, g_phiphi = metric(spin, r, theta)

    def res_c_l(ll):
        return (ll / c) ** (2.0 / n) \
            + (ll * g_phiphi + ll ** 2 * g_tphi) / (g_tphi + ll * g_tt)
    try:
        l_val = brentq(res_c_l, variable_l_min, variable_l_max)
    except ValueError:
        l_val = np.nan
    return l_val


# Chakrabarti h (C 2.16)
def c_h(spin, r, theta, ll, c, n, r_in):
    l_in = c_l(spin, r_in, np.pi / 2.0, c, n)
    u_t = ajs_u_t(spin, r, theta, ll)
    u_t_in = ajs_u_t(spin, r_in, np.pi / 2.0, l_in)
    h = u_t_in / u_t
    if n == 1.0:
        h *= (l_in / ll) ** (c ** 2 / (c ** 2 - 1.0))
    else:
        h *= abs(1.0 - c ** (2.0 / n) * ll ** (2.0 - 2.0 / n)) ** (n / (2.0 - 2.0 * n)) \
            * abs(1.0 - c ** (2.0 / n) * l_in ** (2.0 - 2.0 / n)) ** (n / (2.0 * n - 2.0))
    return h


# Parse inputs and execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('torus_type', choices=('fm', 'c'),
                        help='"fm" for Fishbone-Moncrief or "c" for Chakrabarti')
    parser.add_argument('spin', type=float, help='dimensionless spin')
    parser.add_argument('r_in', type=float, help='inner edge in gravitational radii')
    parser.add_argument('r_out', type=float, help='outer edge in gravitational radii')
    parser.add_argument('--n', type=float, help='slope of angular momentum scaling',
                        default=0.0)
    args = parser.parse_args()
    main(**vars(args))
