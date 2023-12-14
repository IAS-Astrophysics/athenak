#! /usr/bin/env python

# Adapted from @c-white's tori.py script.  Soiled by @pdmullen.

# Python modules
import numpy as np
import scipy.optimize
import warnings

# Python plotting modules
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Physical parameters
a = 0.9375
r_edge = 6.0
r_peak = 12.0
gamma_adi = 4.0/3.0
rho_max = 1.0

# Grid parameters
r_max_global = 50.0
nx = 400
nz = 400


# Main function
def main(**kwargs):
    # Set up grid
    x = np.linspace(0.0, r_max_global, nx)
    z = np.linspace(-r_max_global/2.0, r_max_global/2.0, nz)
    x_grid, z_grid = np.meshgrid(x, z)
    r_grid = (x_grid**2 + z_grid**2)**0.5
    theta_grid = np.arccos(z_grid / r_grid)

    # Set up dictionary
    data = {}

    # Calculate Chakrabarti torus
    data['C'] = {}
    c, n = c_cn(r_edge, r_peak)
    ll = c_l_vec(r_grid, theta_grid, c, n)
    l_peak = c_l(r_peak, np.pi/2.0, c, n)
    h = c_h(r_grid, theta_grid, ll, c, n, r_edge)
    h_peak = c_h(r_peak, np.pi/2.0, l_peak, c, n, r_edge)
    rho = calculate_rho(h, h_peak)
    tt = calculate_tt(h, h_peak)
    data['C']['rho'] = np.where(r_grid >= r_edge, rho, np.nan)
    data['C']['tt'] = np.where(r_grid >= r_edge, tt, np.nan)

    # Calculate FM torus
    data['FM'] = {}
    f = fm_f_vec(r_grid, theta_grid, r_peak)
    f_in = fm_f(r_edge, np.pi/2.0, r_peak)
    f_peak = fm_f(r_peak, np.pi/2.0, r_peak)
    h = fm_h_vec(f, f_in)
    h_peak = fm_h(f_peak, f_in)
    rho = calculate_rho(h, h_peak)
    tt = calculate_tt(h, h_peak)
    data['FM']['rho'] = np.where(r_grid >= r_edge, rho, np.nan)
    data['FM']['tt'] = np.where(r_grid >= r_edge, tt, np.nan)

    # Prepare Figure
    fig = plt.figure(figsize=(18, 18))
    r_hor = 1.0 + (1.0 - a**2)**0.5
    plt.rc('text')
    plt.rc('font', family='serif', size=22)

    # Make Chakrabarti Density Plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_aspect('equal')
    ax1.add_artist(plt.Circle((0, 0), r_hor, color='white'))
    ax1.text(0.03, 0.93, 'Chakrabarti', color='white', transform=ax1.transAxes)
    im = ax1.pcolormesh(x_grid, z_grid, np.log10(data['C']['rho']), cmap='inferno',
                        vmin=-8, vmax=0)
    cmap = matplotlib.cm.get_cmap('inferno')
    ax1.set_facecolor(cmap(0.0))
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.tick_params(which='both', direction='in')
    plt.minorticks_on()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='$\\log_{10} \\rho$')
    ax1.tick_params(axis='both', which='both', colors='white')
    plt.setp(ax1.get_xticklabels(), color="black")

    # Make Chakrabarti Pressure Plot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_aspect('equal')
    ax2.add_artist(plt.Circle((0, 0), r_hor, color='white'))
    ax2.text(0.03, 0.93, 'Chakrabarti', color='white', transform=ax2.transAxes)
    im = ax2.pcolormesh(x_grid, z_grid,
                        np.log10(data['C']['rho']*data['C']['tt']), cmap='viridis',
                        vmin=-10, vmax=-2)
    cmap = matplotlib.cm.get_cmap('viridis')
    ax2.set_facecolor(cmap(0.0))
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.tick_params(which='both', direction='in')
    plt.minorticks_on()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='$\\log_{10} P_g$')
    ax2.tick_params(axis='both', which='both', colors='white')
    plt.setp(ax2.get_xticklabels(), color="black")

    # Make FM Density Plot
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_aspect('equal')
    ax3.add_artist(plt.Circle((0, 0), r_hor, color='white'))
    ax3.text(0.03, 0.93, 'Fishbone-Moncrief', color='white', transform=ax3.transAxes)
    im = ax3.pcolormesh(x_grid, z_grid, np.log10(data['FM']['rho']), cmap='inferno',
                        vmin=-8, vmax=0)
    cmap = matplotlib.cm.get_cmap('inferno')
    ax3.set_facecolor(cmap(0.0))
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.tick_params(which='both', direction='in')
    plt.minorticks_on()
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='$\\log_{10} \\rho$')
    ax3.tick_params(axis='both', which='both', colors='white')
    plt.setp(ax3.get_xticklabels(), color="black")

    # Make FM Pressure Plot
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_aspect('equal')
    ax4.add_artist(plt.Circle((0, 0), r_hor, color='white'))
    ax4.text(0.03, 0.93, 'Fishbone-Moncrief', color='white', transform=ax4.transAxes)
    im = ax4.pcolormesh(x_grid, z_grid,
                        np.log10(data['FM']['rho']*data['FM']['tt']), cmap='viridis',
                        vmin=-10, vmax=-2)
    cmap = matplotlib.cm.get_cmap('viridis')
    ax4.set_facecolor(cmap(0.0))
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    ax4.tick_params(which='both', direction='in')
    plt.minorticks_on()
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='$\\log_{10} P_g$')
    ax4.tick_params(axis='both', which='both', colors='white')
    plt.setp(ax4.get_xticklabels(), color="black")

    # Save Figure
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.3, hspace=-0.2)
    fig.savefig('gr_equilibria.png')
    plt.close(fig)


# Geometric factors
def geometry(r, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    delta = r**2 - 2.0*r + a**2
    sigma = r**2 + a**2*c**2
    aa = (r**2 + a**2)**2 - a**2*delta*s**2
    return s, delta, sigma, aa


# Metric factors
def metric(r, theta):
    s, delta, sigma, aa = geometry(r, theta)
    g_tt = -1.0 + 2.0*r / sigma
    g_tphi = -2.0*a*r / sigma*s**2
    g_phiphi = (r**2 + a**2 + 2.0*a**2*r / sigma*s**2)*s**2
    gtt = -aa / (delta*sigma)
    gtphi = -2.0*a*r / (delta*sigma)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in power',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value encountered in double_scalars',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value encountered in scalar power',
                                RuntimeWarning)
        alpha = (-gtt)**-0.5
    return g_tt, g_tphi, g_phiphi, gtt, gtphi, alpha


# Keplerian ll
def l_k(r):
    return (r**0.5*((1.0 - 2.0*a*(1.0/r)**0.5/r + a**2/r**2) /
                    (1.0 - 2.0/r + a*(1.0/r)**0.5/r)))


# Von Zeipel parameter
def vz(r, ll):
    return ((r**3 + a**2*r + 2.0*a*(a-ll)) / (r + 2*a/ll - 2.0))**0.5


# C c and n
def c_cn(r_edge, r_peak):
    l_in = l_k(r_edge)
    l_peak = l_k(r_peak)
    lambda_in = vz(r_edge, l_in)
    lambda_peak = vz(r_peak, l_peak)
    n = np.log(l_peak/l_in) / np.log(lambda_peak/lambda_in)
    c = l_in*lambda_in**(-n)
    return c, n


# C ll
def c_l(r, theta, c, n):
    g_tt, g_tphi, g_phiphi, _, _, _ = metric(r, theta)
    res = lambda ll : (ll/c)**(2.0/n) + (ll*g_phiphi + ll**2*g_tphi) / (g_tphi + ll*g_tt)  # noqa
    try:
        l_val = scipy.optimize.brentq(res, 1.0, 100.0)
    except ValueError:
        l_val = np.nan
    return l_val
c_l_vec = np.vectorize(c_l)  # noqa


# C h
def c_h(r, theta, ll, c, n, r_edge):
    l_in = c_l(r_edge, np.pi/2.0, c, n)
    u_t = c_u_t(r, theta, ll)
    u_t_in = c_u_t(r_edge, np.pi/2.0, l_in)
    h = u_t_in / u_t
    if n == 1.0:
        h *= (l_in/ll)**(c**2/(c**2-1.0))
    else:
        h *= (abs(1.0 - c**(2.0/n)*ll**(2.0-2.0/n))**(n/(2.0-2.0*n)) *
              abs(1.0 - c**(2.0/n)*l_in**(2.0-2.0/n))**(n/(2.0*n-2.0)))
    return h


# C u_t
def c_u_t(r, theta, ll):
    g_tt, g_tphi, g_phiphi, _, _, _ = metric(r, theta)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt',
                                RuntimeWarning)
        u_t = -((g_tphi**2 - g_tt*g_phiphi)/(g_phiphi + 2.0*ll*g_tphi + ll**2*g_tt))**0.5
    return u_t


# FM l_*
def fm_ls(r_peak):
    return ((1.0 / r_peak**3) ** 0.5*((r_peak**4 + a**2 * r_peak**2 - 2.0*a**2 * r_peak
                                       - a * (r_peak)**0.5 * (r_peak**2 - a**2)) /
                                      (r_peak**2 - 3.0 * r_peak + 2.0*a * (r_peak)**0.5)))


# FM f
def fm_f(r, theta, r_peak):
    s, delta, sigma, aa = geometry(r, theta)
    ls = fm_ls(r_peak)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in double_scalars',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value encountered in log',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'divide by zero encountered in double_scalars',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value encountered in scalar power',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value encountered in scalar subtract',
                                RuntimeWarning)
        warnings.filterwarnings('ignore', 'divide by zero encountered in scalar divide',
                                RuntimeWarning)
        f = (0.5 * np.log(aa/(delta*sigma) + ((aa/(delta*sigma))**2
                          + 4.0*ls**2/(delta*s**2)) ** 0.5)
             - 0.5 * (1.0 + 4.0*ls**2*delta*sigma**2/(aa**2*s**2)) ** 0.5 - 2.0*a*r*ls/aa)
    return f
fm_f_vec = np.vectorize(fm_f)  # noqa


# FM h
def fm_h(f, f_in):
    return np.exp(f - f_in)


# FM h vec
def fm_h_vec(f, f_in):
    h = fm_h(f, f_in)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in greater',
                                RuntimeWarning)
        h = np.where(h > 1.0, h, np.nan)
    return h


# Density from enthalpy
def calculate_rho(h, h_peak):
    tt = (gamma_adi-1.0)/gamma_adi*(h - 1.0)
    tt_peak = (gamma_adi-1.0)/gamma_adi*(h_peak - 1.0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in power',
                                RuntimeWarning)
        rho = rho_max*(tt / tt_peak)**(1.0/(gamma_adi-1.0))
    return rho


# Temperature from enthalpy
def calculate_tt(h, h_peak):
    tt = (gamma_adi-1.0)/gamma_adi*(h - 1.0)
    return tt


# Execute main function
if __name__ == '__main__':
    main()
