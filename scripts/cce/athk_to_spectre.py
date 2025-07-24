#!/usr/bin/env python3
# Alireza Rashti - Oct 2024 (C)
# usage:
# $ ./me -h
#
# BUG:
# -[] see TODOs
# -[] see NOTEs

import os
import numpy as np
from scipy import special
import math
import argparse
import h5py
import struct
import glob
from scipy.interpolate import interp1d

# from itertools import product
# import matplotlib.pyplot as plt
# import glob
# import sympy
# import re

# ---------------------------------------------------------------------- #

# field names
g_field_names = [
    "gxx",
    "gxy",
    "gxz",
    "gyy",
    "gyz",
    "gzz",
    "betax",
    "betay",
    "betaz",
    "alp",
]

g_name_map = {
    "gxx": "gxx",
    "gxy": "gxy",
    "gxz": "gxz",
    "gyy": "gyy",
    "gyz": "gyz",
    "gzz": "gzz",
    "betax": "Shiftx",
    "betay": "Shifty",
    "betaz": "Shiftz",
    "alp": "Lapse",
}

g_name_index = {
    "gxx": 4,
    "gxy": 5,
    "gxz": 6,
    "gyy": 7,
    "gyz": 8,
    "gzz": 9,
    "betax": 1,
    "betay": 2,
    "betaz": 3,
    "alp": 0,
}

# real/imag
g_re = 0
g_im = 1

# args
g_args = None
# various attrs
g_attrs = None

# sign convention
g_sign = None

# debug
g_debug_max_l = 2


def parse_cli():
    """
    arg parser
    """
    p = argparse.ArgumentParser(description="convert Athenak CCE dumps to Spectre CCE")
    p.add_argument(
        "-fpath", type=str, required=True, help="/path/to/cce/dir/or/file/dumps"
    )
    p.add_argument(
        "-ftype",
        type=str,
        required=True,
        help="input file type:{h5:for pitnull,bin: for athenak}",
    )
    p.add_argument("-d_out", type=str, required=True, help="/path/to/output/dir")
    p.add_argument("-debug", type=str, default="n", help="debug=[y,n]")
    p.add_argument(
        "-radius",
        type=float,
        required=True,
        help="interpolate all fields and their derivatives at this radius.",
    )
    p.add_argument(
        "-t_deriv",
        type=str,
        default="fin_diff",
        help="method to take the time derivative of fields:{Fourier,fin_diff}",
    )
    p.add_argument(
        "-r_deriv",
        type=str,
        default="ChebU",
        help="method to take the radial derivative of \
            fields:{ChebU:Chebyshev of second kind}",
    )
    p.add_argument(
        "-interpolation",
        type=str,
        default="ChebU",
        help="method to interpolate fields at a given r:{ChebU:Chebyshev of second kind}",
    )

    args = p.parse_args()
    return args


class AngularTransform:
    """
    angular coordinate transformation from pittnull to spectre
    """

    def __init__(self, attrs: dict):
        self.attrs = attrs
        # no. of collocation pnts
        self.npnts = 2 * self.attrs["max_l"] + 1
        self.l_root = self._legendre_root()
        # theta collocation coords
        self.th = self._theta_gauss_legendre()
        # phi collocation coords
        self.ph = self._phi_equispace()
        # legendre roots
        # ylm(real/imag,th,ph,lm) on collocation coords
        self.ylm_pit = self._ylm_on_gl()
        # assert not np.any(np.isnan(self.th))
        # assert not np.any(np.isnan(self.ph))
        # assert not np.any(np.isnan(self.ylm_pit))
        # assert not np.any(np.isnan(self.l_root))

    def _theta_gauss_legendre(self):
        """
        creating gl collocation pnts for theta
        """
        th = np.empty(shape=(self.npnts), dtype=float)
        for i in range(self.npnts):
            th[i] = math.acos(-self.l_root[i])

        return th

    def _legendre_root(self):
        """
        legendre roots
        """
        nth = self.npnts
        x, _ = np.polynomial.legendre.leggauss(nth)
        x.sort()
        return x

    def _phi_equispace(self):
        """
        create equispace collocation pnts on phi
        """
        phi = np.linspace(0, 2 * np.pi, self.npnts, endpoint=False)

        return phi

    def _ylm_on_gl(self):
        """
        compute ylm on gauss legnedre for theta, and equispace for phi
        ylm[real/imag, theta, phi, lm]
        """

        ylms = np.empty(
            shape=(
                len([g_re, g_im]),
                self.npnts,
                self.npnts,
                self.attrs["max_lm"],
            ),
            dtype=float,
        )
        for i in range(self.npnts):
            th = self.th[i]
            for j in range(self.npnts):
                ph = self.ph[j]
                for ll in range(0, self.attrs["max_l"] + 1):
                    for m in range(ll, -ll - 1, -1):
                        lm = lm_mode(ll, m)
                        y = special.sph_harm(m, ll, ph, th)
                        # assert not np.isnan(y)
                        ylms[g_re, i, j, lm] = y.real
                        ylms[g_im, i, j, lm] = y.imag

        return ylms

    def reconstruct_pit_on_gl(self, coeff: np.array):
        """
        reconstruct field on gauss lengendre collocation pnts from coeffs of pittnull
        """
        rylm = self.ylm_pit[g_re]
        iylm = self.ylm_pit[g_im]

        field = np.dot(
            coeff[g_re, :, :, :], rylm[:, :, np.newaxis, :, np.newaxis]
        ) - np.dot(coeff[g_im, :, :, :], iylm[:, :, np.newaxis, :, np.newaxis])

        return field[:, :, :, :, 0, 0]

    def _sp_expansion(self, f):
        """
        Expands function f(theta, phi) in spherical harmonics up to degree L_max
        using Gauss-Legendre quadrature in theta and uniform quadrature in phi.

        Parameters:
            f : function [theta, phi] -> float
                Function to expand.
        Returns:
            coeff :Expansion coefficients.
        """

        # Gauss-Legendre quadrature points and weights for theta
        _, w_theta = np.polynomial.legendre.leggauss(self.npnts)
        dphi = 2 * np.pi / self.npnts
        Ylm = self.ylm_pit[g_re, ...] + 1j * self.ylm_pit[g_im, ...]

        coeff = np.einsum("trij,ijk,i->trk", f, np.conj(Ylm), w_theta, optimize=True)
        coeff *= dphi

        # coeff = np.transpose(coeff, (2, 1, 0))

        return coeff

    def transform_field_to_coeff_gl(self, field: np.array):
        """
        transformation of the given field on gauss legendre points to
        spherical harmonics basis on gauss legendre points.
        field[time,radius,theta,phi]
        """

        shape = (
            len([g_re, g_im]),
            self.attrs["lev_t"],
            self.attrs["max_n"],
            self.attrs["max_lm"],
        )
        coeff = np.empty(shape=shape, dtype=float)
        c = self._sp_expansion(field)

        coeff[g_re, ...] = c.real
        coeff[g_im, ...] = c.imag

        return coeff

    def transform_pit_coeffs_to_spec_coeffs(self, coeff: np.array):
        """
        transform pit coeffs which are on equispace theta and phi to spectre coeffs
        which are equispace on phi and gauss legendre on theta.
        """

        # first reconstruct field on gl collocation points
        field = self.reconstruct_pit_on_gl(coeff)

        return self.transform_field_to_coeff_gl(field)


def load(fpath: str, field_name: str, attrs: dict) -> list:
    """
    read the field accroding to attrs.
    return convention:
      ret[real/imag, time_level, n, lm], eg:
      ret[g_re,3,2,:] = Re(C_2lm(t=3)) for all lm
      ret[g_im,3,2,:] = Im(C_2lm(t=3)) for all lm
    """

    if attrs["file_type"] == "h5":
        lev_t = attrs["lev_t"]
        max_n = attrs["max_n"]
        max_lm = attrs["max_lm"]
        shape = (len([g_re, g_im]), lev_t, max_n, max_lm)
        ret = np.empty(shape=shape, dtype=float)

        with h5py.File(fpath, "r") as h5f:
            coords = AngularTransform(attrs)
            # read & save
            for i in range(0, lev_t):
                key = f"{i}"
                h5_re = h5f[f"{key}/{field_name}/re"]
                h5_im = h5f[f"{key}/{field_name}/im"]
                ret[g_re, i, ...] = h5_re[:, 0:max_lm]
                ret[g_im, i, ...] = h5_im[:, 0:max_lm]

            # transform from PITTNull coordinates to Spectre coordinates
            # assert not np.any(np.isnan(ret)), f"{field_name} has nans before transf.!"
            print(f"transforming {field_name} from pitt to spectre", flush=True)
            ret = coords.transform_pit_coeffs_to_spec_coeffs(ret)
            # assert not np.any(np.isnan(ret)), f"{field_name} got nans after transf.!"

    elif attrs["file_type"] == "bin":
        # Load the list of files
        # TODO: this depends on file name
        flist = sorted(glob.glob(fpath + "/cce_*.bin"))
        dat_real = []
        dat_imag = []
        t = []
        for f in flist:
            # TODO: it reads all field and only use one of them. it's inefficient
            (
                nr,
                num_l_modes,
                time,
                rin,
                rout,
                data_real,
                data_imag,
                index_to_lm,
            ) = read_cce_file(f)
            t.append(time)
            dat_real.append(data_real[:, g_name_index[field_name], :])
            dat_imag.append(data_imag[:, g_name_index[field_name], :])
        dat_real = np.array(dat_real)
        dat_imag = np.array(dat_imag)
        t = np.array(t)
        f_real = interp1d(t, dat_real, axis=0)
        f_imag = interp1d(t, dat_imag, axis=0)

        attrs["lev_t"] = len(flist)
        attrs["max_n"] = nr
        attrs["max_l"] = num_l_modes
        attrs["max_lm"] = int((num_l_modes + 1) ** 2)
        attrs["r_in"] = np.array([rin])
        attrs["r_out"] = np.array([rout])
        attrs["time"] = np.linspace(t.min(), t.max(), t.shape[0])

        shape = (
            len([g_re, g_im]),
            attrs["lev_t"],
            attrs["max_n"],
            attrs["max_lm"],
        )
        ret = np.empty(shape=shape, dtype=float)
        # NOTE: is it fine that we do interpolation?
        ret[g_re, :] = f_real(attrs["time"])
        ret[g_im, :] = f_imag(attrs["time"])
    else:
        raise ValueError("no such option")

    # print(ret)
    return ret


def lm_mode(ll, m):
    """
    ll and m mode convention
    """
    return ll * ll + ll + m


def read_cce_file(filename):
    """
    Reads binary data from a CCE output file generated by the provided C++ function.

    Parameters:
        filename (str): The path to the binary file.

    Returns:
        nr (int): Number of radial points.
        num_l_modes (int): Number of ll modes.
        time (float): Time value from the simulation.
        rin (float): Inner radial boundary.
        rout (float): Outer radial boundary.
        data_real (numpy.ndarray): Real part of the data, shape
                (nr, 10, num_angular_modes).
        data_imag (numpy.ndarray): Imaginary part of the data, shape
                (nr, 10, num_angular_modes).
        index_to_lm (dict): Mapping from data index to (ll, m) values.
    """
    with open(filename, "rb") as f:
        # Read number of radial points (nr)
        nr_bytes = f.read(4)
        nr = struct.unpack("<i", nr_bytes)[0]  # little-endian integer

        # Read number of ll modes (num_l_modes)
        num_l_modes_bytes = f.read(4)
        num_l_modes = struct.unpack("<i", num_l_modes_bytes)[0]

        # Read time
        time_bytes = f.read(8)
        time = struct.unpack("<d", time_bytes)[0]  # little-endian double

        # Read inner and outer radial boundaries (rin and rout)
        rin_bytes = f.read(8)
        rin = struct.unpack("<d", rin_bytes)[0]

        rout_bytes = f.read(8)
        rout = struct.unpack("<d", rout_bytes)[0]

        # Calculate the number of angular modes
        num_angular_modes = (num_l_modes + 1) * (num_l_modes + 1)

        # Total number of data points
        count = 10 * nr * num_angular_modes

        # Read data_real array
        data_real = np.fromfile(f, dtype="<d", count=count)  # little-endian double
        data_real = data_real.reshape((nr, 10, num_angular_modes))

        # Read data_imag array
        data_imag = np.fromfile(f, dtype="<d", count=count)
        data_imag = data_imag.reshape((nr, 10, num_angular_modes))

        # Create a mapping from index to (ll, m)
        index_to_lm = {}
        for ll in range(1, num_l_modes + 1):
            for m in range(-ll, ll + 1):
                index = ll * ll + ll + m
                index_to_lm[index] = (ll, m)

    return nr, num_l_modes, time, rin, rout, data_real, data_imag, index_to_lm


def get_attribute(
    fpath: str, field_name: str = "gxx", type: str = "h5", args=None
) -> dict:
    """
    find attributes such as num. of time level, and n, lm in C_nlm
    also saves the time value at each slice.
    """
    attrs = {}
    if type == "h5":
        attrs["file_type"] = "h5"
        with h5py.File(fpath, "r") as h5f:
            # find attribute about num. of time level, and n,ll,m in C_nlm
            attrs["lev_t"] = len(h5f.keys()) - 1
            if attrs["lev_t"] % 2 == 0:  # for fourier transformation
                attrs["lev_t"] -= 1

            attrs["max_n"], attrs["max_lm"] = h5f[f"1/{field_name}/re"].shape
            attrs["max_l"] = (
                int(math.sqrt(attrs["max_lm"])) - 1
            )  # NOTE:ll must be inclusive in loops
            attrs["max_lm"] = int((attrs["max_l"] + 1) ** 2)
            attrs["r_in"] = h5f["metadata"].attrs["Rin"]
            attrs["r_out"] = h5f["metadata"].attrs["Rout"]
            # read & save time
            time = []
            for i in range(0, attrs["lev_t"]):
                key = f"{i}"
                t = h5f[key].attrs["Time"][0]
                time.append(t)
            attrs["time"] = np.array(time)
    elif type == "bin":
        attrs["file_type"] = "bin"
    else:
        raise ValueError("no such option")

    # print(attrs)
    return attrs


def time_derivative_findiff(
    field: np.array, field_name: str, attrs: dict, args
) -> np.array:
    """
    return the time derivative of the given field using finite diff. 2nd order
    field(rel/img,t,n,lm)
    """

    print(f"finite difference time derivative: {field_name}", flush=True)
    _, len_t, len_n, len_lm = field.shape
    time = attrs["time"]
    dt = np.gradient(time, 2)
    dfield = np.empty_like(field)

    for n in range(len_n):
        for lm in range(len_lm):
            dfield[g_re, :, n, lm] = np.gradient(field[g_re, :, n, lm], 2) / dt
            dfield[g_im, :, n, lm] = np.gradient(field[g_im, :, n, lm], 2) / dt

    return dfield


def time_derivative_fourier(
    field: np.array, field_name: str, attrs: dict, args
) -> np.array:
    """
    return the time derivative of the given field using Fourier method
    field(rel/img,t, n,lm)
    """

    # TODO: Fourier time derives not tested!
    print(f"Fourier time derivative: {field_name}", flush=True)
    _, len_t, len_n, len_lm = field.shape
    dt = attrs["time"][2] - attrs["time"][1]
    wm = math.pi * 2.0 / (len_t * dt)

    dfield = np.empty_like(field)
    for n in range(len_n):
        for lm in range(len_lm):
            coeff = field[g_re, :, n, lm] + 1j * field[g_im, :, n, lm]
            # F. transform
            fft_coeff = np.fft.fft(coeff)

            # time derivative
            half = len_t // 2 + 1
            omega = np.empty(shape=half)
            for i in range(0, half):
                omega[i] = i * wm

            dfft_coeff = np.empty_like(fft_coeff)
            dfft_coeff[0] = 0

            dfft_coeff[1:half] = (
                -np.imag(fft_coeff[1:half]) + 1j * np.real(fft_coeff[1:half])
            ) * omega[1:]
            dfft_coeff[half:] = (
                np.imag(fft_coeff[half:]) - 1j * np.real(fft_coeff[half:])
            ) * (omega[::-1][1:])

            # not optimized version
            """
      dfft_coeff[0] = 0
      for i in range(1, half):
        omega = i * wm
        re = np.real(fft_coeff[i])
        im = np.imag(fft_coeff[i])
        re2 = np.real(fft_coeff[-i])
        im2 = np.imag(fft_coeff[-i])

        dfft_coeff[i] = omega*complex(-im, re)
        dfft_coeff[-i] = omega*complex(im2, -re2)

      """
            # F. inverse
            coeff = np.fft.ifft(dfft_coeff)
            dfield[g_re, :, n, lm] = np.real(coeff)
            dfield[g_im, :, n, lm] = np.imag(coeff)

    if args["debug"] == "y":
        for n in range(len_n):
            for ll in range(2, g_debug_max_l + 1):
                for m in range(-ll, ll + 1):
                    hfile = f"{args['d_out']}/debug_{field_name}_n{n}ll{ll}m{m}.txt"
                    write_data = np.column_stack(
                        (
                            attrs["time"],
                            dfield[g_re, :, n, lm_mode(ll, m)],
                            dfield[g_im, :, n, lm_mode(ll, m)],
                            field[g_re, :, n, lm_mode(ll, m)],
                            field[g_im, :, n, lm_mode(ll, m)],
                        )
                    )
                    np.savetxt(hfile, write_data, header="t dre/dt dim/dt re im")

    return dfield


def dUk_dx(order: int, x: float) -> float:
    """
    d(Chebyshev of second kind)/dx
    """
    assert x != 1 and x != -1
    t = special.chebyt(order + 1)(x)
    u = special.chebyu(order)(x)
    duk_dx = (order + 1) * t - x * u
    duk_dx /= x**2 - 1

    return duk_dx


def radial_derivative_at_r_chebu(
    field: np.array, field_name: str, attrs: dict, args
) -> np.array:
    """
    return the radial derivative of the given field using Chebyshev of
    2nd kind method at the radius of interest.

    f(x) = sum_{i=0}^{N-1} C_i U_i(x), U_i(x) Chebyshev of 2nd kind
    collocation points (roots of U_i): x_i = cos(pi*(i+1)/(N+1))
    x = g_sign*(2*r - r_1 - r_2)/(r_2 - r_1), notes: x != {1 or -1}

    field(rel/img,t,n,lm)
    """

    r_1 = attrs["r_in"][0]
    r_2 = attrs["r_out"][0]
    r = args["radius"]

    print(
        f"ChebyU radial derivative: {field_name}, at r={r} in [{r_1},{r_2}]",
        flush=True,
    )

    _, len_t, len_n, len_lm = field.shape

    assert r_1 != r_2
    dx_dr = g_sign * 2 / (r_2 - r_1)

    if args["debug"] == "y":
        # populate collocation points, roots of U_i
        x_i = np.empty(shape=len_n, dtype=float)
        for i in range(len_n):
            x_i[i] = math.cos(math.pi * (i + 1) / (len_n + 1))

        # dU_k/dx|x=x_i
        duk_dx = np.empty(shape=(len_n, len_n), dtype=float)
        for k in range(len_n):
            for i in range(len_n):
                t = special.chebyt(k + 1)(x_i[i])
                u = special.chebyu(k)(x_i[i])
                duk_dx[k, i] = (k + 1) * t - x_i[i] * u

        duk_dx /= np.square(x_i) - 1

        uk = np.empty(shape=len_n, dtype=float)
        tk = np.empty(shape=len_n, dtype=float)
        for k in range(len_n):
            hfile = f"{args['d_out']}/cheb_k{k}.txt"
            for i in range(len_n):
                tk[i] = special.chebyt(k)(x_i[i])
                uk[i] = special.chebyu(k)(x_i[i])

            write_data = np.column_stack((x_i, uk, tk, duk_dx[k, :]))
            np.savetxt(
                hfile,
                write_data,
                header=f"x_i uk{k}(x_i) tk{k}(x_i) duk{k}(x_i)/dx",
            )

    dfield = np.zeros(shape=(len([g_re, g_im]), len_t, len_lm))
    r = args["radius"]
    x = g_sign * (2 * r - r_1 - r_2) / (r_2 - r_1)
    for k in range(len_n):
        dfield[:, :, :] += field[:, :, k, :] * dUk_dx(k, x)

    return dfield * dx_dr


def time_derivative(field: np.array, field_name: str, attrs: dict, args):
    """
    return the time derivative of the given field
    """

    if args["t_deriv"] == "Fourier":
        return time_derivative_fourier(field, field_name, attrs, args)
    elif args["t_deriv"] == "fin_diff":
        return time_derivative_findiff(field, field_name, attrs, args)
    else:
        raise ValueError("no such option")


class ChebUExpansion:
    """
    f(x) = sum_{i=0}^{N-1} C_i U_i(x), U_i(x) Chebyshev of 2nd kind
    collocation points (roots of U_i): x_i = cos(pi*(i+1)/(N+1))
    """

    def __init__(self, attrs: dict, args: dict):
        self.N = N = attrs["max_n"]  # num. of coeffs or num of collocation pnts
        pi = math.pi

        # U_N roots:
        a_i = np.arange(0, N, dtype=int) + 1
        self.x_i = x_i = np.cos(math.pi * a_i / (N + 1))
        # quadrature weights
        self.w_i = (1 - np.square(x_i)) * pi / (N + 1)

        # ChebU_j(x_i):
        Uj_xi = np.empty(shape=(N, N), dtype=float)
        for j in range(N):
            Uj_xi[j, :] = special.chebyu(j)(x_i)
        self.Uj_xi = Uj_xi

        if args["debug"] == "y":
            self.__debug()

    def coefficients(self, field: np.array) -> np.array:
        """
        use Gauss Quadrature to expand the given field in ChebU
        f(x) = sum_{i=0}^{N-1} C_i U_i(x), U_i(x) Chebyshev of 2nd kind
        collocation points (roots of U_i): x_i = cos(pi*(i+1)/(N+1))
        x = g_sign*(2*r - r_1 - r_2)/(r_2 - r_1), notes: x != {1 or -1}

        """

        w_i = self.w_i
        Uji = self.Uj_xi

        coeffs = Uji @ (w_i * field[::-g_sign])
        coeffs *= 2.0 / math.pi

        return coeffs

    def __debug(self):
        """
        test if the expansion works for different known functions
        """

        N = self.N + 1  # +1 to see the roots of U_N
        x_i = self.x_i

        # populate funcs using bases themselves
        fs = []
        for n in range(N):
            f = special.chebyu(n)(x_i)
            fs.append((f, n))

        # find coeffs
        cs = []
        for n in range(N):
            c = self.coefficients(fs[n][0])
            cs.append((c, fs[n][1]))

        # expect to see only the n-th entry is 1 for the chebyu(n) of order n
        for n in range(N):
            order = cs[n][1]
            print(f"chebyu{order}, coeffs = {cs[n][0]}\n", flush=True)


def radial_expansion_chebu(
    field: np.array, field_name: str, attrs: dict, args
) -> np.array:
    """
    expands field in the radial direction using chebyshev of second kind
    and returns the expansion coefficients. we have

    f(x) = sum_{i=0}^{N-1} C_i U_i(x), U_i(x) Chebyshev of 2nd kind
    collocation points (roots of U_i): x_i = cos(pi*(i+1)/(N+1))
    x = g_sign*(2*r - r_1 - r_2)/(r_2 - r_1), notes: x != {1 or -1}

    field(rel/img,t,n,lm)

    """

    print(f"ChebU expansion {field_name}", flush=True)
    expand = ChebUExpansion(attrs, args)
    _, len_t, _, len_lm = field.shape

    for t in range(len_t):
        for lm in range(len_lm):
            field[g_re, t, :, lm] = expand.coefficients(field[g_re, t, :, lm])
            field[g_im, t, :, lm] = expand.coefficients(field[g_im, t, :, lm])

    return field


def radial_expansion(field: np.array, field_name: str, attrs: dict, args):
    """
    expands field in the radial direction(if needed) and returns
    """

    if args["ftype"] == "bin":
        return radial_expansion_chebu(field, field_name, attrs, args)
    elif args["ftype"] == "h5":
        return field
    else:
        raise ValueError("no such option")


def radial_derivative_at_r(field: np.array, field_name: str, attrs: dict, args):
    """
    return the radial derivative of the given field at R=r
    """

    if args["r_deriv"] == "ChebU":
        return radial_derivative_at_r_chebu(field, field_name, attrs, args)
    else:
        raise ValueError("no such option")


class Interpolate_at_r:
    def __init__(self, attrs: dict, args: dict):
        """
        interpolate the given field at R=r
        """
        self.attrs = attrs
        self.args = args
        self.len_t = attrs["lev_t"]
        self.len_n = attrs["max_n"]
        self.len_lm = attrs["max_lm"]
        r_1 = attrs["r_in"][0]
        r_2 = attrs["r_out"][0]
        self.r = r = args["radius"]
        self.x = g_sign * (2 * r - r_1 - r_2) / (r_2 - r_1)

        assert -1 < self.x < 1, f"x = {self.x}"

        if args["interpolation"] == "ChebU":
            self.Uk = np.empty(shape=self.len_n)
            for k in range(self.len_n):
                self.Uk[k] = special.chebyu(k)(self.x)
            # print("x,Uk",self.x,self.Uk)
            self.interp = self.interpolate_at_r_chebu
        else:
            raise ValueError("no such option")

    def interpolate_at_r_chebu(self, field: np.array, field_name: str):
        """
        interpolate at R=r using Cheb U.
        """
        print(f"Interpolating at R={self.r}: {field_name}", flush=True)

        field_r = np.zeros(shape=(len([g_re, g_im]), self.len_t, self.len_lm))
        for k in range(self.len_n):
            field_r[:, :, :] += field[:, :, k, :] * self.Uk[k]

        return field_r

    def interpolate(self, field: np.array, field_name: str):
        return self.interp(field, field_name)


def process_field(field_name: str) -> dict:
    """
    - read data
    - find time derives
    - find radial derives
    - interpolate at R=r
    """

    # return
    attrs = g_attrs
    args = g_args
    db = {}

    # load data
    field = load(args["fpath"], field_name, attrs)
    # db[f"{field_name}"] = field

    # radial expansion
    field = radial_expansion(field, field_name, attrs, args)

    # time derivative
    dfield_dt = time_derivative(field, field_name, attrs, args)

    # interpolate at a specific radii
    interpolate = Interpolate_at_r(attrs, args)
    field_at_r = interpolate.interpolate(field, field_name)
    db[f"{field_name}|r"] = field_at_r

    dfield_dt_at_r = interpolate.interpolate(dfield_dt, f"d{field_name}/dt")
    db[f"d{field_name}/dt|r"] = dfield_dt_at_r

    # radial derivative at R=r
    dfield_dr_at_r = radial_derivative_at_r(field, field_name, attrs, args)
    db[f"d{field_name}/dr|r"] = dfield_dr_at_r

    return db


def h5_create_group(h5file, group_name: str):
    """
    create a group for h5
    """
    h5group = None

    # create group if not exists
    if h5file.get(group_name, default=None) is None:
        h5group = h5file.create_group(group_name)
    else:
        raise ValueError("this group {group_name} is already exists.")

    return h5group


def h5_write_data(h5file, data: np.array, data_name: str, attrs: dict, args: dict):
    """
    reminder:
      data[real/imag, time_level, lm]

    write syntax, eg:

    h5["gxx.dat"] =
      [time_level, ['time', 'gxx_Re(0,0)', 'gxx_Im(0,0)',
      'gxx_Re(1,1)', 'gxx_Im(1,1)', ...] ]

    h5["gxx.dat"].attrs['Legend'] = the associated column =
      array(['time', 'gxx_Re(0,0)', 'gxx_Im(0,0)', 'gxx_Re(1,1)', 'gxx_Im(1,1)', ...])

    # => h5["gxx.dat"][3,0] = value of time at the dump level 3
    # => h5["gxx.dat"][4,1] = value of gxx_Re(0,0) at the dump level 4

    """

    dataset_conf = dict(
        name=f"{data_name}",
        shape=(
            attrs["lev_t"],
            len([g_re, g_im]) * (attrs["max_l"] + 1) ** 2 + 1,  # the last +1 for time
        ),
        dtype=float,  # chunks=True,
        # compression="gzip",
        # shuffle=True,
    )

    data_attrs = ["time"]

    if args["debug"] == "y":
        print(dataset_conf, flush=True)

    h5file.create_dataset(**dataset_conf)

    flat = 0
    h5file[f"{data_name}"][:, flat] = attrs["time"]
    flat += 1
    for ll in range(0, attrs["max_l"] + 1):
        for m in range(ll, -ll - 1, -1):
            assert not np.any(np.isnan(data[g_re, :, lm_mode(ll, m)]))
            assert not np.any(np.isnan(data[g_im, :, lm_mode(ll, m)]))

            data_attrs.append(f"{data_name[:-4]}_Re({ll},{m})")
            data_attrs.append(f"{data_name[:-4]}_Im({ll},{m})")
            h5file[f"{data_name}"][:, flat] = data[g_re, :, lm_mode(ll, m)]
            h5file[f"{data_name}"][:, flat + 1] = data[g_im, :, lm_mode(ll, m)]
            flat += 2
    Legend = [s.encode("ascii", "ignore") for s in data_attrs]
    h5file[f"{data_name}"].attrs["Legend"] = Legend


def write(f: str, db: dict, attrs: dict, args: dict):
    """
    write data on disk
    """
    print(f"writing: {f}", flush=True)

    field_name = g_name_map[f"{f}"]
    field_name_key = f"{field_name}.dat"
    dfield_name_dr_key = f"Dr{field_name}.dat"
    dfield_name_dt_key = f"Dt{field_name}.dat"

    r = args["radius"]
    file_name = os.path.join(args["d_out"], f"CceR{r:07.2f}.h5")
    with h5py.File(file_name, "a") as h5file:
        name = field_name_key
        data = db[f"{f}|r"]
        h5_write_data(h5file, data, name, attrs, args)

        name = dfield_name_dr_key
        data = db[f"d{f}/dr|r"]
        h5_write_data(h5file, data, name, attrs, args)

        name = dfield_name_dt_key
        data = db[f"d{f}/dt|r"]
        h5_write_data(h5file, data, name, attrs, args)


def main(args):
    """
    create output required by Specter code
    ref: https://spectre-code.org/tutorial_cce.html
    """

    global g_attrs
    global g_args
    global g_sign

    g_args = args
    """
    note:
    in Pitnull:
    r = 0.5[(r_2 - r_1)*x + (r_2 + r_1)], r_2 > r_1 & x = [-1,1]
    => x = -1 -> r = r_1
    => x = +1 -> r = r_2

    in AthenaK:
    r = 0.5[(r_1 - r_2)*x + (r_2 + r_1)], r_2 > r_1 & x = [1,-1]
    => x = -1 -> r = r_2
    => x = +1 -> r = r_1

  """
    g_sign = int(-1) if args["ftype"] == "bin" else int(1)

    # check if output dir exist, if not, mkdir
    if not os.path.exists(g_args["d_out"]):
        os.makedirs(g_args["d_out"])
    # find attribute for an arbitrary field
    g_attrs = get_attribute(args["fpath"], type=args["ftype"])

    # for each field
    # I'm afraid this method takes too much memory
    # from multiprocessing import Pool
    # with Pool(processes=len(g_field_names)) as p:
    #  db = p.map(process_field, g_field_names)

    for f in g_field_names:
        db = process_field(f)
        # write on disk
        write(f, db, g_attrs, g_args)


if __name__ == "__main__":
    args = parse_cli()
    main(args.__dict__)
