#!/usr/bin/env python3
## Alireza Rashti - Oct 2024 (C)
## usage:
## $ ./me -h
##

import sys
import os
import numpy as np
from scipy import special
import math
import argparse
import h5py
from multiprocessing import Pool

# from itertools import product
# import matplotlib.pyplot as plt
# import glob
# import sympy
# import re

## ---------------------------------------------------------------------- ##

## field names
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

## real/imag
g_re = 0
g_im = 1

## args
g_args = None
g_attr = None

## debug
g_debug_max_l = 2


def parse_cli():
  """
    arg parser
    """
  p = argparse.ArgumentParser(
      description="convert Athenak CCE dumps to Spectre CCE")
  p.add_argument("-f_h5", type=str, required=True, help="/path/to/cce/h5/dumps")
  p.add_argument("-d_out", type=str, required=True, help="/path/to/output/dir")
  p.add_argument("-debug", type=str, default="y", help="debug=[y,n]")
  p.add_argument(
      "-radius",
      type=float,
      required=True,
      help="interpolate all fields and their derivatives at this radius.",
  )
  p.add_argument(
      "-t_deriv",
      type=str,
      default="Fourier",
      help="method to take the time derivative of fields:{Fourier}",
  )
  p.add_argument(
      "-r_deriv",
      type=str,
      default="ChebU",
      help=
      "method to take the radial derivative of fields:{ChebU:Chebyshev of second kind}",
  )
  p.add_argument(
      "-interpolation",
      type=str,
      default="ChebU",
      help=
      "method to interpolate fields at a given r:{ChebU:Chebyshev of second kind}",
  )

  args = p.parse_args()
  return args


def load(fpath: str, field_name: str, attr: dict) -> list:
  """
    read the field accroding to attr.
    return convention:
      ret[real/imag, time_level, n, lm], eg:
      ret[g_re,3,2,:] = Re(C_2lm(t=3)) for all lm
      ret[g_im,3,2,:] = Im(C_2lm(t=3)) for all lm
    """

  if attr["file_type"] == "h5":
    lev_t = attr["lev_t"]
    max_n = attr["max_n"]
    max_lm = attr["max_lm"]
    shape = (len([g_re, g_im]), lev_t, max_n, max_lm)
    ret = np.empty(shape=shape, dtype=float)
    with h5py.File(fpath, "r") as h5f:
      # read & save
      for i in range(0, lev_t):
        key = f"{i}"
        h5_re = h5f[f"{key}/{field_name}/re"]
        h5_im = h5f[f"{key}/{field_name}/im"]
        ret[g_re, i, :] = h5_re
        ret[g_im, i, :] = h5_im

  else:
    raise ValueError("no such option")

  # print(ret)
  return ret


def lm_mode(l, m):
  """
    l and m mode convention
    """
  return l * l + l + m


def get_attribute(fpath: str,
                  field_name: str = "gxx",
                  type: str = "h5",
                  args=None) -> dict:
  """
    find attributes such as num. of time level, and n, lm in C_nlm
    also saves the time value at each slice.
    """
  attr = {}
  if type == "h5":
    attr["file_type"] = "h5"
    with h5py.File(fpath, "r") as h5f:
      # find attribute about num. of time level, and n,l,m in C_nlm
      attr["lev_t"] = len(h5f.keys()) - 1
      attr["max_n"], attr["max_lm"] = h5f[f"1/{field_name}/re"].shape
      attr["r_in"] = h5f["metadata"].attrs["Rin"]
      attr["r_out"] = h5f["metadata"].attrs["Rout"]
      # read & save time
      time = []
      for i in range(0, attr["lev_t"]):
        key = f"{i}"
        t = h5f[key].attrs["Time"][0]
        time.append(t)

      attr["time"] = np.array(time)

  else:
    raise ValueError("no such option")

  # print(attr)
  return attr


def time_derivative_fourier(field: np.array, field_name: str, attr: dict,
                            args) -> np.array:
  """
    return the time derivative of the given field using Fourier method
    field(t,rel/img,n,lm)
    """

  print(f"Fourier time derivative: {field_name}", flush=True)
  _, len_t, len_n, len_lm = field.shape
  dt = attr["time"][2] - attr["time"][1]
  wm = math.pi * 2.0 / (len_t * dt)

  dfield = np.empty_like(field)
  for n in range(len_n):
    for lm in range(len_lm):
      coeff = field[g_re, :, n, lm] + 1j * field[g_im, :, n, lm]
      # F. transform
      fft_coeff = np.fft.fft(coeff)
      # if args["debug"] == 'y':
      #  print("debug: normalization?",round(coeff[1],6) == round(np.fft.ifft(fft_coeff)[1],6))

      # time derivative
      half = len_t // 2 + 1
      omega = np.empty(shape=half)
      for i in range(0, half):
        omega[i] = i * wm

      dfft_coeff = np.empty_like(fft_coeff)
      dfft_coeff[0] = 0
      dfft_coeff[1:half] = (-np.imag(fft_coeff[1:half]) +
                            1j * np.real(fft_coeff[1:half])) * omega[1:]
      dfft_coeff[half:] = (np.imag(fft_coeff[half:]) -
                           1j * np.real(fft_coeff[half:])) * omega[::-1][1:]

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
      for l in range(2, g_debug_max_l + 1):
        for m in range(-l, l + 1):
          hfile = (f"{args['d_out']}/debug_{field_name}_n{n}l{l}m{m}.txt")
          write_data = np.column_stack((
              attr["time"],
              dfield[g_re, :, n, lm_mode(l, m)],
              dfield[g_im, :, n, lm_mode(l, m)],
              field[g_re, :, n, lm_mode(l, m)],
              field[g_im, :, n, lm_mode(l, m)],
          ))
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


def radial_derivative_at_r_chebu(field: np.array,
                                 field_name: str,
                                 attr: dict,
                                 args) -> np.array:
  """
    return the radial derivative of the given field using Chebyshev of
    2nd kind method at the radius of interest.

    f(x) = sum_{i=0}^{N-1} C_i U_i(x), U_i(x) Chebyshev of 2nd kind
    collocation points (roots of U_i): x_i = cos(pi*(i+1)/(N+1))
    x = (2*r - r_1 - r_2)/(r_2 - r_1), notes: x != {1 or -1}

    field(t,rel/img,n,lm)
    """

  print(f"ChebyU radial derivative: {field_name}", flush=True)
  _, len_t, len_n, len_lm = field.shape

  r_1 = attr["r_in"][0]
  r_2 = attr["r_out"][0]
  assert r_1 != r_2
  dx_dr = 2 / (r_2 - r_1)

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
  x = (2 * r - r_1 - r_2) / (r_2 - r_1)
  for k in range(len_n):
    dfield[:, :, :] += field[:, :, k, :] * dUk_dx(k, x)

  return dfield * dx_dr


def time_derivative(field: np.array, field_name: str, attr: dict, args):
  """
    return the time derivative of the given field
    """

  if args["t_deriv"] == "Fourier":
    return time_derivative_fourier(field, field_name, attr, args)
  else:
    raise ValueError("no such option")


def radial_derivative_at_r(field: np.array, field_name: str, attr: dict, args):
  """
    return the radial derivative of the given field at R=r
    """

  if args["r_deriv"] == "ChebU":
    return radial_derivative_at_r_chebu(field, field_name, attr, args)
  else:
    raise ValueError("no such option")


class Interpolate_at_r:

  def __init__(self, attr: dict, args: dict):
    """
        interpolate the given field at R=r
        """
    self.attr = attr
    self.args = args
    self.len_t = attr["lev_t"]
    self.len_n = attr["max_n"]
    self.len_lm = attr["max_lm"]
    r_1 = attr["r_in"][0]
    r_2 = attr["r_out"][0]
    self.r = r = args["radius"]
    self.x = (2 * r - r_1 - r_2) / (r_2 - r_1)

    if args["interpolation"] == "ChebU":
      self.Uk = np.empty(shape=self.len_n)
      for k in range(self.len_n):
        self.Uk[k] = dUk_dx(k, self.x)
      self.interp = self.interpolate_at_r_chebu
    else:
      raise ValueError("no such option")

  def interpolate_at_r_chebu(self, field: np.array, field_name: str):
    """
        interpolate at R=r using Cheb U.
        """
    print(f"Interpolating at R={self.r}: {field_name}", flush=True)

    dfield = np.zeros(shape=(len([g_re, g_im]), self.len_t, self.len_lm))
    for k in range(self.len_n):
      dfield[:, :, :] += field[:, :, k, :] * self.Uk[k]

    return dfield

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
  attr = g_attr
  args = g_args
  db = {}

  # load data
  field = load(args["f_h5"], field_name, attr)
  db[f"{field_name}"] = field

  # time derivative
  dfield_dt = time_derivative(field, field_name, attr, args)

  # interpolate at a specific radii
  interpolate = Interpolate_at_r(attr, args)
  field_at_r = interpolate.interpolate(field, field_name)
  db[f"{field_name}|r"] = field_at_r

  dfield_dt_at_r = interpolate.interpolate(dfield_dt, f"d{field_name}/dt")
  db[f"d{field_name}/dt|r"] = dfield_dt_at_r

  # radial derivative at R=r
  dfield_dr_at_r = radial_derivative_at_r(field, field_name, attr, args)
  db[f"d{field_name}/dr|r"] = dfield_dr_at_r

  return db


def main(args):
  """
    create output required by Specter code
    ref: https://spectre-code.org/tutorial_cce.html
    """

  # find attribute for an arbitrary field
  global g_attr
  global g_args

  g_args = args
  g_attr = get_attribute(args["f_h5"])

  # for each field
  with Pool as p:
    results = p.map(process_field, g_field_names)

  print(results)

  # process_field(field_name, attr, args)


if __name__ == "__main__":
  args = parse_cli()
  main(args.__dict__)
