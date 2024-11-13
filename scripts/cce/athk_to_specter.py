#!/usr/bin/env python3
## Alireza Rashti - Oct 2024 (C)
## usage:
## $ ./me -h
##

import sys
import os
import numpy as np
import math
import argparse
import re
import h5py

# from itertools import product

# import matplotlib.pyplot as plt
# import glob
# import sympy
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
      "-t_deriv",
      type=str,
      default="Fourier",
      help="method to take the time derivative of fields:{Fourier}",
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


def time_derivative_fourier(field: np.array, attr: dict, args) -> np.array:
  """
    return the time derivative of the given field using Fourier method
    field(t,rel/img,n,lm)
    """

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
      fft_coeff[0] = complex(0)
      for i in range(1, len_t // 2 + 1):
        omega = i * wm
        re = np.real(fft_coeff[i])
        im = np.imag(fft_coeff[i])
        re2 = np.real(fft_coeff[-i])
        im2 = np.imag(fft_coeff[-i])

        fft_coeff[i] = complex(-im * omega, re * omega)
        fft_coeff[-i] = complex(-im2 * omega, re2 * omega)

      # F. inverse
      coeff = np.fft.ifft(fft_coeff)
      dfield[g_re, :, n, lm] = np.real(coeff)
      dfield[g_im, :, n, lm] = np.imag(coeff)

  print(dfield)


def time_derivative(field_name: str, db: dict, args):
  """
    return the time derivative of the given field
    """

  if args["t_deriv"] == "Fourier":
    return time_derivative_fourier(field_name, db, args)
  else:
    raise ValueError("no such option")


def main(args):
  """
    create output required by Specter code
    ref: https://spectre-code.org/tutorial_cce.html
    """

  # find attribute for an arbitrary field
  attr = get_attribute(args["f_h5"])

  # for each field
  field_name = "gxx"

  # load data
  field = load(args["f_h5"], field_name, attr)

  # print(dat)

  # time derivative
  dfield_dt = time_derivative(field, attr, args)

  # radial derivative

  # save


if __name__ == "__main__":
  args = parse_cli()
  main(args.__dict__)
