#!/usr/bin/env python3
## Alireza Rashti - Jan 2025 (C)
## usage:
## $ ./me -h
##

import sys
import os
import numpy as np
import math as m
import argparse
import h5py

# import matplotlib.pyplot as plt
# import glob
# import sympy
## ---------------------------------------------------------------------- ##

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


def parse_cli():
  """
    arg parser
    """
  p = argparse.ArgumentParser(description="debugging athk_to_spectre.py")
  p.add_argument(
      "-debug",
      type=str,
      default="plot_simple",
      help="debug type=[plot_simple]",
  )
  p.add_argument(
      "-fpath",
      type=str,
      required=True,
      help="path/to/output/athk_to_spectre.py/h5",
  )
  p.add_argument(
      "-field_name",
      type=str,
      default="gxx",
      help="plot for this field [gxx,gxy,...]",
  )
  p.add_argument(
      "-field_mode",
      type=str,
      default="Re_(2,2)",
      help="plot this mode[Re_(l,m),Im_(l,m)]",
  )

  args = p.parse_args()
  return args


def find_h5_mode(h5f, field_name, mode_name, args):
  mode = 0
  flag = False
  for m in h5f[field_name].attrs["Legend"]:
    if m.find(mode_name) != -1:
      flag = True
      break
    mode += 1

  assert flag == True
  return mode


def read_h5_mode_and_derivs(args):

  field_name = g_name_map[f"{field_name}"]
  field_name_key = f"{field_name}.dat"
  dfield_name_dr_key = f"Dr{field_name}.dat"
  dfield_name_dt_key = f"Dt{field_name}.dat"

  with h5py.File(args["fpath"], "r") as h5f:
    h5file[f"{data_name}"].attrs["Legend"] = Legend

    mode = find_h5_mode(h5f, f"{field_name_key}", args["field_mode"], args)
    f = h5f[f"{field_name_key}"][:, mode]

    mode = find_h5_mode(h5f, f"{dfield_name_dr_key}", args["field_mode"], args)
    drf = h5f[f"{dfield_name_dr_key}"][:, mode]

    mode = find_h5_mode(h5f, f"{dfield_name_dt_key}", args["field_mode"], args)
    dtf = h5f[f"{dfield_name_dt_key}"][:, mode]

  return f, drf, dtf


def debug_plot_simple(args):
  dat = read_h5_mode_and_derivs(args)
  plot_simple_live(dat, args)


def main(args):
  """
    debug
    """

  if args["debug"] == "plot_simple":
    debug_plot_simple(args)
  else:
    raise ValueError("no such option")


if __name__ == "__main__":
  args = parse_cli()
  main(args.__dict__)
