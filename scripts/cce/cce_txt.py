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
import re
import h5py

# import matplotlib.pyplot as plt
# import glob
# import sympy
## ---------------------------------------------------------------------- ##


def parse_cli():
  """
    arg parser
    """
  p = argparse.ArgumentParser(description="plotting cce spectre output")
  p.add_argument("-f",
                 type=str,
                 required=True,
                 help="/path/to/specter/cce/output.h5")
  p.add_argument("-dout",
                 type=str,
                 required=True,
                 help="/path/to/output/file.txt")
  p.add_argument(
      "-field",
      type=str,
      default="Psi4",
      help="field names:[News,Psi0,Psi1,Psi2,Psi3,Psi4,Strain]",
  )
  p.add_argument(
      "-mode",
      type=str,
      default="Y_2,2",
      help='field modes:["Y_2,-2","Y_2,-2", ...]',
  )
  args = p.parse_args()
  return args


def find_h5_1mode(h5f, field_name, mode_name, args):
  mode = 0
  flag = False
  for m in h5f[field_name].attrs["Legend"]:
    if m.find(mode_name) != -1:
      print("found mode for", field_name, m, mode_name)
      flag = True
      break
    mode += 1

  assert flag == True
  return mode


def read_save_to_txt(args):

  with h5py.File(args["f"], "r") as h5f:
    # find group
    keys = h5f.keys()
    group = [k for k in keys if k.find("Spectre") != -1][0]
    group = "/" + group + "/"
    dataset = group+args["field"]
    mode_re = find_h5_1mode(h5f, dataset, "Real "+args["mode"], args)
    mode_im = find_h5_1mode(h5f, dataset, "Imag "+args["mode"], args)
    t = h5f[dataset][:, 0]
    re = h5f[dataset][:, mode_re]
    im = h5f[dataset][:, mode_im]

  mode_txt = args["mode"].replace(",", "_")
  mode_txt = mode_txt.replace(" ", "")
  out = os.path.join(args["dout"], args["field"] + "_" + mode_txt + ".txt")
  #print(out)
  stack = np.column_stack((t, re, im))
  np.savetxt(out,
             stack,
             header=f'# t re{args["mode"]} im{args["mode"]}',
             fmt="%.16e %.16e %.16e")

def main(args):
  """
    read cce spectre file
    """

  read_save_to_txt(args)


if __name__ == "__main__":
  args = parse_cli()
  main(args.__dict__)
