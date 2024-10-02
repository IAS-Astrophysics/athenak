#!/usr/bin/env python3
## Alireza Rashti - Oct 2024 (C)
## usage:
## $ ./me -h
##

import sys
import os
import numpy as np
import math as m
import argparse
import re
# import matplotlib.pyplot as plt
# import glob
# import sympy
## ---------------------------------------------------------------------- ##


def parse_cli():
  """
    arg parser
    """
  p = argparse.ArgumentParser(description="convert Athenak CCE dumps to Specter CCE")
  p.add_argument("-f_h5", type=str, required=True, help="/path/to/cce/h5/dumps")

  args = p.parse_args()
  return args


def main(args):
  """
    create output required by Specter code
    ref: https://spectre-code.org/tutorial_cce.html
    """
  
  # load data
  
  # time derivative
  
  # radial derivative

if __name__ == "__main__":
  args = parse_cli()
  main(args.__dict__)

