#!/usr/bin/env python3
"""oneAPI 2026.1 workaround for host physics registration and destruction.

This file launches no kernels. Compile it with the SYCL headers/types but skip
its pathological device-side recursive check of CCE/mesh pointer containers.
All kernel translation units retain the original compiler arguments.
"""
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
if any(x.endswith('/mesh/meshblock_pack.cpp') for x in args):
    args = [x for x in args if not x.startswith(('-fsycl', '-fno-sycl'))]
    compiler = Path(shutil.which(args[0]) or args[0]).resolve()
    include = compiler.parent.parent / 'include'
    if not (include / 'sycl/sycl.hpp').is_file():
        raise SystemExit(f'SYCL headers unavailable at {include}')
    args += ['-DSYCL_LANGUAGE_VERSION=202001', '-isystem', str(include)]
os.execvp(args[0], args)
