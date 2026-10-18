#!/usr/bin/env python3
"""Compile direct native GRMHD Jacobian tests using an existing serial build."""
import argparse,subprocess,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();root=Path(__file__).resolve().parents[2]
flags={}
for line in (b/'src/CMakeFiles/athena.dir/flags.make').read_text().splitlines():
 if ' = ' in line:
  key,value=line.split(' = ',1);flags[key]=shlex.split(value)
objs=[b/'src/CMakeFiles/athena.dir'/name for name in ['parameter_input.cpp.o','outputs/io_wrapper.cpp.o','globals.cpp.o','eos/primitive-solver/unit_system.cpp.o']]
libs=[b/name for name in ['kokkos/containers/src/libkokkoscontainers.a','kokkos/core/src/libkokkoscore.a','kokkos/simd/src/libkokkossimd.a']]
exe=b/'characteristic-native-modes'
cmd=['c++','-std=c++17','-O2',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],str(root/'scripts/tests/characteristic_native_modes.cpp'),*map(str,objs),*map(str,libs),'-o',str(exe)]
subprocess.run(cmd,check=True,timeout=120);subprocess.run([str(exe)],check=True,timeout=30)
