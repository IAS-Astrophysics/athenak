#!/usr/bin/env python3
"""Test the exact production C++ matrix kernels against independent NumPy eigensystems."""
import ctypes,json,subprocess,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
s=(ROOT/'src/dyn_grmhd/characteristic_reservoir.hpp').read_text()
s=s.split('namespace char_reservoir {',1)[1].split('template<int ivx',1)[0]
code='#include <cmath>\nusing Real=double;\n#define KOKKOS_INLINE_FUNCTION inline\n'+s+'\nextern "C" int sign7(double *a){return Sign(reinterpret_cast<double (*)[7]>(a));}\n'
rng=np.random.default_rng(62714);worst=0.;cases=0
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'matrix.cpp').write_text(code)
 subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(p/'matrix.cpp'),'-o',str(p/'matrix.so')],check=True)
 lib=ctypes.CDLL(str(p/'matrix.so'));fun=lib.sign7;fun.argtypes=[ctypes.POINTER(ctypes.c_double)];fun.restype=ctypes.c_int
 for shift in [-11.3137,-.2,-.03,0,.03,.2,11.3137]:
  for trial in range(20):
   R=np.eye(7)+rng.normal(size=(7,7))*.1;L=np.linalg.inv(R)
   eig=np.array([-.09,-.07,-.04,0,.04,.07,.09])-shift
   A=R@np.diag(eig)@L; scale=max(abs(A).max(),1e-8)
   expected=R@np.diag(np.sign(eig+1e-8*scale))@L
   actual=np.array(A,order='C');assert fun(actual.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))==1
   error=float(abs(actual-expected).max());worst=max(worst,error);assert error<1e-6,(shift,error)
   cases+=1
print(json.dumps(dict(cases=cases,max_absolute_error=worst,status='passed')))
