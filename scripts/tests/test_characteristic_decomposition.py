#!/usr/bin/env python3
"""Compile native symbols, independently decompose all seven RMHD families."""
import argparse,json,subprocess,shlex,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from characteristic_decomposition import speeds,decompose,outgoing_projector

def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();b=a.build.resolve();root=Path(__file__).resolve().parents[2];a.out.parent.mkdir(parents=True,exist_ok=True)
 flags={}
 for line in (b/'src/CMakeFiles/athena.dir/flags.make').read_text().splitlines():
  if ' = ' in line:
   k,v=line.split(' = ',1);flags[k]=shlex.split(v)
 objs=[b/'src/CMakeFiles/athena.dir'/name for name in ['parameter_input.cpp.o','outputs/io_wrapper.cpp.o','globals.cpp.o','eos/primitive-solver/unit_system.cpp.o']]
 libs=[b/name for name in ['kokkos/containers/src/libkokkoscontainers.a','kokkos/core/src/libkokkoscore.a','kokkos/simd/src/libkokkossimd.a']]
 exe=b/'characteristic-native-spectrum';raw=b/'characteristic-native-spectrum.jsonl'
 subprocess.run(['c++','-std=c++17','-O2',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],str(root/'scripts/tests/characteristic_native_spectrum.cpp'),*map(str,objs),*map(str,libs),'-o',str(exe)],check=True,timeout=120)
 with raw.open('w') as f:subprocess.run([str(exe)],stdout=f,check=True,timeout=120)
 worst=dict(eigen_residual=0.,inverse_error=0.,reconstruction_error=0.,sign_projector_error=0.,dispersion_error=0.,condition=0.)
 count=degenerate=compared=0;examples=[];rng=np.random.default_rng(1701)
 for line in raw.read_text().splitlines():
  if not line.startswith('{'):continue
  r=json.loads(line);assert r['sign_ok'], 'Native matrix sign failed';gg=r['g'];g=np.array([[gg[0],gg[1],gg[2]],[gg[1],gg[3],gg[4]],[gg[2],gg[4],gg[5]]]);A=np.asarray(r['A']);d=r['axis'];light=r['alpha']*np.sqrt(np.linalg.inv(g)[d,d])
  lam=speeds(r['q'],r['bn'],g,r['alpha'],r['beta'],d);basis=decompose(A,lam,light);R=basis['R'];L=basis['L']
  numerical=np.linalg.eigvals(A);error=max(abs(np.sort(numerical.real)-np.sort(lam)))/light
  assert max(abs(numerical.imag))/light<2e-5 and error<2e-5,(r,error,numerical)
  inverse=float(abs(L@R-np.eye(7)).max());delta=rng.normal(size=7);reconstruction=float(abs(R@(L@delta)-delta).max())
  assert inverse<1e-8 and reconstruction<1e-8
  for outward in (-1,1):
   projector=outgoing_projector(basis,outward)
   assert np.max(abs(projector@projector-projector))<1e-8
  assert np.min(lam)>=-r['beta'][d]-light-1e-5*light
  assert np.max(lam)<=-r['beta'][d]+light+1e-5*light
  degenerate+=any(len(group['indices'])>1 for group in basis['groups'])
  # Native sign shifts true stationary modes outward by 1e-8 of its matrix norm.
  # Match only unambiguous spectra; tiny finite-difference glancing displacements
  # do not have a physically meaningful sign at a zero-speed contact.
  norm=max(abs(A).max(),1e-8);safe=min(abs(lam))>1e-6*max(norm,light)
  signerror=0.
  if safe and r['sign_ok']:
   signerror=float(abs(outgoing_projector(basis)-.5*(np.eye(7)+r['sign'])).max());assert signerror<2e-4,(r,signerror);compared+=1
  for key,value in [('eigen_residual',basis['residual']),('inverse_error',inverse),('reconstruction_error',reconstruction),('sign_projector_error',signerror),('dispersion_error',error),('condition',basis['condition'])]:worst[key]=max(worst[key],value)
  if r['axis']==2 and r['metric']==0 and r['flow']==0 and r['alpha']==1.3 and r['beta'][2]==0:
   examples.append(dict(field=r['field'],speeds=lam.tolist(),groups=basis['groups'],condition=basis['condition']))
  count+=1
 assert count==1620,count
 result=dict(status='passed',native_states=count,degenerate_states=degenerate,non_glancing_projector_comparisons=compared,worst=worst,examples=examples,
  scope='Offline full seven-wave left/right decomposition; not a GPU solver replacement or performance claim. Normal CT B held fixed. Near-degenerate families grouped; glancing projector comparisons excluded.')
 result['source_sha256']={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__).resolve(),root/'scripts/characteristic_decomposition.py',root/'scripts/tests/characteristic_native_spectrum.cpp',root/'src/dyn_grmhd/characteristic_reservoir.hpp',root/'src/dyn_grmhd/rsolvers/hlle_dyn_grmhd.hpp']}
 result['diagnostic_executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
if __name__=='__main__':main()
