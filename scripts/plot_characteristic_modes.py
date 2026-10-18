#!/usr/bin/env python3
"""Measure saved planar incoming amplitudes with the full native eigenbasis.

Requires test_characteristic_decomposition.py to compile the native exporter.
All seven primitive components are decomposed; coincident-speed polarizations are
reported together. The plotted acoustic family is nondegenerate for this vertical
B reservoir (the slow branch). Compare with the matched larger-domain result,
not with zero; these are small-amplitude diagnostics rather than nonlinear
reflection coefficients. No evolution source or executable is modified.
"""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from characteristic_decomposition import speeds,decompose
sys.path.insert(0,str(Path(__file__).resolve().parent/'tests'))
from test_binary_zoom import load


def profile(path,reference,scale):
 d,f=load(path);sl=tuple(slice((d[f'nx{i}_out_mb']-d[f'nx{i}_mb'])//2,(d[f'nx{i}_out_mb']-d[f'nx{i}_mb'])//2+d[f'nx{i}_mb']) for i in (3,2,1));sel=(slice(None),)+sl
 # Normal z means q[5]=Bx and q[6]=By. Output velocities are utilde, not v.
 q=np.stack([f[k][sel].astype(float) for k in ('dens','press','velx','vely','velz','bcc1','bcc2')],axis=-1)
 q=(q-reference)/scale;zc=[];qc=[]
 for m,box in enumerate(d['mb_geometry']):
  zc.extend(box[4]+(np.arange(d['nx3_mb'])+.5)*(box[5]-box[4])/d['nx3_mb']);qc.extend(q[m].mean(axis=(1,2)))
 zc,qc=np.asarray(zc),np.asarray(qc);z=np.unique(zc)
 return d['time'],z,np.array([qc[zc==x].mean(axis=0) for x in z])


def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resolutions',type=int,nargs='+',default=[32,64]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 output=subprocess.check_output([str((a.build/'characteristic-native-spectrum').resolve()),'--reservoir'],text=True)
 r=next(json.loads(line) for line in output.splitlines() if line.startswith('{') and json.loads(line)['axis']==2)
 lam=speeds(r['q'],r['bn'],np.eye(3),1,np.zeros(3),2);basis=decompose(r['A'],lam,1);R,L=basis['R'],basis['L'];reference=np.asarray(r['q']);scale=np.asarray(r['scale']);gamma=5/3;seed=.01
 metadata=dict(scope='Full seven-wave offline decomposition; normal CT B fixed; coincident families grouped. Weak planar acoustic diagnostic, not a nonlinear reflection coefficient.',q_reference=r['q'],scale=r['scale'],speeds=lam.tolist(),R=R.tolist(),L=L.tolist(),groups=basis['groups'],eigen_residual=basis['residual'])
 series=[];fig,axs=plt.subplots(1,len(a.resolutions),figsize=(5.5*len(a.resolutions),4),squeeze=False,constrained_layout=True)
 for n,ax in zip(a.resolutions,axs.ravel()):
  refs=sorted((a.runs/f'planar-acoustic-reference-{n}'/'bin').glob('*.mhd_w_bcc.*.bin'))
  for choice in ('fixed','order1','order2','order3'):
   files=sorted((a.runs/f'planar-acoustic-{choice}-{n}'/'bin').glob('*.mhd_w_bcc.*.bin'));assert len(files)==len(refs)
   times=[];errors=[];all_errors=[];density_errors=[]
   for path,ref in zip(files,refs):
    t,z,q=profile(path,reference,scale);tr,rz,qr=profile(ref,reference,scale);assert abs(t-tr)<1e-9
    j=np.searchsorted(rz,z);assert np.allclose(rz[j],z,atol=1e-10)
    amplitudes=(q-qr[j])@L.T
    # Each group reconstructs a physical pressure contribution. Individual basis
    # amplitudes depend on normalization; pressure sum is basis invariant.
    contributions=[];density_contributions=[]
    for group in basis['groups']:
     ids=group['indices'];contributions.append(amplitudes[:,ids]@R[1,ids]/(gamma*seed));density_contributions.append(amplitudes[:,ids]@R[0,ids]/seed)
    inward=sum(c for c,g in zip(contributions,basis['groups']) if g['speed']< -1e-7)
    times.append(t);errors.append(float(abs(inward).max()));all_errors.append([float(abs(c).max()) for c in contributions]);density_errors.append([float(abs(c).max()) for c in density_contributions])
   series.append(dict(n=n,treatment=choice,times=times,incoming_pressure_error_over_seed=errors,group_pressure_errors_over_seed=all_errors,group_density_errors_over_seed=density_errors))
   ax.plot(times,errors,label=choice)
  ax.set(title=f'Full seven-wave decomposition, nz={n}',xlabel='t / M',ylabel='Max incoming pressure contribution / seed');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.out/'full_mode_acoustic.png',dpi=160);plt.close(fig)
 (a.out/'full_mode_acoustic.json').write_text(json.dumps(dict(basis=metadata,series=series),indent=2)+'\n')
if __name__=='__main__':main()
