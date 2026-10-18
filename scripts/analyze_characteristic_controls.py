#!/usr/bin/env python3
"""Audit completed fixed-reservoir GPU controls; no characteristic claims."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent/'tests'))
from test_binary_zoom import load,final_output
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows=[]
for case in sorted(a.runs.glob('control-*')):
 if not case.is_dir():continue
 rss=tracked=budgets=0;hosts=set()
 for rank in range(12):
  r=json.loads((case/f'resources-{rank}.json').read_text());m=json.loads((case/f'memory-{rank}.json').read_text())
  assert r['returncode']==0 and r['stopped'] is None
  assert r['peak_rss_bytes']<=r['memory_budget_bytes'] and m['combined_peak_bytes']<=m['budget_bytes']
  log=(case/f'rank-{rank}.log').read_text();identity=re.search(rf'rank={rank} host=(\S+) affinity={rank//2}\.{rank%2}\s',log);assert identity
  hosts.add(identity[1]);rss+=r['peak_rss_bytes'];tracked+=m['combined_peak_bytes'];budgets+=r['memory_budget_bytes']+m['budget_bytes']
  assert r['command'][r['command'].index('-t')+1]=='00:55:00'
 assert len(hosts)==1 and budgets<=30e9 and rss+tracked<30e9
 assert 'Terminating on time limit' in (case/'rank-0.log').read_text()
 d,f=load(final_output(case,'mhd_w_bcc'));md,g=load(final_output(case,'adm'));dd,df=load(final_output(case,'mhd_divb'))
 assert d['time']==64 and md['time']==64 and dd['time']==64
 assert all(np.isfinite(x).all() for x in f.values()) and f['dens'].min()>0 and f['press'].min()>0
 nx=d['nx1_mb'];ng=(d['nx1_out_mb']-nx)//2
 active=tuple(slice(ng,ng+nx) for _ in range(3));B=np.sqrt(sum(f[f'bcc{i}'][(slice(None),)+active].astype(float)**2 for i in (1,2,3)))
 div=df['divb'];assert np.isfinite(div).all();dx=(d['mb_geometry'][:,1]-d['mb_geometry'][:,0])/nx
 scaled=float(np.max(abs(div)*dx[:,None,None,None]/np.maximum(B,1e-6)))
 assert scaled<1e-9,scaled
 omega=0 if 'inertial' in case.name else 32**-1.5
 coords=[]
 for dim in range(3):
  xyz=d['mb_geometry'][:,2*dim,None]+(d['mb_index'][:,2*dim,None]+np.arange(d[f'nx{dim+1}_out_mb'])[None,:]+.5)*(d['mb_geometry'][:,2*dim+1]-d['mb_geometry'][:,2*dim])[:,None]/nx
  coords.append(xyz[:,None,None,:] if dim==0 else xyz[:,None,:,None] if dim==1 else xyz[:,:,None,None])
 x,y,z=np.broadcast_arrays(*coords);external=(abs(x)>2048)|(abs(y)>2048)|(abs(z)>2048)
 metric=np.array([[g['adm_gxx'],g['adm_gxy'],g['adm_gxz']],[g['adm_gxy'],g['adm_gyy'],g['adm_gyz']],[g['adm_gxz'],g['adm_gyz'],g['adm_gzz']]],dtype=float)
 u=np.array([f['vel'+c] for c in 'xyz'],dtype=float);W=np.sqrt(1+np.einsum('ab...,a...,b...->...',metric,u,u))
 velocity=g['adm_alpha']*u/W-np.array([g['adm_beta'+c] for c in 'xyz']);velocity[0]-=omega*y;velocity[1]+=omega*x
 ghost_error=float(abs(velocity[:,external]).max());assert ghost_error<2e-6,ghost_error
 pinf=.004/((5/3)*(1-.004/(2/3)));assert np.allclose(f['dens'][external],1,atol=1e-6);assert np.allclose(f['press'][external],pinf,rtol=2e-6)
 face=[]
 for axis in range(3):
  for sign in (-1,1):
   values=[]
   for m,box in enumerate(d['mb_geometry']):
    if abs(box[2*axis+(sign>0)]-sign*2048)>1e-6:continue
    s=[slice(ng,ng+nx)]*3;s[2-axis]=ng if sign<0 else ng+nx-1
    values.append(np.array([f['dens'][m][tuple(s)]-1,f['press'][m][tuple(s)]/pinf-1]).reshape(2,-1))
   values=np.concatenate(values,axis=1);face.append(dict(mean=values.mean(axis=1).tolist(),maxabs=abs(values).max(axis=1).tolist()))
 rows.append(dict(case=case.name,time=64,rss_plus_tracked_bytes=rss+tracked,scaled_divergence=scaled,ghost_inertial_velocity_error=ghost_error,faces=face))
assert len(rows)==4
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for row in rows:
 for j,ax in enumerate(axs):ax.plot(range(6),[100*f['mean'][j] for f in row['faces']],'.-',label=row['case'].replace('control-','').replace('-8941753',''))
for ax,title in zip(axs,['Density deviation (%)','Pressure deviation (%)']):
 ax.set_xticks(range(6),['x−','x+','y−','y+','z−','z+']);ax.set_ylabel(title);ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig.suptitle('Existing fixed-reservoir GPU controls, t=64 M; first active layer')
fig.savefig(a.out/'fixed_controls.png',dpi=160);plt.close(fig)
(a.out/'validation.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
