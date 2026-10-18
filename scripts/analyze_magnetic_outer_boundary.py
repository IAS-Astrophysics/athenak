#!/usr/bin/env python3
"""Inspect active layers beside the six fixed-reservoir faces (no ghost/face flux claim)."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_bondi_validation import load
from magnetic_pilot_metric import check_pilot
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--metric',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
md,mg=load(a.metric);lookup={tuple(b):i for i,b in enumerate(md['mb_geometry'])};c=check_pilot(md['header']);rhoinf=float(c['problem/rho_inf']);gam=float(c['mhd/gamma']);cs2=float(c['problem/cs_inf'])**2;pinf=rhoinf*cs2/(gam*(1-cs2/(gam-1)));binf=np.sqrt(2*pinf/float(c['problem/beta_inf']));omega=float(c['problem/frame_omega']);bound=float(c['mesh/x1max']);n=md['nx1_out_mb'];rows=[]
paths=[(0.,a.initial)]
for run in a.runs:
 v=json.loads((run/'validation.json').read_text())
 paths.extend((s['time'],run/'bin'/s['file']) for s in v['states'] if (run/'bin'/s['file']).exists())
paths=sorted({round(t,5):(t,f) for t,f in paths}.values())
for t,path in paths:
 d,f=load(path);assert all(np.isfinite(v).all() for v in f.values()) and f['dens'].min()>0 and f['press'].min()>0
 profiles=[]
 for axis in range(3):
  for sign in (-1,1):
   values=[[] for _ in range(6)]
   for mb,box in enumerate(d['mb_geometry']):
    if abs(box[2*axis+(sign>0)]-sign*bound)>1e-6:continue
    assert tuple(box) in lookup,'Outer block metric geometry changed'
    mi=lookup[tuple(box)];g=np.array([[mg['adm_gxx'][mi],mg['adm_gxy'][mi],mg['adm_gxz'][mi]],[mg['adm_gxy'][mi],mg['adm_gyy'][mi],mg['adm_gyz'][mi]],[mg['adm_gxz'][mi],mg['adm_gyz'][mi],mg['adm_gzz'][mi]]],dtype=float)
    u=np.array([f['vel'+k][mb] for k in 'xyz'],dtype=float);W=np.sqrt(1+np.einsum('ab...,a...,b...->...',g,u,u));v=mg['adm_alpha'][mi]*u/W-np.array([mg['adm_beta'+k][mi] for k in 'xyz'],dtype=float)
    x=box[0]+(np.arange(n)+.5)*(box[1]-box[0])/n;y=box[2]+(np.arange(n)+.5)*(box[3]-box[2])/n
    v[0]-=omega*y[None,:,None];v[1]+=omega*x[None,None,:]
    vals=np.array([f['dens'][mb]/rhoinf-1,f['press'][mb]/pinf-1,sign*v[axis],np.sqrt(np.sum(v*v,axis=0)),f['bcc1'][mb]/binf,f['bcc2'][mb]/binf,f['bcc3'][mb]/binf-1])
    for layer in range(6):values[layer].append(np.take(vals,layer if sign<0 else n-1-layer,axis=3-axis))
   profiles.append([{'mean':np.concatenate([q.reshape(7,-1) for q in vs],axis=1).mean(axis=1).tolist(),'maxabs':np.abs(np.concatenate([q.reshape(7,-1) for q in vs],axis=1)).max(axis=1).tolist()} for vs in values])
 rows.append({'time':t,'source':str(path),'profiles':profiles})
means=np.array([[[p['mean'] for p in face] for face in row['profiles']] for row in rows]);maxima=np.array([[[p['maxabs'] for p in face] for face in row['profiles']] for row in rows]);assert np.isfinite(means).all() and np.isfinite(maxima).all();assert maxima[0,:,0,3].max()<2e-6
names=['x-','x+','y-','y+','z-','z+'];fields=['rho/rho_inf-1','p/p_inf-1','outward_inertial_coordinate_v_normal','inertial_coordinate_speed','Btilde_x/Binf','Btilde_y/Binf','Btilde_z/Binf-1'];times=[r['time'] for r in rows];dist=64+128*np.arange(6)
fig,axs=plt.subplots(2,3,figsize=(13,7),dpi=140)
def temporal(ax,values,label):
 line,=ax.plot(times[1:],values[1:],'.-',ms=2,label=label)
 ax.scatter([times[0]],[values[0]],s=8,color=line.get_color())
for i,name in enumerate(names):
 for col,key in enumerate((0,1,6)):temporal(axs[0,col],100*means[:,i,0,key],name)
 temporal(axs[1,0],maxima[:,i,0,3],name)
 axs[1,1].plot(dist,100*means[-1,i,:,1],'.-',label=name)
 temporal(axs[1,2],np.maximum(maxima[:,i,0,4],maxima[:,i,0,5]),name)
for ax in axs.flat:ax.grid(alpha=.2)
for ax,title in zip(axs[0],('Adjacent mean density excess [%]','Adjacent mean pressure excess [%]','Adjacent mean Btilde_z excess [%]')):ax.set(xlabel='Global time / M',title=title)
axs[1,0].set(xlabel='Global time / M',title='Adjacent maximum inertial coordinate speed / c')
axs[1,1].set(xlabel='Distance inward from face / M',title='Final mean pressure excess [%]')
axs[1,2].set(xlabel='Global time / M',title='Adjacent max |Btilde_x,y| / Binf')
axs[0,0].legend(ncol=2,fontsize=8);fig.suptitle('Six-face outer active layers: t=0 calibration + selected smooth-chain snapshots\nFixed reservoir faces at ±2048 M; these are not ghost cells or boundary face fluxes',fontsize=11);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(a.out/'outer_boundary.png');plt.close(fig)
summary={'time_range_M':[times[0],times[-1]],'selected_frames':len(rows),'final_adjacent_max_abs_face_mean':dict(zip(fields,np.abs(means[-1,:,0]).max(axis=0).tolist())),'final_adjacent_max_abs_over_all_cells_faces':dict(zip(fields,maxima[-1,:,0].max(axis=0).tolist())),'all_time_adjacent_max_abs_over_all_cells_faces':dict(zip(fields,maxima[:,:,0].max(axis=(0,1)).tolist())),'initial_max_inertial_coordinate_speed':float(maxima[0,:,0,3].max()),'final_face_means':{name:dict(zip(fields,means[-1,i,0].tolist())) for i,name in enumerate(names)},'light_cylinder_M':1/omega,'outer_face_midplane_rotation_speed_c':omega*bound,'outer_equatorial_corner_rotation_speed_c':omega*bound*np.sqrt(2),'reservoir':dict(rho=rhoinf,pressure=pinf,Btilde_z=binf,cs=np.sqrt(cs2)),'limits':'Active-cell, not ghost or face-flux diagnostics. No nonreflection or mass-budget proof. Corners counted on each adjoining face. Outer metric geometry matched exactly. Float32 saved ADM limits small velocity cancellation accuracy.'}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.out/'profiles.json').write_text(json.dumps({'fields':fields,'faces':names,'cell_distances_M':dist.tolist(),'rows':rows},indent=2)+'\n');print(json.dumps(summary,indent=2))
