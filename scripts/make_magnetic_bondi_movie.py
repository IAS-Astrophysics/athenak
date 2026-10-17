#!/usr/bin/env python3
"""Render validated physical MHD snapshots and inspect adjacent outer active cells.

No ghost/face-flux outputs are available: these diagnostics cannot establish
nonreflection or a conservative boundary mass budget. Initial stationary ADM
blocks are reused only where exact outer-block geometry matches.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from matplotlib.patches import Circle
from plot_bondi_validation import load

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--metric',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--frame-dt',type=float,default=100.)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
frames=a.out/'frames';frames.mkdir(exist_ok=True)
md,mg=load(a.metric);lookup={tuple(b):i for i,b in enumerate(md['mb_geometry'])}
pinf=(2/500)/((5/3)*(1-(2/500)/(2/3)));binf=np.sqrt(2*pinf);omega=32**-1.5
paths=[];events=[]
for run in a.runs:
    validation=json.loads((run/'validation.json').read_text())
    paths += [(s['time'],run/'bin'/s['file']) for s in validation['states']]
    log=(run/'rank-0.log').read_text()
    events += [(float(t),int(z)) for z,t in re.findall(r'CyclicZoom: new id = \d+ zone = (\d+).*?CyclicZoom: time = ([\d.eE+-]+)',log,re.S)]
paths=sorted({round(t,5):(t,path) for t,path in paths}.values());events.sort()
chosen=[];next_time=-1
for t,path in paths:
    if t>=next_time or path==paths[-1][1]:chosen.append((t,path));next_time=t+a.frame_dt
rows=[];manifest=[]
for number,(t,path) in enumerate(chosen):
    d,f=load(path)
    assert all(np.isfinite(v).all() for v in f.values()) and f['dens'].min()>0 and f['press'].min()>0
    n=d['nx1_out_mb'];assert n==d['nx1_mb']==8
    # Six faces, six cell layers inward; corners belong to each relevant face.
    profiles=[]
    for axis in range(3):
      for sign in (-1,1):
        values=[[] for _ in range(6)]
        for mb,box in enumerate(d['mb_geometry']):
          if abs(box[2*axis+(sign>0)]-sign*2048)>1e-6:continue
          assert tuple(box) in lookup,'Outer metric topology differs'
          mi=lookup[tuple(box)];g=np.array([[mg['adm_gxx'][mi],mg['adm_gxy'][mi],mg['adm_gxz'][mi]],[mg['adm_gxy'][mi],mg['adm_gyy'][mi],mg['adm_gyz'][mi]],[mg['adm_gxz'][mi],mg['adm_gyz'][mi],mg['adm_gzz'][mi]]])
          u=np.array([f['vel'+k][mb] for k in 'xyz']);W=np.sqrt(1+np.einsum('ab...,a...,b...->...',g,u,u))
          v=mg['adm_alpha'][mi]*u/W-np.array([mg['adm_beta'+k][mi] for k in 'xyz'])
          x=box[0]+(np.arange(n)+.5)*(box[1]-box[0])/n
          y=box[2]+(np.arange(n)+.5)*(box[3]-box[2])/n
          v[0]-=omega*y[None,:,None];v[1]+=omega*x[None,None,:]
          vals=np.array([f['dens'][mb]-1,f['press'][mb]/pinf-1,sign*v[axis],np.sqrt(sum(vv*vv for vv in v)),f['bcc1'][mb]/binf,f['bcc2'][mb]/binf,f['bcc3'][mb]/binf-1])
          for layer in range(6):
            ix=layer if sign<0 else n-1-layer
            values[layer].append(np.take(vals,ix,axis=3-axis))
        profiles.append([{'mean':np.concatenate([q.reshape(7,-1) for q in v],axis=1).mean(axis=1).tolist(),
                          'maxabs':np.abs(np.concatenate([q.reshape(7,-1) for q in v],axis=1)).max(axis=1).tolist()} for v in values])
    rows.append({'time':t,'profiles':profiles})
    zone=next((z for et,z in reversed(events) if et<=t),0)
    fig,axs=plt.subplots(2,2,figsize=(11,10),dpi=100)
    fig.subplots_adjust(left=.08,right=.94,bottom=.12,top=.90,hspace=.32,wspace=.30)
    configs=[(f['dens'],48,LogNorm(.01,100),'magma',r'$\rho/\rho_\infty$'),(f['press']/f['dens'],48,LogNorm(1e-4,1),'inferno',r'$p/(\rho c^2)$'),(f['dens']-1,2048,Normalize(-.08,.08),'RdBu_r',r'Outer view: $\rho/\rho_\infty-1$'),(f['press']/pinf-1,2048,Normalize(-.08,.08),'RdBu_r',r'Outer view: $p/p_\infty-1$')]
    for ax,(field,extent,norm,cmap,title) in zip(axs.flat,configs):
      for mb,box in enumerate(d['mb_geometry']):
        if not(box[2]<=0<box[3]) or box[0]>extent or box[1]<-extent or box[4]>extent or box[5]<-extent:continue
        j=int(-box[2]/(box[3]-box[2])*n);v=field[mb,:,j,:]
        ax.pcolormesh(np.linspace(box[0],box[1],n+1),np.linspace(box[4],box[5],n+1),v,norm=norm,cmap=cmap,shading='flat',rasterized=True)
      if extent==48:
        for hx in (-16,16):ax.add_patch(Circle((hx,0),.8,color='black'))
      else:ax.add_patch(Circle((0,0),500,fill=False,edgecolor='black',lw=.8,ls='--'))
      ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',xlabel='x/M',ylabel='z/M',title=title)
      fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,fraction=.046,pad=.03,extend='both')
    fig.suptitle(f'Magnetized binary reservoir | t={t:,.1f} M | zone {zone} | {d["n_mbs"]} active blocks')
    fig.text(.5,.035,'Actual active AMR snapshots; coarse stages do not show frozen finer fields.\nUniform reservoir, beta_inf=1. Dashed circle: thermal Bondi radius 500 M.\nEqual playback intervals sample saved states; frame times can have restart gaps.',ha='center',fontsize=9)
    fig.savefig(frames/f'{number:04d}.png');plt.close(fig)
    manifest.append({'file':str(path),'time':t,'zone':zone,'blocks':d['n_mbs'],'finite_positive':True})
    if number%10==0:print('Rendered',number+1,'/',len(chosen),flush=True)
(a.out/'frames.json').write_text(json.dumps(manifest,indent=2)+'\n')
(a.out/'boundary.json').write_text(json.dumps({'fields':['rho/rho_inf-1','p/p_inf-1','outward_inertial_coordinate_v_normal','inertial_coordinate_speed','Btilde_x/Binf','Btilde_y/Binf','Btilde_z/Binf-1'],'face_order':['x-','x+','y-','y+','z-','z+'],'cell_distances_from_face_M':[64+128*i for i in range(6)],'rows':rows,'limits':'Active cell diagnostics, not ghost cells or face fluxes. Corners counted on each adjoining face; identical outer metric geometry asserted.'},indent=2)+'\n')
means=np.array([[q[0]['mean'] for q in r['profiles']] for r in rows]);last=np.array([[v['mean'] for v in q] for q in rows[-1]['profiles']]);ts=[r['time'] for r in rows]
fig,axs=plt.subplots(2,3,figsize=(13,7),dpi=140)
for col,key in enumerate((0,1,2)):
    for face,name in enumerate(('x-','x+','y-','y+','z-','z+')):
      axs[0,col].plot(ts,means[:,face,key],label=name)
      axs[1,col].plot([64+128*i for i in range(6)],last[face,:,key],label=name)
    for ax in axs[:,col]:ax.grid(alpha=.25)
    axs[0,col].set(xlabel='Global time / M',title=('Adjacent density excess','Adjacent pressure excess','Outward inertial normal velocity / c')[col])
    axs[1,col].set(xlabel='Active cell distance inward from face / M',title=f'Last saved state: t={ts[-1]:.2f} M')
axs[0,0].legend(ncol=2,fontsize=8);fig.suptitle('Outer boundary: face-averaged active-cell diagnostics (fixed exterior reservoir)')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(a.out/'boundary_profiles.png');plt.close(fig)
print('Complete',len(manifest),'frames; final time',ts[-1],flush=True)
