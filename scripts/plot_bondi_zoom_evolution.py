#!/usr/bin/env python3
"""Render saved cyclic-zoom AMR snapshots without interpolating the mesh."""
import argparse
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np
from plot_bondi_validation import load, midplane, michel

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--residence-scale',type=float,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
paths=sorted((a.run/'bin').glob('*.mhd_w_bcc.*.bin'))
assert len(paths)>=2
states=[load(paths[0]),load(paths[-1])]
rhoinf=michel(np.array([161.]))[2]
report=[]
for d,f in states:
 assert all(np.isfinite(v).all() for v in f.values())
 assert f['dens'].min()>0 and f['press'].min()>0
 report.append(dict(time=d['time'],cycle=d['cycle'],blocks=d['n_mbs'],
  density_min=float(f['dens'].min()),density_max=float(f['dens'].max())))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
specs=[('dens',256,LogNorm(1e-7,1e2),'magma',r'Density $\rho/\rho_\infty$ — reservoir'),
       ('dens',40,LogNorm(1e-7,1e2),'magma',r'Density $\rho/\rho_\infty$ — binary'),
       ('temperature',40,LogNorm(1e-5,.2),'inferno',r'Temperature $p/\rho c^2$ — binary')]
for row,(d,f) in enumerate(states):
 for col,(key,half,norm,cmap,title) in enumerate(specs):
  ax=axes[row,col];segments=[]
  values=f[key]/rhoinf if key=='dens' else f[key]
  for b,v in midplane(d,values):
   x0,x1,y0,y1=b
   if x1 < -half or x0>half or y1 < -half or y0>half:continue
   ax.pcolormesh(np.linspace(x0,x1,v.shape[1]+1),np.linspace(y0,y1,v.shape[0]+1),v,
                 norm=norm,cmap=cmap,shading='flat',rasterized=True)
   if col==1:segments.append([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)])
  if segments:ax.add_collection(LineCollection(segments,colors='white',linewidths=.22,alpha=.28))
  for x in (-16,16):
   ax.plot(x,0,'+',color='cyan',ms=7,mew=1)
   if col:ax.add_patch(Circle((x,0),1,fill=False,color='cyan',lw=.7))
  if col==0:ax.add_patch(Circle((0,0),161.227,fill=False,color='cyan',ls='--',lw=.8))
  ax.set(xlim=(-half,half),ylim=(-half,half),aspect='equal',xlabel=r'$x/M$',ylabel=r'$y/M$')
  if row==0:ax.set_title(title,pad=10)
  ax.text(.03,.97,f"t = {d['time']:g} M\n{d['n_mbs']} blocks",transform=ax.transAxes,
          va='top',color='white',bbox=dict(facecolor='black',alpha=.6,edgecolor='none',pad=3))
for col,(_,_,norm,cmap,_) in enumerate(specs):
 fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes[:,col],shrink=.78,pad=.02)
label='Physical residence times' if a.residence_scale==1 else f'Accelerated residence times × {a.residence_scale:g}; not a relaxed flow'
fig.text(.5,.985,'Binary Bondi inflow: initial cavity and subsequent evolution',ha='center',va='top',fontsize=14)
fig.text(.5,.95,label,ha='center',va='top',fontsize=11)
fig.get_layout_engine().set(rect=(0,.065,1,.84))
fig.text(.5,.025,'Equatorial AMR slices in the rotating frame. Cyan: hole positions / nominal horizons; dashed circle: Bondi radius.\n'
         'Grid lines show MeshBlocks. Cyclic-zoom regions have different evolution exposure; the final inner grid may be coarsened.',ha='center',fontsize=9)
fig.savefig(a.out/'density_temperature.png',dpi=180)
fig.savefig(a.out/'density_temperature.pdf')
plt.close(fig)
log=(a.run/'rank-0.log').read_text()
events=re.findall(r'CyclicZoom: new id = (\d+) zone = (\d+).*?CyclicZoom: time = ([\d.eE+-]+)',log,re.S)
t=[0.]+[float(e[2]) for e in events]+[states[-1][0]['time']]
z=[0]+[int(e[1]) for e in events];z.append(z[-1])
fig,ax=plt.subplots(figsize=(9,3.1),layout='constrained')
ax.step(t,z,where='post',color='#0072b2',lw=1.8)
ax.set(xlabel=r'Simulation clock $t/M$',ylabel='Zoom zone (0 = finest)',title='Executed zoom sequence — '+label)
ax.set_yticks(range(max(z)+1));ax.grid(alpha=.2)
fig.savefig(a.out/'zoom_history.png',dpi=180);fig.savefig(a.out/'zoom_history.pdf')
(a.out/'validation.json').write_text(json.dumps(dict(states=report,residence_scale=a.residence_scale,events=events),indent=2)+'\n')
print(json.dumps(report,indent=2))
