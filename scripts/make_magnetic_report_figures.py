#!/usr/bin/env python3
"""Generate schedule/geometry illustrations from physical logs for the ApJ report."""
import json,re,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
root=Path(__file__).resolve().parents[1];out=root/'output/pdf/magnetic_bondi_status';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
jobs=(8937401,8937572,8937793,8938001,8938136,8938200,8938971)
logs='\n'.join((root/f'docs/magnetic_bondi_validation/physical-{j}/rank-0.log').read_text() for j in jobs)+'\n'+(out/'live-rank0-tail.txt').read_text()
events=[(0.,0)]+[(float(t),int(z)) for z,t in re.findall(r'CyclicZoom: new id = \d+ zone = (\d+).*?CyclicZoom: time = ([\d.eE+-]+)',logs,re.S)]
events=sorted(set(events));latest=float(re.findall(r'elapsed=.*?time=([\d.eE+-]+)',logs)[-1])
xs=[e[0] for e in events]+[latest];zs=[e[1] for e in events]+[events[-1][1]]
exposure=[0.]
for i in range(len(xs)-1):exposure.append(exposure[-1]+(xs[i+1]-xs[i] if zs[i]==0 else 0))
fig,axs=plt.subplots(2,1,figsize=(10,6),dpi=150,sharex=True)
axs[0].step(xs,zs,where='post',color='#0072b2',lw=2);axs[0].set(ylabel='Active zoom stage',yticks=range(8),ylim=(-.3,7.4))
axs[1].plot(xs,exposure,color='#d55e00',lw=2);axs[1].set(xlabel='Global simulation time / M',ylabel='Cumulative zone-0 exposure / M')
for i in range(len(xs)-1):
 if zs[i]==0:
  for ax in axs:ax.axvspan(xs[i],xs[i+1],color='#d55e00',alpha=.12)
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('One completed physical roundtrip: the inner clock is not the global clock')
fig.tight_layout();fig.savefig(figs/'zoom_timeline.pdf');fig.savefig(figs/'zoom_timeline.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4),dpi=150)
for ax,r,title in zip(axs,[4.,16.,32.],['Stage 3: two masks, r=4 M','Stage 5: two masks, r=16 M','Stage 6: merged COM mask, r=32 M']):
 centers=[-16,16] if r<32 else [0]
 for c in centers:
  ax.add_patch(Circle((c,0),r,color='#56b4e9',alpha=.25))
  ax.add_patch(Circle((c,0),r,fill=False,color='#0072b2',lw=1.5))
 for hx in (-16,16):ax.add_patch(Circle((hx,0),.8,color='black'));ax.plot(hx,0,'k+',ms=6)
 ax.set(xlim=(-52,52),ylim=(-40,40),aspect='equal',xlabel='x/M',ylabel='y/M',title=title);ax.grid(alpha=.2)
fig.suptitle('Zoom masks (blue) store/reapply unresolved matter; physical puncture excision (black) is separate',fontsize=11)
fig.tight_layout();fig.savefig(figs/'mask_geometry.pdf');fig.savefig(figs/'mask_geometry.png');plt.close(fig)
for src,dst in [('latest_finest_xy.png','flow_xy.png'),('latest_finest.png','flow_xz.png'),('movie-first-return/boundary_profiles.png','boundary_profiles.png'),('movie-first-return/boundary_magnetic.png','boundary_magnetic.png')]:shutil.copy2(root/'docs/magnetic_bondi_validation'/src,figs/dst)
cs=np.sqrt(.004);tb=500/cs;period=2*np.pi*32**1.5
schedule=[]
for i in range(8):
 r=.5*2**i;mass=.5 if r<32 else 1.;dyn=np.sqrt(r**3/mass)
 dwell=250 if i==0 else 0 if i in (1,2) else min(50*dyn,.15*tb)
 schedule.append({'stage':i,'radius':r,'centers':2 if r<32 else 1,'local_mass':mass,'r_ref':4*r,'mask_dx':r/16,'dynamical_time':dyn,'dwell':dwell,'dwell_over_dyn':dwell/dyn})
meta={'last_logged_time_M':latest,'zone':zs[-1],'finest_exposure_M':exposure[-1],'global_binary_orbits':latest/period,'finest_binary_orbits':exposure[-1]/period,'bondi_times':latest/tb,'binary_period_M':period,'bondi_time_M':tb,'cs':cs,'pinf':.004/((5/3)*(1-.004/(2/3))),'Binf':np.sqrt(2*.004/((5/3)*(1-.004/(2/3)))),'events':events,'schedule':schedule,'nominal_roundtrip_M':sum(q['dwell'] for q in schedule)+sum(q['dwell'] for q in schedule[1:-1])}
(out/'schedule.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k not in ('events','schedule')},indent=2))
