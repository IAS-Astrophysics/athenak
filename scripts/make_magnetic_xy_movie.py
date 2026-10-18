#!/usr/bin/env python3
"""Actual saved xy evolution with physical beta at every active AMR topology."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
from plot_bondi_validation import load
from magnetic_pilot_metric import check_pilot,spatial_metric,beta_from_primitives,validate_saved_metric

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--metric',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--frame-dt',type=float,default=100.)
p.add_argument('--start-from-restart',action='store_true',help='Label an evolved checkpoint start explicitly')
p.add_argument('--available-only',action='store_true',help='Use an explicitly selective local download; report missing files')
p.add_argument('--meshblocks',action='store_true',help='Overlay active leaf meshblock edges in the saved xy slice')
a=p.parse_args();assert a.frame_dt>0
a.out.mkdir(parents=True,exist_ok=True);frames=a.out/'frames';frames.mkdir(exist_ok=True)
md,mf=load(a.metric);metric_errors=validate_saved_metric(md,mf);del md,mf
paths=[];events=[];missing=0
for run in a.runs:
    validation=json.loads((run/'validation.json').read_text())
    for state in validation['states']:
        path=run/'bin'/state['file']
        if a.available_only and not path.exists():missing+=1;continue
        paths.append((state['time'],path))
    events += [(float(t),int(z)) for z,t in re.findall(r'CyclicZoom: new id = \d+ zone = (\d+).*?CyclicZoom: time = ([\d.eE+-]+)',(run/'rank-0.log').read_text(),re.S)]
paths=sorted({round(t,5):(t,path) for t,path in paths}.values());events.sort()
chosen=[];next_time=-1
for t,path in paths:
    if t>=next_time or path==paths[-1][1]:chosen.append((t,path));next_time=t+a.frame_dt
assert chosen[0][0]==0 or a.start_from_restart,'Use --start-from-restart for an evolved start'
start_label='evolution from checkpoint' if a.start_from_restart else 'evolution from initial data'
report=[];cache={}
for number,(t,path) in enumerate(chosen):
    d,f=load(path);c=check_pilot(d['header'])
    assert all(np.isfinite(v).all() for v in f.values()) and f['dens'].min()>0 and f['press'].min()>0
    smooth=c['coord/smooth_excision'].lower() in ('1','true')
    treatment='smooth puncture drain' if smooth else 'hard puncture excision'
    assert float(c['coord/excise_1_rad'])==float(c['coord/excise_2_rad'])==.8
    n=d['nx1_out_mb'];assert n==d['nx1_mb']==8
    # Use the topology actually written, not a rounded event time near AMR.
    zone=int(c['mesh_refinement/num_levels'])-1-int(d['mb_logical'][:,3].max())
    assert 0<=zone<int(c['amr_criterion1/nlevels'])
    fig,axes=plt.subplots(1,3,figsize=(15,5),dpi=120)
    fig.subplots_adjust(left=.05,right=.97,bottom=.25,top=.80,wspace=.27)
    norms=[LogNorm(.01,100),LogNorm(1e-4,1),LogNorm(1e-3,1e3)]
    cmaps=['magma','inferno','viridis']
    block_edges=set();slice_blocks=0
    for mb,box in enumerate(d['mb_geometry']):
      if not(box[4]<=0<box[5]) or box[0]>48 or box[1]<-48 or box[2]>48 or box[3]<-48:continue
      slice_blocks+=1
      if a.meshblocks:
        x0,x1,y0,y1=map(float,box[:4])
        block_edges.update(((x0,y0,x1,y0),(x0,y1,x1,y1),(x0,y0,x0,y1),(x1,y0,x1,y1)))
      k=int(-box[4]/(box[5]-box[4])*n)
      key=tuple(box)
      if key not in cache:
        x=box[0]+(np.arange(n)+.5)*(box[1]-box[0])/n
        y=box[2]+(np.arange(n)+.5)*(box[3]-box[2])/n
        z=box[4]+(k+.5)*(box[5]-box[4])/n
        g=spatial_metric(x[None,:],y[:,None],z)
        mask=np.zeros((n,n),dtype=bool)
        for hx in (-16.,16.):
          lor=1/np.sqrt(1-(hx*32**-1.5)**2)
          mask|=(x[None,:]-hx)**2+(lor*y[:,None])**2+z*z<=.8**2
        cache[key]=(g,mask)
      g,mask=cache[key]
      u=np.array([f['vel'+v][mb,k] for v in 'xyz'],dtype=np.float64)
      B=np.array([f['bcc'+v][mb,k] for v in '123'],dtype=np.float64)
      beta=beta_from_primitives(g,u,B,f['press'][mb,k])
      for ax,values,norm,cm in zip(axes,[f['dens'][mb,k],f['press'][mb,k]/f['dens'][mb,k],beta],norms,cmaps):
        ax.pcolormesh(np.linspace(box[0],box[1],n+1),np.linspace(box[2],box[3],n+1),np.ma.masked_where(mask,values),norm=norm,cmap=cm,shading='flat',rasterized=True)
    for ax,norm,cm,title in zip(axes,norms,cmaps,[r'$\rho/\rho_\infty$',r'$p/(\rho c^2)$',r'$\beta=2p/b^2$ (fluid frame)']):
      if a.meshblocks:
        segments=[((x0,y0),(x1,y1)) for x0,y0,x1,y1 in sorted(block_edges)]
        edges=LineCollection(segments,colors='white',linewidths=.28,alpha=.65,zorder=3,clip_on=True)
        edges.set_path_effects([path_effects.Stroke(linewidth=.65,foreground='black'),path_effects.Normal()])
        ax.add_collection(edges)
      for hx in (-16.,16.):
        lor=1/np.sqrt(1-(hx*32**-1.5)**2)
        ax.add_patch(Ellipse((hx,0),1.6,1.6/lor,color='black',zorder=4))
        ax.plot(hx,0,'+',color='cyan',ms=6,zorder=5)
      ax.set(xlim=(-48,48),ylim=(-48,48),aspect='equal',xlabel='x/M',ylabel='y/M',title=title)
      fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),ax=ax,fraction=.046,pad=.025,extend='both')
    fig.suptitle(f'Magnetized binary: {start_label} | t={t:,.2f} M | zone {zone} | {d["n_mbs"]} active blocks',y=.96,fontsize=14)
    mesh_caption=' Fine outlines: active meshblocks (8³ cells each).' if a.meshblocks else ''
    fig.text(.5,.045,f'Rotating-frame xy cell slice; actual active AMR only. Black masks: {treatment}.{mesh_caption}\n'
             'Temperature proxy p/rho; beta uses the physical fluid-frame field. Zoom exposure differs by region; relaxation is not established.\n'
             'Saved time gaps are shown in the clock; the playback does not interpolate states.',ha='center',fontsize=8.5)
    fig.savefig(frames/f'{number:04d}.png');plt.close(fig)
    report.append({'file':str(path),'time':t,'zone':zone,'blocks':d['n_mbs'],'finite_positive':True,'smooth_excision':smooth,'meshblock_overlay':a.meshblocks,'visible_slice_blocks':slice_blocks,'unique_slice_edges':len(block_edges)})
    if number%10==0:print('Rendered',number+1,'/',len(chosen),flush=True)
(a.out/'frames.json').write_text(json.dumps(report,indent=2)+'\n')
(a.out/'metric-validation.json').write_text(json.dumps({'saved_metric':str(a.metric),'spatial_component_max_error_normalized_by_max_1_abs_saved':metric_errors,'checked_all_saved_ADM_cells':True,'method':'Exact nonspinning boosted KS specialization, corotating t=0; no spatial interpolation','cached_slice_topologies':len(cache)},indent=2)+'\n')
(a.out/'selection.json').write_text(json.dumps({'meshblock_overlay':a.meshblocks,'available_only':a.available_only,'omitted_not_downloaded':missing,'frames':len(report),'first_saved_time':report[0]['time'],'last_saved_time':report[-1]['time']},indent=2)+'\n')
print('Complete',len(report),'frames; final time',report[-1]['time'],flush=True)
