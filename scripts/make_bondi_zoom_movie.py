#!/usr/bin/env python3
"""Animate actual saved AMR snapshots; no interpolated simulation states."""
import argparse,json,re,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from plot_bondi_validation import load,midplane,michel
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ffmpeg')
p.add_argument('--snapshot',type=Path,help='Render this single saved state and skip movie encoding')
p.add_argument('--fps',type=int,default=4)
p.add_argument('--caption',default='Different zoom regions have different evolution exposure; only the active AMR mesh is shown.')
a=p.parse_args();assert a.fps>0
a.out.mkdir(parents=True,exist_ok=True);frames=a.out/'frames';frames.mkdir(exist_ok=True)
paths=[a.snapshot] if a.snapshot else sorted((a.run/'bin').glob('*.mhd_w_bcc.*.bin'))
assert paths and (a.snapshot or (len(paths)>2 and a.ffmpeg))
log=(a.run/'rank-0.log').read_text();events=[(float(t),int(z)) for z,t in re.findall(r'CyclicZoom: new id = \d+ zone = (\d+).*?CyclicZoom: time = ([\d.eE+-]+)',log,re.S)]
rhoinf=michel(np.array([161.]))[2];report=[]
for n,path in enumerate(paths):
 d,f=load(path);assert all(np.isfinite(v).all() for v in f.values()),path
 assert f['dens'].min()>0 and f['press'].min()>0,path
 t=d['time'];zone=next((z for et,z in reversed(events) if et<=t),0)
 fig,axes=plt.subplots(1,2,figsize=(12,6),dpi=110)
 fig.subplots_adjust(left=.07,right=.95,bottom=.20,top=.78,wspace=.30)
 for ax,key,cmap,norm,label in zip(axes,['dens','temperature'],['magma','inferno'],[LogNorm(.1,100),LogNorm(1e-4,1)],['Density / ambient density',r'Temperature $p/(\rho c^2)$']):
  cm=plt.get_cmap(cmap).copy();cm.set_under('#080812')
  for b,v in midplane(d,f[key]/rhoinf if key=='dens' else f[key]):
   x0,x1,y0,y1=b
   if x1 < -48 or x0>48 or y1 < -48 or y0>48:continue
   ax.pcolormesh(np.linspace(x0,x1,v.shape[1]+1),np.linspace(y0,y1,v.shape[0]+1),v,norm=norm,cmap=cm,shading='flat',rasterized=True)
  for x in (-16,16):ax.plot(x,0,'+',color='cyan',ms=7)
  ax.set(xlim=(-48,48),ylim=(-48,48),aspect='equal',xlabel=r'$x/M$',ylabel=r'$y/M$',title=label)
  fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),ax=ax,fraction=.046,pad=.025,extend='min')
 fig.text(.5,.96,'Binary Bondi inflow — saved low-resolution evolution',ha='center',fontsize=16)
 fig.text(.5,.90,f't = {t:,.1f} M  |  global clock = {t/(2*np.pi*32**1.5):.2f} binary orbits  |  zoom zone {zone}  |  {d["n_mbs"]} blocks',ha='center',fontsize=12)
 fig.text(.5,.055,'Full residence times; rotating-frame equatorial slices. Cyan crosses mark the black holes.\n'+a.caption,ha='center',fontsize=9)
 fig.savefig(a.out/'latest_finest.png' if a.snapshot else frames/f'{n:04d}.png');plt.close(fig)
 report.append(dict(file=path.name,time=t,cycle=d['cycle'],blocks=d['n_mbs'],zone=zone,finite=True))
 if n%50==0:print(f'Rendered {n+1}/{len(paths)} frames',flush=True)
if a.snapshot:
 (a.out/'latest_finest.json').write_text(json.dumps(report,indent=2)+'\n')
 raise SystemExit(0)
subprocess.run([a.ffmpeg,'-y','-loglevel','error','-framerate',str(a.fps),'-i',str(frames/'%04d.png'),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2,tpad=stop_mode=clone:stop_duration=2','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out/'bondi_evolution.mp4')],check=True)
subprocess.run([a.ffmpeg,'-y','-loglevel','error','-i',str(a.out/'bondi_evolution.mp4'),'-vf','fps=4,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',str(a.out/'bondi_evolution.gif')],check=True)
(a.out/'frames.json').write_text(json.dumps(report,indent=2)+'\n');print('Validated frames:',len(report),'final time:',report[-1]['time'])
