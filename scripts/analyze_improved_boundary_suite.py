#!/usr/bin/env python3
"""Audit all30 improved cases and compare planar waves with frozen prior controls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_grmhd_characteristic import audit,compare_curved
from analyze_rotating_characteristic import analyze
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
analyze(a.runs,a.out/'rotating');rows=json.loads((a.out/'rotating/audits.json').read_text());assert len(rows)==18
new={}
fresh={}
for n in (32,64):
 for choice in ('reference','fixed'):
  row=audit(a.runs/f'planar-acoustic-{choice}-{n}',target=180,boundary='fixed',wave_mode='acoustic');rows.append(row);fresh[n,choice]=row
 for order in (1,2,3):
  case=a.runs/f'planar-acoustic-order{order}-{n}';row=audit(case,target=180,boundary='characteristic',wave_mode='acoustic');rows.append(row);new[n,order]=row
for b in ('fixed','characteristic'):rows.append(audit(a.runs/f'curved-amr-{b}'))
assert len(rows)==30
(a.out/'all_audits.json').write_text(json.dumps(rows,indent=2)+'\n');(a.out/'curved_inner_comparison.json').write_text(json.dumps(compare_curved(a.runs),indent=2)+'\n')
old={r['name']:r for r in json.loads(a.previous.read_text())};summary=[]
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
def profile(row):
 q=np.asarray(row['profile']);z=np.unique(q[:,0]);return z,np.array([q[q[:,0]==zz,1].mean() for zz in z])
for n,ax in zip((32,64),axs):
 rz,rv=profile(fresh[n,'reference'])
 cases=[('fixed',fresh[n,'fixed']),('old first-order',old[f'acoustic-characteristic-{n}']),('optimized first-order',new[n,1]),('limited second-order',new[n,2]),('limited quadratic',new[n,3])]
 for label,row in cases:
  z,v=profile(row);j=np.searchsorted(rz,z);assert np.allclose(rz[j],z,atol=1e-10);assert row['states'][-1]['cycle']==fresh[n,'reference']['states'][-1]['cycle']
  err=(v-rv[j])/.01
  ax.plot(z,err,label=label)
  summary.append(dict(n=n,treatment=label,max_incoming_difference_over_seed=float(abs(err).max()),rms_incoming_difference_over_seed=float(np.sqrt(np.mean(err*err))),max_rank_seconds=row['max_rank_seconds']))
 ax.set(title=f'Acoustic, nz={n}',xlabel='z / M',ylabel='Incoming difference minus reference / seed');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(a.out/'planar_comparison.png',dpi=160);plt.close(fig)
(a.out/'planar_comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
# Time-resolved planar inward-pressure proxy distinguishes a propagating return
# from a late nonlinear density/entropy offset. This is still not a face-flux audit.
from analyze_grmhd_characteristic import params
from test_binary_zoom import load

def inward(path):
 d,f=load(path);cfg=params(d['header']);gamma=float(cfg['mhd/gamma']);rho=float(cfg['problem/rho_inf']);cs2=2/float(cfg['problem/r_bondi']);pinf=rho*cs2/(gamma*(1-cs2/(gamma-1)))
 sl=tuple(slice((d[f'nx{i}_out_mb']-d[f'nx{i}_mb'])//2,(d[f'nx{i}_out_mb']-d[f'nx{i}_mb'])//2+d[f'nx{i}_mb']) for i in (3,2,1));sel=(slice(None),)+sl
 u=np.array([f['vel'+c][sel] for c in 'xyz'],dtype=float);vz=u[2]/np.sqrt(1+np.sum(u*u,axis=0));q=.5*((f['press'][sel].astype(float)/pinf-1)/gamma-vz/np.sqrt(cs2));z=[];v=[]
 for mb,box in enumerate(d['mb_geometry']):
  z.extend((box[4]+(np.arange(d['nx3_mb'])+.5)*(box[5]-box[4])/d['nx3_mb']).tolist());v.extend(q[mb].mean(axis=(1,2)).tolist())
 z,v=np.array(z),np.array(v);zz=np.unique(z);return d['time'],zz,np.array([v[z==x].mean() for x in zz])
series=[];fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for n,ax in zip((32,64),axs):
 refs=sorted((a.runs/f'planar-acoustic-reference-{n}'/'bin').glob('*.mhd_w_bcc.*.bin'))
 for choice in ('fixed','order1','order2','order3'):
  files=sorted((a.runs/f'planar-acoustic-{choice}-{n}'/'bin').glob('*.mhd_w_bcc.*.bin'));assert len(files)==len(refs)
  times=[];errors=[]
  for path,ref in zip(files,refs):
   t,z,v=inward(path);tr,rz,rv=inward(ref);assert abs(t-tr)<1e-9
   j=np.searchsorted(rz,z);assert np.allclose(rz[j],z,atol=1e-10);error=float(np.max(abs(v-rv[j]))/.01);times.append(t);errors.append(error)
  series.append(dict(n=n,treatment=choice,times=times,max_inward_pressure_proxy_error_over_seed=errors));ax.plot(times,errors,label=choice)
 ax.set(title=f'Acoustic nz={n}',xlabel='t / M',ylabel='Max inward pressure-mode difference / seed');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(a.out/'planar_time_series.png',dpi=160);plt.close(fig)
(a.out/'planar_time_series.json').write_text(json.dumps(series,indent=2)+'\n')
