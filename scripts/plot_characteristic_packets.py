#!/usr/bin/env python3
"""Compare full-solver packet residuals with a matched larger-domain control."""
import argparse,json,os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
rows=json.loads(a.results.read_text());lookup={r['name']:r for r in rows};records=[]
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for mode,ax in zip(('acoustic','alfven'),axs):
 for nz in (32,64):
  ref=lookup.get(f'{mode}-reference-{nz}')
  if ref is None:continue
  rp=np.array(ref['profile']);rz=np.unique(rp[:,0]);rv=np.array([rp[rp[:,0]==z,1].mean() for z in rz])
  for boundary in ('fixed','characteristic'):
   row=lookup.get(f'{mode}-{boundary}-{nz}')
   if row is None:continue
   assert row['time']==ref['time']==180
   q=np.array(row['profile']);z=np.unique(q[:,0]);v=np.array([q[q[:,0]==zz,1].mean() for zz in z])
   j=np.searchsorted(rz,z);assert np.allclose(rz[j],z,atol=1e-10)
   error=v-rv[j];norm=row.get('seed_amplitude',.001)
   ax.plot(z,error/norm,label=f'{boundary}, nz={nz}')
   records.append(dict(mode=mode,boundary=boundary,nz=nz,max_reference_subtracted_incoming_relative_to_seed=float(abs(error).max()/norm)))
 ax.set_title(mode.capitalize());ax.set_xlabel('z / M');ax.set_ylabel('Incoming residual minus reference / seed');ax.grid(alpha=.2)
 if ax.lines:ax.legend(fontsize=8)
fig.suptitle('Full GRMHD flat-metric packets, t=180 M; residual normalized by seed')
fig.savefig(a.out/'packet_residuals.png',dpi=160);plt.close(fig)
(a.out/'packet_comparison.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
