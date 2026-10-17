#!/usr/bin/env python3
"""Compare initial/first-return MHD finest visits without importing hydro normalization."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_bondi_validation import load
from bondi_hole_accretion import geometry,flux
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[1];jobs=(8937401,8937572,8937793,8938001,8938136,8938200,8938971)
runs=[a.base/f'evolution-magnetic-{j}' for j in jobs]
logs='\n'.join((root/f'docs/magnetic_bondi_validation/physical-{j}/rank-0.log').read_text() for j in jobs)
events=[(0.,0)]+[(float(t),int(z)) for z,t in re.findall(r'CyclicZoom: new id = \d+ zone = (\d+).*?CyclicZoom: time = ([\d.eE+-]+)',logs,re.S)]
entry=[t for t,z in events if z==0];assert len(entry)==2
metric=runs[0]/'bin/magnetic_bondi_binary.adm.00000.bin';ad,af=load(metric)
radii=[24.,32.,64.,128.,256.,500.];edges=np.array([24.,32.,64.,128.,256.,500.])
geo={(r,c,n):geometry(ad,af,r,c,n) for r,c in [(r,0.) for r in radii]+[(r,c) for c in (-16.,16.) for r in (1.5,2.,3.)] for n in (24,48)}
b=ad['mb_geometry'];n=ad['nx1_out_mb'];xyz=[]
for i in range(3):
 q=b[:,2*i,None]+(np.arange(n)+.5)*(b[:,2*i+1]-b[:,2*i])[:,None]/n
 xyz.append(q[:,None,None,:] if i==0 else q[:,None,:,None] if i==1 else q[:,:,None,None])
x,y,z=np.broadcast_arrays(*xyz);rr=np.sqrt(x*x+y*y+z*z)
sel=(rr>=edges[0])&(rr<edges[-1]);shell=np.searchsorted(edges,rr[sel],side='right')-1
G=np.array([[af['adm_gxx'][sel],af['adm_gxy'][sel],af['adm_gxz'][sel]],[af['adm_gxy'][sel],af['adm_gyy'][sel],af['adm_gyz'][sel]],[af['adm_gxz'][sel],af['adm_gyz'][sel],af['adm_gzz'][sel]]],dtype=float)
det=np.linalg.det(np.moveaxis(G,(0,1),(-2,-1)));assert (det>0).all()
vol=np.prod(b[:,[1,3,5]]-b[:,[0,2,4]],axis=1)/n**3
vol=np.broadcast_to(vol[:,None,None,None],rr.shape)[sel]
rows=[];seen=set()
for run in runs:
 for path in sorted((run/'bin').glob('*.mhd_w_bcc.*.bin')):
  d,f=load(path);t=d['time']
  if t in seen:continue
  assert np.array_equal(d['mb_geometry'],b) and np.array_equal(d['mb_logical'],ad['mb_logical'])
  assert all(np.isfinite(v).all() for v in f.values()) and f['dens'].min()>0 and f['press'].min()>0
  seen.add(t);visit=int(t>=entry[1]);phase=t-entry[visit]
  U=np.array([f['vel'+k][sel] for k in 'xyz'],dtype=float)
  W=np.sqrt(1+np.einsum('ij...,i...,j...->...',G,U,U))
  masses=np.bincount(shell,weights=f['dens'][sel]*W*np.sqrt(det)*vol,minlength=len(edges)-1)
  com=[];comerr=[];holes=[];holeerr=[]
  for r,c in [(r,0.) for r in radii]+[(r,c) for c in (-16.,16.) for r in (1.5,2.,3.)]:
   low,_=flux(d,f,geo[r,c,24],r);high,_=flux(d,f,geo[r,c,48],r)
   (com if c==0 else holes).append(high);(comerr if c==0 else holeerr).append(abs(high-low))
  rows.append(dict(time=t,visit=visit,phase=phase,rates=com,angular_errors=comerr,hole_rates=holes,hole_errors=holeerr,masses=masses.tolist(),source=str(path)))
  print('Analyzed',t,flush=True)
rows.sort(key=lambda q:q['time']);means=[];phases=[50.,100.,200.]
for v in (0,1):
 qs=[q for q in rows if q['visit']==v];xp=np.array([q['phase'] for q in qs])
 assert xp.min()<=min(phases) and xp.max()>=max(phases)
 m={'visit':v}
 for key in ('rates','masses','hole_rates','angular_errors','hole_errors'):
  values=np.array([q[key] for q in qs]);m[key]=[float(np.interp(phases,xp,col).mean()) for col in values.T]
 means.append(m)
changes={k:(100*(np.array(means[1][k])/np.array(means[0][k])-1)).tolist() for k in ('rates','masses','hole_rates')}
fig,axs=plt.subplots(2,2,figsize=(11,8),dpi=150)
for v in (0,1):
 q=[s for s in rows if s['visit']==v];t=[s['phase'] for s in q]
 for hole,ls in ((0,'-'),(1,'--')):axs[0,0].plot(t,[s['hole_rates'][hole*3+1] for s in q],ls,marker='.',label=f'Visit {v}, hole {hole+1}, r=2')
 axs[0,1].plot(radii,means[v]['rates'],'o-',label=f'Visit {v}')
 axs[1,0].plot(np.sqrt(edges[:-1]*edges[1:]),np.array(means[v]['masses'])/means[0]['masses'],'o-',label=f'Visit {v}')
axs[0,0].set(xlabel='Finest-visit phase / M',ylabel='Inward rest-mass rate [code units]',title='Near-hole accretion: startup versus first return')
axs[0,1].set(xlabel='COM radius / M',ylabel='Inward rest-mass rate [code units]',title='Matched-phase mean net inflow')
axs[0,1].set_xscale('log');axs[1,0].set_xscale('log')
axs[1,0].set(xlabel='Shell geometric-mean radius / M',ylabel='Shell rest mass / initial-visit mean',title='Identical-topology shell comparison')
axs[1,1].bar(np.arange(len(radii))-.17,changes['rates'],.34,label='Flux change')
axs[1,1].set_xticks(np.arange(len(radii)));axs[1,1].set_xticklabels([f'{r:g}' for r in radii])
axs[1,1].set(xlabel='COM radius / M',ylabel='Change between visits [%]',title='First-return flux drift relative to startup')
for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('MHD pilot: matched phases 50, 100, 200 M; only two finest visits\nInterpolation between saved times; no continuous budget inferred across frozen intervals',fontsize=11)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.out/'inflow_equilibrium.png');fig.savefig(a.out/'inflow_equilibrium.pdf');plt.close(fig)
result={'metric':str(metric),'radii':radii,'shell_edges':edges.tolist(),'hole_order':[(c,r) for c in (-16.,16.) for r in (1.5,2.,3.)],'matched_phases':phases,'finest_entries':entry,'samples':rows,'means':means,'changes_percent':changes,'normalization':'raw code units; no hydro Michel/Bondi normalization','caveats':'Only startup and first-return visits; angular differences do not bound spatial errors. Coarse shell assignment; no frozen-interval conservation residual.'}
(a.out/'inflow_equilibrium.json').write_text(json.dumps(result,indent=2)+'\n')
print('CHANGES',json.dumps(changes),flush=True)
