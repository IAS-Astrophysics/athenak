#!/usr/bin/env python3
"""Matched-phase MHD flux and shell-mass comparisons across smooth finest visits.

Native checkpoint metadata determines visit entries. Never extrapolates phases,
never mixes hard/smooth states, and never treats freezing as a continuous budget.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_bondi_validation import load,AMRSampler
from magnetic_pilot_metric import check_pilot
from bondi_hole_accretion import geometry,flux


class BoundedAMRSampler(AMRSampler):
 """Convex trilinear weights; clamp half-cell edges lacking ghost data.

 This avoids extrapolation across sharp magnetized gradients. It does not
 reconstruct neighboring blocks or provide a conservative face-flux budget.
 """
 def __init__(self,data,points):
  super().__init__(data,points)
  self.frac=np.clip(self.frac,0.,1.)

class NearestAMRSampler(BoundedAMRSampler):
 """A second bounded estimator used only to measure sampling sensitivity."""
 def __init__(self,data,points):
  super().__init__(data,points)
  self.frac=(self.frac>=.5).astype(float)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--runs',type=Path,nargs='+',required=True)
 p.add_argument('--metric',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--phases',type=float,nargs='+',default=[100.,150.,200.])
 p.add_argument('--finest-dwell',type=float,default=250.)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 ad,af=load(a.metric);check_pilot(ad['header']);b=ad['mb_geometry'];n=ad['nx1_out_mb']
 validations=[json.loads((r/'validation.json').read_text()) for r in a.runs]
 entries={}
 for v in validations:
  c=v['smooth_checkpoint']
  assert len(v['ranks'])==12 and v['conservative_sum_bytes']<30e9 and v['max_scaled_divb']<1e-9
  if c['zone']==0:
   t=c['next_time']-a.finest_dwell
   if c['id'] in entries:assert abs(entries[c['id']]-t)<1e-9
   entries[c['id']]=t
 events=sorted(entries);entry=np.array([entries[e] for e in events]);assert len(entry)>=2 and np.all(np.diff(entry)>0)
 radii=[24.,32.,64.,128.,256.,500.];edges=np.array(radii)
 surfaces=[(r,0.) for r in radii]+[(r,c) for c in (-16.,16.) for r in (1.5,2.,3.)]
 geo={(r,c,nq):geometry(ad,af,r,c,nq,sampler_class=BoundedAMRSampler) for r,c in surfaces for nq in (24,48)}
 nearest_geo={(r,c):geometry(ad,af,r,c,48,sampler_class=NearestAMRSampler) for r,c in surfaces}
 xyz=[]
 for i in range(3):
  q=b[:,2*i,None]+(np.arange(n)+.5)*(b[:,2*i+1]-b[:,2*i])[:,None]/n
  xyz.append(q[:,None,None,:] if i==0 else q[:,None,:,None] if i==1 else q[:,:,None,None])
 x,y,z=np.broadcast_arrays(*xyz);rr=np.sqrt(x*x+y*y+z*z);sel=(rr>=edges[0])&(rr<edges[-1]);shell=np.searchsorted(edges,rr[sel],side='right')-1
 G=np.array([[af['adm_gxx'][sel],af['adm_gxy'][sel],af['adm_gxz'][sel]],[af['adm_gxy'][sel],af['adm_gyy'][sel],af['adm_gyz'][sel]],[af['adm_gxz'][sel],af['adm_gyz'][sel],af['adm_gzz'][sel]]],dtype=float)
 det=np.linalg.det(np.moveaxis(G,(0,1),(-2,-1)));assert np.isfinite(det).all() and (det>0).all()
 vol=np.prod(b[:,[1,3,5]]-b[:,[0,2,4]],axis=1)/n**3;vol=np.broadcast_to(vol[:,None,None,None],rr.shape)[sel]
 rows=[];seen=set()
 for run,v in zip(a.runs,validations):
  for s in v['states']:
   path=run/'bin'/s['file']
   if s['max_level']!=12 or s['blocks']!=ad['n_mbs'] or not path.exists():continue
   d,f=load(path);c=v['smooth_checkpoint'];cycle=d['cycle']
   if cycle in seen:continue
   native=cycle==c['cycle'];t=c['time'] if native else d['time']
   visit=int(np.searchsorted(entry,t,side='right')-1);assert visit>=0
   phase=t-entry[visit];assert -.1<=phase<=a.finest_dwell+.1,(path,phase)
   for key in ('mb_geometry','mb_logical'):assert np.array_equal(d[key],ad[key]),(path,key)
   cfg=check_pilot(d['header']);assert cfg['coord/smooth_excision'].lower() in ('1','true')
   assert all(np.isfinite(vv).all() for vv in f.values()) and f['dens'].min()>0 and f['press'].min()>0
   assert any(np.any(f['bcc'+k]!=0) for k in '123')
   seen.add(cycle);U=np.array([f['vel'+k][sel] for k in 'xyz'],dtype=float);W=np.sqrt(1+np.einsum('ij...,i...,j...->...',G,U,U))
   masses=np.bincount(shell,weights=f['dens'][sel]*W*np.sqrt(det)*vol,minlength=len(edges)-1)
   rates=[];errors=[];sampling_errors=[]
   for r,center in surfaces:
    low,_=flux(d,f,geo[r,center,24],r,sampler_class=BoundedAMRSampler);high,_=flux(d,f,geo[r,center,48],r,sampler_class=BoundedAMRSampler)
    nearest,_=flux(d,f,nearest_geo[r,center],r,sampler_class=NearestAMRSampler)
    sampling_errors.append(abs(nearest-high))
    rates.append(high);errors.append(abs(high-low))
   rows.append(dict(time=t,header_time=d['time'],native_time_used=native,cycle=cycle,visit=visit,event=events[visit],phase=phase,rates=rates[:6],angular_errors=errors[:6],sampling_errors=sampling_errors[:6],hole_sampling_errors=sampling_errors[6:],hole_rates=rates[6:],hole_errors=errors[6:],masses=masses.tolist(),source=str(path)))
   print('Analyzed',t,'visit',visit,'phase',phase,flush=True)
 rows.sort(key=lambda q:q['time']);means=[];coverage=[]
 for visit in range(len(entry)):
  qs=[q for q in rows if q['visit']==visit];xp=np.array([q['phase'] for q in qs]);assert len(xp)>=2 and np.all(np.diff(xp)>0)
  coverage.append([float(xp.min()),float(xp.max())])
  assert xp.min()<=min(a.phases) and xp.max()>=max(a.phases),'Requested matched phases are not bracketed; do not extrapolate'
  mean=dict(visit=visit,event=events[visit],phase_values={})
  for key in ('rates','masses','hole_rates','angular_errors','hole_errors','sampling_errors','hole_sampling_errors'):
   values=np.array([q[key] for q in qs]);samples=np.array([np.interp(a.phases,xp,col) for col in values.T]).T
   mean[key]=samples.mean(axis=0).tolist();mean['phase_values'][key]=samples.tolist()
  means.append(mean)
 changes=[]
 for old,new in zip(means[:-1],means[1:]):
  change=dict(from_visit=old['visit'],to_visit=new['visit'])
  for key in ('rates','masses','hole_rates'):
   before=np.array(old[key]);after=np.array(new[key]);assert np.all(abs(before)>1e-30)
   change[key]=(100*(after/before-1)).tolist()
  change['rate_differences']=(np.array(new['rates'])-np.array(old['rates'])).tolist()
  change['flux_sign_changed']=(np.sign(new['rates'])!=np.sign(old['rates'])).tolist()
  changes.append(change)
 labels=['Smooth restart']+[f'Return {i}' for i in range(1,len(entry))];colors=plt.cm.viridis(np.linspace(.1,.85,len(entry)))
 fig,axs=plt.subplots(2,2,figsize=(11,8),dpi=150)
 for vi,color,label in zip(range(len(entry)),colors,labels):
  qs=[q for q in rows if q['visit']==vi]
  for hole,ax in enumerate(axs[0]):
   ax.errorbar([q['phase'] for q in qs],[q['hole_rates'][hole*3+1] for q in qs],yerr=[max(q['hole_errors'][hole*3+1],q['hole_sampling_errors'][hole*3+1]) for q in qs],fmt='o-',ms=3,color=color,label=label)
  axs[1,0].errorbar(radii,means[vi]['rates'],yerr=np.maximum(means[vi]['angular_errors'],means[vi]['sampling_errors']),fmt='o-',ms=4,color=color,label=label)
 for i,c in enumerate(changes):axs[1,1].plot(radii,c['rate_differences'],'o-',label=f'{labels[i]} to {labels[i+1]}')
 for hole,ax in enumerate(axs[0]):ax.set(xlabel='Finest-visit phase / M',ylabel='Inward rest-mass rate [code units]',title=f'Hole {hole+1} at x={(-16,16)[hole]} M; extraction r=2 M')
 axs[1,0].set(xlabel='COM radius / M',ylabel='Inward rest-mass rate [code units]',title='Mean over matched visit phases')
 axs[1,1].set(xlabel='COM radius / M',ylabel='Change in net inward rate [code units]',title='Successive visits at matched phases')
 axs[1,0].axhline(0,color='gray',lw=.7);axs[1,1].axhline(0,color='gray',lw=.7)
 for ax in axs[1]:ax.set_xscale('log')
 for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Smooth-excision MHD: accretion across two completed zoom cycles\nMatched phases '+', '.join(f'{x:g}' for x in a.phases)+' M; bars show max quadrature / interpolation sensitivity',fontsize=11)
 fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.out/'accretion.png');fig.savefig(a.out/'accretion.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4.5),dpi=150);centers=np.sqrt(edges[:-1]*edges[1:])
 for vi,color,label in zip(range(len(entry)),colors,labels):axs[0].plot(centers,np.array(means[vi]['masses'])/means[0]['masses'],'o-',color=color,label=label)
 for i,c in enumerate(changes):axs[1].plot(centers,c['masses'],'o-',label=f'{labels[i]} to {labels[i+1]}')
 axs[0].set(xlabel='Shell geometric-mean radius / M',ylabel='Rest mass / smooth-restart visit mean',title='Fixed shells: [24,32,64,128,256,500] M')
 axs[1].set(xlabel='Shell geometric-mean radius / M',ylabel='Shell rest-mass change [%]',title='Successive matched-phase comparisons')
 for ax in axs:ax.set_xscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Identical finest-stage topology; no continuous conservation residual across frozen intervals',fontsize=10)
 fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.out/'shell_mass.png');fig.savefig(a.out/'shell_mass.pdf');plt.close(fig)
 result=dict(metric=str(a.metric),runs=[str(r) for r in a.runs],radii=radii,shell_edges=edges.tolist(),hole_order=[(c,r) for c in (-16.,16.) for r in (1.5,2.,3.)],matched_phases=a.phases,finest_entries=entry.tolist(),events=events,phase_coverage=coverage,samples=rows,means=means,changes_percent=changes,interpolation='Bounded trilinear; half-cell block edges clamped to center hull. Nearest-cell comparison separately recorded.',normalization='raw code units; no hydro normalization',caveats='Matched-phase means interpolate bracketed saved times, not full-visit time averages. Quadrature and nearest-cell sensitivities are not rigorous spatial error bounds. Shell cell assignment is coarse. No continuous mass balance through frozen intervals. Near-drain pressure is timestep sensitive.')
 (a.out/'inflow_equilibrium.json').write_text(json.dumps(result,indent=2)+'\n');print('CHANGES',json.dumps(changes),flush=True)

if __name__=='__main__':main()
