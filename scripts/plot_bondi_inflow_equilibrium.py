#!/usr/bin/env python3
"""Assess local cyclic inflow repeatability; no new simulations are launched."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bondi_hole_accretion import geometry, flux
from plot_bondi_validation import load

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'docs/bondi_visualizations/hole_accretion'
data=json.loads((out/'hole_accretion.json').read_text())
base=ROOT/'build-mac-zoom-opt/aurora-sane/gradual-chain'
ad,af=load(ROOT/data['metric'])
radii=np.array([1.25,1.5,2,3,4,6,8,10,12])
profiles=[]
for visit in [4,5,6]:
    candidates=[r for r in data['records'] if r['visit']==visit and r['radius']==2 and r['center']==-16]
    record=min(candidates,key=lambda r:abs(r['phase']-30))
    d,f=load(base/record['source'])
    for center in [-16.,16.]:
        rates=[];errors=[]
        for r in radii:
            low,_=flux(d,f,geometry(ad,af,r,center,24),r)
            high,dx=flux(d,f,geometry(ad,af,r,center,48),r)
            rates.append(high/data['normalization']);errors.append(abs(high-low)/data['normalization'])
        profiles.append(dict(visit=visit,center=center,time=record['time'],phase=record['phase'],
                             radii=radii.tolist(),rates=rates,angular_errors=errors))

fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={4:'#009e73',5:'#d55e00',6:'#0072b2'}
for ax,center in zip(axs[0],[-16.,16.]):
    for q in profiles:
        if q['center']==center:
            ax.plot(radii,q['rates'],'o-',color=colors[q['visit']],label=f"Cycle {q['visit']}, phase {q['phase']:.1f} M")
    ax.axvspan(.8,1.,color='gray',alpha=.2,label='Excision to nominal horizon')
    ax.set(xlabel='Coordinate radius about hole [M]',ylabel=r'$\dot M/\dot M_{\mathrm{Bondi,total}}$',
           title=f'Radial flux profile: hole at x={center:g} M',xlim=(.8,12.3))
    ax.legend(fontsize=8);ax.grid(alpha=.2)
summaries=[]
for center,style in [(-16.,'-'),(16.,'--')]:
    means=[q for q in data['matched_phase_means'] if q['radius']==2 and q['center']==center]
    axs[1,0].plot([q['visit'] for q in means],[q['matched_phase_mean'] for q in means],style,
                  marker='o',label=f'Hole x={center:g}')
    for visit in [4,5,6]:
        rows=[r for r in data['records'] if r['center']==center and r['radius']==2 and r['visit']==visit]
        phase=np.array([r['phase'] for r in rows]);y=np.array([r['mdot_over_bondi'] for r in rows])
        reference=np.interp(10,phase,y)
        axs[1,1].plot(phase,100*(y/reference-1),style,marker='.',color=colors[visit],
                      label=f'Cycle {visit}, x={center:g}')
        change=float(100*(np.interp(40,phase,y)/reference-1))
        summaries.append(dict(visit=visit,center=center,within_visit_change_percent=change))
axs[1,0].set(xlabel='Completed zoom cycle',ylabel='Mean flux / total Bondi rate',title='Same finest-stage phases: 10, 20, 30, 40 M')
axs[1,1].set(xlabel='Time since finest-stage entry [M]',ylabel='Change relative to phase 10 M [%]',title='Residual evolution within the finest stage (r=2 M)')
for ax in axs[1]:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Inflow-equilibrium checks after six cyclic-zoom cycles\nNear-hole coordinate-sphere fluxes; the outer reservoir is not tested here',fontsize=13)
fig.savefig(out/'inflow_equilibrium.png',dpi=175);fig.savefig(out/'inflow_equilibrium.pdf')
(out/'inflow_equilibrium.json').write_text(json.dumps(dict(profiles=profiles,within_visit=summaries),indent=2)+'\n')
for q in profiles:
    if q['visit']==6:
        y=np.array(q['rates']);print(q['center'],'final profile',y,'spread 1.5-3M %',100*np.ptp(y[1:4])/np.mean(y[1:4]),'spread to12M %',100*np.ptp(y)/np.mean(y))
print(summaries)
