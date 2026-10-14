#!/usr/bin/env python3
"""Instantaneous COM-sphere entropy and inward/outward rest-mass transport."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bondi_hole_accretion import geometry
from plot_bondi_validation import load, AMRSampler

root=Path(__file__).resolve().parents[1]
out=root/'docs/bondi_visualizations/hole_accretion'
info=json.loads((out/'hole_accretion.json').read_text())
d,f=load(root/info['latest_finest_snapshot'])
ad,af=load(root/info['metric'])
rows=[]
for r in [24.,32.,48.,64.,96.,128.,161.226654]:
    points,normal,weights,g,alpha,beta,sqrtg=geometry(ad,af,r,0.,96)
    s=AMRSampler(d,points);rho=s.sample(f['dens']);p=s.sample(f['press'])
    u=np.column_stack([s.sample(f['vel'+k]) for k in 'xyz'])
    W=np.sqrt(1+np.einsum('ni,nij,nj->n',u,g,u))
    jr=sqrtg*rho*np.sum((alpha[:,None]*u-W[:,None]*beta)*normal,axis=1)
    scale=4*np.pi*r*r/info['normalization']
    incoming=scale*np.sum(np.maximum(-jr,0)*weights)
    outgoing=scale*np.sum(np.maximum(jr,0)*weights)
    entropy=p/rho**(5/3)
    rows.append(dict(radius=r,incoming=float(incoming),outgoing=float(outgoing),
        entropy_proxy_mean=float(np.sum(entropy*weights)),
        entropy_proxy_angular_rms=float(np.sqrt(np.sum((entropy-np.sum(entropy*weights))**2*weights)))))
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
r=[q['radius'] for q in rows]
axes[0].plot(r,[q['incoming'] for q in rows],'o-',label='Inward')
axes[0].plot(r,[q['outgoing'] for q in rows],'o-',label='Outward')
axes[0].set(xlabel='COM radius [M]',ylabel='Mass flux / analytic Bondi rate',title='Instantaneous inward / outward flux');axes[0].legend()
axes[1].errorbar(r,[q['entropy_proxy_mean'] for q in rows],yerr=[q['entropy_proxy_angular_rms'] for q in rows],fmt='o-',capsize=3,label='Mean ± angular RMS')
axes[1].axhline(1,color='gray',ls='--',label='Initial exterior K=1')
axes[1].set(xlabel='COM radius [M]',ylabel=r'$K=p/\rho^{5/3}$',title='Entropy proxy; RMS is spatial variation');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle(f"Latest finest-stage snapshot: t={d['time']:g} M; not a time-averaged turbulence diagnostic")
fig.savefig(out/'entropy_transport.png',dpi=175);fig.savefig(out/'entropy_transport.pdf')
(out/'entropy_transport.json').write_text(json.dumps(dict(time=d['time'],rows=rows),indent=2)+'\n')
print(json.dumps(dict(time=d['time'],rows=rows)))
