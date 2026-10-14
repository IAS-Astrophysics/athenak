#!/usr/bin/env python3
"""Compare COM flux and shell rest mass out to rB at matched finest-stage phases.

Cell-center shell assignment has coarse-grid boundary error. Temporal changes
on the identical restored mesh are useful; absolute shell masses are approximate.
This does not infer a continuous conservation residual across frozen stages.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bondi_hole_accretion import geometry, flux
from plot_bondi_validation import load

root=Path(__file__).resolve().parents[1]
out=root/'docs/bondi_visualizations/hole_accretion'
data=json.loads((out/'hole_accretion.json').read_text())
base=root/'build-mac-zoom-opt/aurora-sane/gradual-chain'
ad,af=load(root/data['metric'])
radii=np.array([24.,32.,48.,64.,80.,96.,112.,128.,144.,161.226654])
edges=np.array([24.,32.,48.,64.,96.,128.,161.226654])
geos={(r,n):geometry(ad,af,r,0.,n) for r in radii for n in [48,96]}
mb=ad['mb_geometry']; n=ad['nx1_out_mb']
z,y,x=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n,(np.arange(n)+.5)/n,indexing='ij')
rr=np.sqrt(sum((mb[:,2*i,None,None,None]+(mb[:,2*i+1]-mb[:,2*i])[:,None,None,None]*q)**2 for i,q in enumerate([x,y,z])))
volume=np.prod(mb[:,[1,3,5]]-mb[:,[0,2,4]],axis=1)/n**3
mask=(rr>=edges[0])&(rr<edges[-1])
gxx,gxy,gxz,gyy,gyz,gzz=[af['adm_g'+k][mask] for k in ['xx','xy','xz','yy','yz','zz']]
det=gxx*(gyy*gzz-gyz**2)-gxy*(gxy*gzz-gxz*gyz)+gxz*(gxy*gyz-gxz*gyy)
assert (det>0).all()
vol=np.broadcast_to(volume[:,None,None,None],rr.shape)[mask]
shell=np.searchsorted(edges,rr[mask],side='right')-1
samples=[]
for rec in data['records']:
    if rec['radius']!=2 or rec['center']!=-16:continue
    d,f=load(base/rec['source'])
    assert np.array_equal(d['mb_geometry'],mb), 'Shell comparison requires identical restored topology'
    assert all(np.isfinite(v).all() for v in f.values())
    assert (f['dens']>0).all() and (f['press']>0).all()
    ux,uy,uz=[f['vel'+k][mask] for k in 'xyz']
    W=np.sqrt(1+gxx*ux**2+gyy*uy**2+gzz*uz**2+2*(gxy*ux*uy+gxz*ux*uz+gyz*uy*uz))
    mass=np.bincount(shell,weights=f['dens'][mask]*W*np.sqrt(det)*vol,minlength=len(edges)-1)
    rates=[];errors=[]
    for r in radii:
        low,_=flux(d,f,geos[r,48],r);high,dx=flux(d,f,geos[r,96],r)
        rates.append(high/data['normalization']);errors.append(abs(high-low)/data['normalization'])
    samples.append(dict(visit=rec['visit'],phase=rec['phase'],time=rec['time'],rates=rates,masses=mass.tolist(),angular_errors=errors))
means=[]
for visit in sorted({r['visit'] for r in samples}):
    rows=sorted([r for r in samples if r['visit']==visit],key=lambda r:r['phase']);xp=[r['phase'] for r in rows]
    if len(rows)<2 or xp[0]>10 or xp[-1]<40:continue
    def average(key):
        a=np.array([r[key] for r in rows])
        return np.array([np.interp([10,20,30,40],xp,v).mean() for v in a.T])
    means.append(dict(visit=visit,rates=average('rates').tolist(),masses=average('masses').tolist()))
assert len(means)>=2
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for m in means:
    axs[0,0].plot(radii,m['rates'],'o-',label=f"Cycle {m['visit']}")
axs[0,0].set(xlabel='COM radius [M]',ylabel='Mass inflow / analytic Bondi rate',title='Matched-phase radial inflow')
axs[0,0].legend()
x=[m['visit'] for m in means];rates=np.array([m['rates'] for m in means]);masses=np.array([m['masses'] for m in means])
for j,r in enumerate(radii):axs[0,1].plot(x[1:],100*np.diff(rates[:,j])/abs(rates[:-1,j]),'o-',label=f'{r:g} M')
axs[0,1].set(xlabel='Zoom cycle',ylabel='Change from previous sampled cycle [%]',title='Residual drift in inflow rate')
axs[0,1].legend(fontsize=7,ncol=2)
for j in range(len(edges)-1):
    label=f'{edges[j]:g}–{edges[j+1]:.0f} M'
    axs[1,0].plot(x,masses[:,j]/masses[0,j],'o-',label=label)
    axs[1,1].plot(x[1:],100*np.diff(masses[:,j])/masses[:-1,j],'o-',label=label)
axs[1,0].set(xlabel='Zoom cycle',ylabel='Shell rest mass / first sampled cycle',title='Gas accumulation in fixed COM shells')
axs[1,1].set(xlabel='Zoom cycle',ylabel='Shell-mass change per sampled cycle [%]',title='Residual shell-mass drift')
axs[1,0].legend(fontsize=7,ncol=2)
for ax in axs.flat:ax.grid(alpha=.2)
fig.suptitle('Inflow-equilibrium assessment out to the Bondi radius (161.23 M)\nPhases 10, 20, 30, 40 M of finest visits; fixed mesh for shell comparisons',fontsize=12)
fig.savefig(out/'reservoir_equilibrium.png',dpi=175);fig.savefig(out/'reservoir_equilibrium.pdf')
result=dict(radii=radii.tolist(),shell_edges=edges.tolist(),samples=samples,means=means,
    last_flux_change_percent=(100*(rates[-1]/rates[-2]-1)).tolist(),
    last_mass_change_percent=(100*(masses[-1]/masses[-2]-1)).tolist(),
    caveat='No continuous mass-budget closure inferred; coarse-grid shell assignment and spatial interpolation errors remain.')
(out/'reservoir_equilibrium.json').write_text(json.dumps(result,indent=2)+'\n')
print('Last flux changes %:',result['last_flux_change_percent'])
print('Last shell mass changes %:',result['last_mass_change_percent'])
