#!/usr/bin/env python3
"""Audit logged dwell times and estimate local advection times from saved data."""
import json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bondi_hole_accretion import geometry
from plot_bondi_validation import load,AMRSampler

root=Path(__file__).resolve().parents[1]
out=root/'docs/bondi_visualizations/hole_accretion'
h=json.loads((out/'hole_accretion.json').read_text())
d,f=load(root/h['latest_finest_snapshot']);ad,af=load(root/h['metric'])
records=[]
for r in [1.5,2.,3.,4.,8.]:
    q,normal,w,g,alpha,beta,sqrtg=geometry(ad,af,r,-16.,96)
    s=AMRSampler(d,q);rho=s.sample(f['dens'])
    u=np.column_stack([s.sample(f['vel'+a]) for a in 'xyz'])
    W=np.sqrt(1+np.einsum('ni,nij,nj->n',u,g,u))
    transport=alpha[:,None]*u-W[:,None]*beta
    inward=np.maximum(-np.sum(transport*normal,axis=1),0)
    # (dM/dlnr)/mdot_in in coordinate time; local estimate, not a trajectory integral.
    tin=r*np.sum(rho*W*sqrtg*w)/np.sum(rho*sqrtg*inward*w)
    records.append(dict(radius=r,dynamical_scale=float(np.sqrt(r**3/.5)),
                        inflow_estimate=float(tin),finest_dwell_over_inflow=float(50/tin)))
logs='\n'.join((run/'rank-0.log').read_text() for run in sorted((root/'build-mac-zoom-opt/aurora-sane/gradual-chain').iterdir()) if (run/'rank-0.log').exists())
events=[(float(t),int(z),float(dt)) for z,t,dt in re.findall(
    r'CyclicZoom: new id = \d+ zone = (\d+).*?CyclicZoom: time = ([\d.eE+-]+) runtime = ([\d.eE+-]+)',logs,re.S)]
final=124768.4;clock=0.;zone=0;dwell={}
for t,z,dt in events+[(final,0,0.)]:
    assert t>=clock
    dwell[zone]=dwell.get(zone,0)+t-clock;clock=t;zone=z
expected=[50.,0.,0.,10*np.sqrt(4**3/.5),320.,10*np.sqrt(16**3/.5),10*np.sqrt(32**3),5120.]
for _,z,dt in events:assert np.isclose(dt,expected[z],rtol=1e-5,atol=1e-6)
period=2*np.pi*32**1.5
result=dict(final_global_time=final,binary_period=period,finest_visits=11,
    requested_residences=expected,logged_dwell=dwell,finest_binary_orbits=dwell[0]/period,
    radius_by_zone=[.5,1,2,4,8,16,32,64],local_times=records,
    snapshot_time=d['time'],mask_cells_per_radius=4,
    caveat='Dwell totals use rounded logs; inflow estimates use one sphere snapshot and coordinate-time transport, not turbulent correlation or viscous times.')
(out/'residence_audit.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
axes[0].bar(range(8),expected)
axes[0].set_yscale('symlog',linthresh=50)
axes[0].set(xlabel='Zoom zone',ylabel='Requested dwell per visit [M]',title='Implemented mass-scaled hydro schedule',xticks=range(8))
r=[v['radius'] for v in records]
axes[1].plot(r,[v['inflow_estimate'] for v in records],'o-',label=r'Estimated $(dM/d\ln r)/\dot M_{in}$')
axes[1].plot(r,[v['dynamical_scale'] for v in records],'o--',label=r'$\sqrt{r^3/m_{BH}}$ (scale estimate)')
axes[1].axhline(50,color='black',ls=':',label='Finest visit: 50 M')
axes[1].set(xlabel='Radius about one hole [M]',ylabel='Coordinate-time scale [M]',title=f'Local flow at t={d["time"]:g} M')
axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(out/'residence_audit.png',dpi=175);fig.savefig(out/'residence_audit.pdf')
print(json.dumps(result))
