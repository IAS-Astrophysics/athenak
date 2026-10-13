#!/usr/bin/env python3
"""Near-hole rest-mass flux on finest-zone visits, for the stationary a=32 binary.

Positive flux is inward. Uses coordinate spheres fixed in the corotating frame,
outside excision; these are not apparent horizons. A finest-grid ADM snapshot
can be reused because the metric and hole coordinates are time independent.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_bondi_validation import load, AMRSampler, sphere_grid, michel


def snapshot_time(path):
    with path.open('rb') as f:
        assert f.readline().startswith(b'Athena')
        n = int(f.readline().split(b'=')[-1])
        header = dict(line.decode().strip().split('=', 1)
                      for line in [f.readline() for _ in range(n-1)])
    return float({k.strip(): v for k, v in header.items()}['time'])


def zoom_events(runs):
    log = '\n'.join((r/'rank-0.log').read_text() for r in runs)
    found = re.findall(r'CyclicZoom: new id = \d+ zone = (\d+).*?'
                       r'CyclicZoom: time = ([\d.eE+-]+)', log, re.S)
    events = [(0., 0, 0, 0.)]
    visit, entry = 0, 0.
    for z, t in found:
        z, t = int(z), float(t)
        if z == 0:
            visit += 1
            entry = t
        events.append((t, z, visit, entry))
    assert all(b[0] >= a[0] for a, b in zip(events, events[1:]))
    return events


def selected_snapshots(runs, events):
    selected, seen = [], set()
    for run in runs:
        for path in sorted((run/'bin').glob('*.mhd_w_bcc.*.bin')):
            t = snapshot_time(path)
            _, zone, visit, entry = next(e for e in reversed(events) if e[0] <= t)
            if zone == 0 and t not in seen:
                selected.append((path, t, visit, t-entry))
                seen.add(t)
    return selected


def geometry(ad, af, radius, center, nmu):
    points, normal, weights = sphere_grid(np.array([radius]), nmu, 2*nmu)
    points += np.array([center, 0., 0.])
    s = AMRSampler(ad, points)
    g = np.empty((len(points), 3, 3))
    for i, j, name in [(0,0,'xx'),(0,1,'xy'),(0,2,'xz'),
                       (1,1,'yy'),(1,2,'yz'),(2,2,'zz')]:
        g[:,i,j] = g[:,j,i] = s.sample(af['adm_g'+name])
    alpha = s.sample(af['adm_alpha'])
    beta = np.column_stack([s.sample(af['adm_beta'+k]) for k in 'xyz'])
    det = np.linalg.det(g)
    assert (det > 0).all() and (alpha > 0).all()
    return points, normal, weights, g, alpha, beta, np.sqrt(det)


def flux(d, f, geo, radius):
    points, normal, weights, g, alpha, beta, sqrtg = geo
    s = AMRSampler(d, points)
    rho = s.sample(f['dens'])
    u = np.column_stack([s.sample(f['vel'+k]) for k in 'xyz'])
    # Athena stores the normal-frame spatial four-velocity: utilde^i.
    W = np.sqrt(1 + np.einsum('ni,nij,nj->n', u, g, u))
    transport = alpha[:,None]*u - W[:,None]*beta
    integrand = sqrtg*rho*np.sum(transport*normal, axis=1)
    rate = -4*np.pi*radius**2*np.sum(integrand*weights)
    dx = (d['mb_geometry'][s.m,1]-d['mb_geometry'][s.m,0])/d['nx1_out_mb']
    assert np.isfinite(rate) and (rho > 0).all()
    assert radius/dx.max() >= 4, 'Extraction surface is too coarsely sampled'
    return float(rate), float(dx.max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--metric', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--list-only', action='store_true', help='List only finest-state paths for selective transfer')
    a = p.parse_args()
    selected = selected_snapshots(a.runs, zoom_events(a.runs))
    if a.list_only:
        print('\n'.join(str(s[0]) for s in selected))
        return
    assert selected and a.metric and a.out
    a.out.mkdir(parents=True, exist_ok=True)
    ad, af = load(a.metric)
    assert np.isclose((ad['mb_geometry'][:,1]-ad['mb_geometry'][:,0]).min()/ad['nx1_out_mb'], .125)
    radii = [1.5, 2., 3.]
    geos = {(r,c,n): geometry(ad,af,r,c,n) for r in radii for c in [-16.,16.] for n in [24,48]}
    mdot_b = float(michel(np.array([161.]))[4])
    records = []
    for path, t, visit, phase in selected:
        d, f = load(path)
        assert all(np.isfinite(v).all() for v in f.values())
        assert f['dens'].min()>0 and f['press'].min()>0
        assert all(np.count_nonzero(f[k])==0 for k in ['bcc1','bcc2','bcc3'])
        for center in [-16.,16.]:
            for r in radii:
                low, _ = flux(d,f,geos[r,center,24],r)
                high, dx = flux(d,f,geos[r,center,48],r)
                records.append(dict(time=t,visit=visit,phase=phase,center=center,radius=r,
                    mdot=high,mdot_over_bondi=high/mdot_b,dx_max=dx,
                    quadrature_change_over_bondi=abs(high-low)/mdot_b,
                    source=f'{path.parent.parent.name}/bin/{path.name}'))
    visits = sorted({s[2] for s in selected})
    # Only compare complete, equally sampled portions of the 50 M finest dwell.
    averages=[]
    phases=np.array([10.,20.,30.,40.])
    for visit in visits:
        for center in [-16.,16.]:
            for r in radii:
                rows=sorted([v for v in records if v['visit']==visit and v['center']==center and v['radius']==r], key=lambda v:v['phase'])
                xp=np.array([v['phase'] for v in rows]); yp=np.array([v['mdot_over_bondi'] for v in rows])
                if len(xp)>1 and xp[0]<=phases[0] and xp[-1]>=phases[-1]:
                    averages.append(dict(visit=visit,center=center,radius=r,
                        matched_phase_mean=float(np.interp(phases,xp,yp).mean())))
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,center in zip(axs[0],[-16.,16.]):
        for visit in visits:
            rows=[v for v in records if v['center']==center and v['radius']==2 and v['visit']==visit]
            ax.plot([v['phase'] for v in rows],[v['mdot_over_bondi'] for v in rows],'.-',label=f'Visit {visit}')
        ax.set(title=f'Hole at x={center:g} M; sphere r=2 M',xlabel='Time since finest-zone entry [M]',ylabel=r'$\dot M/\dot M_{\rm Bondi,total}$')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    for center,style in [(-16.,'-'),(16.,'--')]:
        for r,color in zip(radii,['#009e73','#0072b2','#d55e00']):
            rows=[v for v in averages if v['center']==center and v['radius']==r]
            x=[v['visit'] for v in rows];y=np.array([v['matched_phase_mean'] for v in rows])
            axs[1,0].plot(x,y,style,marker='o',color=color,label=f'x={center:g}, r={r:g}')
            if len(y)>1:
                change=100*np.diff(y)/np.maximum(np.abs(y[:-1]),1e-30)
                axs[1,1].plot(x[1:],change,style,marker='o',color=color)
    axs[1,0].set(title='Matched phases 10, 20, 30, 40 M',xlabel='Finest-zone visit',ylabel='Mean flux / total Bondi rate')
    axs[1,1].set(title='Change between sampled visits',xlabel='Finest-zone visit',ylabel='Flux change [%]')
    if averages:axs[1,0].legend(fontsize=7,ncol=2)
    else:
        for ax in axs[1]:ax.text(.5,.5,'Insufficient phase coverage yet',ha='center',transform=ax.transAxes)
    fig.suptitle('Near-hole inflow during finest-resolution visits — positive inward\nCoordinate spheres outside excision; no flux inferred during frozen intervals',fontsize=12)
    fig.savefig(a.out/'hole_accretion.png',dpi=170);fig.savefig(a.out/'hole_accretion.pdf')
    result=dict(metric=str(a.metric),normalization=mdot_b,records=records,matched_phase_means=averages,
                latest_finest_snapshot=str(selected[-1][0]),latest_finest_time=selected[-1][1],
                definition='-integral sqrt(gamma) rho (alpha utilde^i - W beta^i) n_i r^2 dOmega',
                caveat='Quadrature comparison does not measure spatial interpolation or grid convergence.')
    (a.out/'hole_accretion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(selected)} finest snapshots, visits {visits}; {len(averages)} matched-phase means')
    print('Max angular quadrature change / total Bondi rate:',max(v['quadrature_change_over_bondi'] for v in records))


if __name__ == '__main__':
    main()
