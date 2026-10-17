#!/usr/bin/env python3
"""Plot a magnetic pilot snapshot using its matching stationary ADM geometry."""
import argparse
import json
from pathlib import Path
import os
# Keep login-host/postprocessing thread use bounded, independently of solver OMP.
for _thread_setting in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_setting]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from plot_bondi_validation import load, midplane

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--snapshot', type=Path, required=True)
p.add_argument('--metric', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--rho-inf', type=float, default=1.)
p.add_argument('--extent', type=float, default=48.)
p.add_argument('--separation', type=float, default=32.)
p.add_argument('--excision-radius', type=float, default=.8)
p.add_argument('--plane',choices=['xy','xz'],default='xy')
a = p.parse_args()
assert np.isfinite(a.rho_inf) and a.rho_inf > 0 and a.extent > 0
assert a.separation > 0 and a.excision_radius >= 0
holes = (-a.separation/2, a.separation/2)
d, f = load(a.snapshot)
m, g = load(a.metric)
# Stationary metric reuse is valid only on exactly the same saved topology.
for key in ('mb_logical', 'mb_geometry'):
    assert np.array_equal(d[key], m[key]), 'Metric/snapshot topology mismatch'
assert all(np.isfinite(v).all() for v in f.values())
assert f['dens'].min() > 0 and f['press'].min() > 0
metric = np.array([[g['adm_gxx'], g['adm_gxy'], g['adm_gxz']],
                   [g['adm_gxy'], g['adm_gyy'], g['adm_gyz']],
                   [g['adm_gxz'], g['adm_gyz'], g['adm_gzz']]], dtype=np.float64)
det = (metric[0,0]*(metric[1,1]*metric[2,2]-metric[1,2]**2)
       -metric[0,1]*(metric[0,1]*metric[2,2]-metric[1,2]*metric[0,2])
       +metric[0,2]*(metric[0,1]*metric[1,2]-metric[1,1]*metric[0,2]))
assert np.isfinite(det).all() and (det > 0).all()
# Valencia output stores densitized Btilde=sqrt(gamma)*B_Eulerian.
B = np.array([f[f'bcc{i}'] for i in (1,2,3)], dtype=np.float64)/np.sqrt(det)
u = np.array([f['vel'+k] for k in 'xyz'], dtype=np.float64)
norm = lambda x,y: np.einsum('ab...,a...,b...->...', metric, x, y)
W2 = 1+norm(u,u)
b2 = (norm(B,B)+norm(B,u)**2)/W2
assert np.isfinite(b2).all() and (b2 >= 0).all() and b2.max() > 0
beta = np.full_like(b2, np.nan)
np.divide(2*f['press'], b2, out=beta, where=b2 > 0)
fields = [f['dens']/a.rho_inf, f['press']/f['dens'], beta]
titles = [r'$\rho/\rho_\infty$', r'$p/(\rho c^2)$', r'$\beta=2p/b^2$ (fluid frame)']
norms = [LogNorm(.01,100), LogNorm(1e-4,1), LogNorm(1e-3,1e3)]
fig, axes = plt.subplots(1,3,figsize=(15,5),dpi=140)
fig.subplots_adjust(left=.05,right=.97,bottom=.25,top=.82,wspace=.27)
def plane_blocks(field):
    if a.plane=='xy':
        yield from midplane(d,field)
    else:
        for mb,box in enumerate(d['mb_geometry']):
            if box[2]<=0<box[3]:
                j=int(-box[2]/(box[3]-box[2])*field.shape[2])
                yield [box[0],box[1],box[4],box[5]],field[mb,:,j,:]

for ax, field, title, norm_, cmap in zip(axes, fields, titles, norms, ['magma','inferno','viridis']):
    for bounds, image in plane_blocks(field):
        x0,x1,y0,y1 = bounds
        if x1 < -a.extent or x0 > a.extent or y1 < -a.extent or y0 > a.extent:
            continue
        xx=x0+(np.arange(image.shape[1])+.5)*(x1-x0)/image.shape[1]
        yy=y0+(np.arange(image.shape[0])+.5)*(y1-y0)/image.shape[0]
        masked=np.ma.masked_invalid(image)
        for hx in holes:
            masked=np.ma.masked_where((xx[None,:]-hx)**2+yy[:,None]**2 < a.excision_radius**2,masked)
        ax.pcolormesh(np.linspace(x0,x1,image.shape[1]+1),
                      np.linspace(y0,y1,image.shape[0]+1), masked,
                      norm=norm_,cmap=cmap,shading='flat',rasterized=True)
    for x in holes:
        ax.add_patch(Circle((x,0),a.excision_radius,color='black'))
        ax.plot(x,0,'+',color='cyan',ms=6)
    ax.set(xlim=(-a.extent,a.extent),ylim=(-a.extent,a.extent),aspect='equal',
           xlabel=r'$x/M$',ylabel=(r'$y/M$' if a.plane=='xy' else r'$z/M$'),title=title)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_,cmap=cmap),ax=ax,
                 fraction=.046,pad=.025,extend='both')
fig.suptitle(f'Magnetized binary Bondi pilot: t = {d["time"]:,.2f} M, {d["n_mbs"]} active blocks',y=.95)
fig.text(.5,.04,f'Rotating-frame {a.plane} slice; cyan crosses mark the holes.\n'
         'Actual saved AMR state. Global time is not uniform exposure across zoom zones; relaxation is not established.',
         ha='center',fontsize=9)
a.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out);plt.close(fig)
a.out.with_suffix('.json').write_text(json.dumps({'snapshot':str(a.snapshot),
    'metric':str(a.metric),'time':d['time'],'cycle':d['cycle'],'blocks':d['n_mbs'],
    'plane':a.plane,'rho_inf':a.rho_inf,'separation':a.separation,'excision_radius':a.excision_radius,'max_b_squared_including_excision':float(b2.max()),
    'zero_field_cells':int((b2==0).sum()),'metric_topology_matches':True},indent=2)+'\n')
