#!/usr/bin/env python3
"""Paper-style figures from saved Bondi controls; never launches a simulation.

Requires NumPy and Matplotlib. The spherical flux uses the saved ADM fields,
including the rotating shift. Surfaces are sampled with Gauss-Legendre angular
quadrature and local linear interpolation/extrapolation of active AMR cells.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vis/python'))
from bin_convert import read_binary

CASES = ('upstream', 'single_inertial', 'single_rotating', 'binary_rotating')
LABELS = ('Original Bondi pgen', 'Single, inertial', 'Single, rotating', 'Binary, rotating')
COLORS = ('#64748b', '#0072b2', '#009e73', '#d55e00')
STYLES = (':', '--', '-', '-')
OMEGA = 32.0**-1.5
GAMMA = 5.0/3.0
ORBITAL_PERIOD = 2*np.pi/OMEGA


def load(path):
    d = read_binary(str(path))
    return d, {k: np.asarray(v) for k, v in d['mb_data'].items()}


def fluid_paths(root, name):
    var = 'hydro_w' if name == 'upstream' else 'mhd_w_bcc'
    return sorted((root/name/'bin').glob(f'*.{var}.*.bin'))


def michel(r):
    """Same HSW transonic solution as the simulation, in total-mass units."""
    n, rc = 1/(GAMMA-1), 8.0
    uc2 = 0.5/rc
    tc = n/(n+1)*uc2/(1-(n+3)*uc2)
    c1 = -tc**n*np.sqrt(uc2)*rc**2
    c2 = (1+(n+1)*tc)**2*(1-1.5/rc)
    def residual(t):
        return (1+(n+1)*t)**2*(1-2/r+(c1/(r*r*t**n))**2)-c2
    lo, hi = np.full_like(r, np.log(1e-12)), np.full_like(r, np.log(100))
    f = 0.3819660112501051
    a, b = lo+f*(hi-lo), hi-f*(hi-lo)
    fa, fb = residual(np.exp(a)), residual(np.exp(b))
    for _ in range(72):
        left = fa < fb
        hi, lo = np.where(left, b, hi), np.where(left, lo, a)
        a, b = lo+f*(hi-lo), hi-f*(hi-lo)
        fa, fb = residual(np.exp(a)), residual(np.exp(b))
    tmin = np.exp(0.5*(lo+hi))
    lo, hi = np.where(r < rc, 1e-12, tmin), np.where(r < rc, tmin, 100)
    flo = residual(lo)
    for _ in range(72):
        t = 0.5*(lo+hi)
        ft = residual(t)
        same = (ft > 0) == (flo > 0)
        lo, hi = np.where(same, t, lo), np.where(same, hi, t)
        flo = np.where(same, ft, flo)
    t = np.where(abs(r-rc) < 1e-9, tc, 0.5*(lo+hi))
    tinf = (np.sqrt(c2)-1)/(n+1)
    rhoinf = tinf**n
    rb = 2/(GAMMA*tinf/np.sqrt(c2))
    return t**n, t, rhoinf, tinf, -4*np.pi*c1, rb


class AMRSampler:
    """Find leaf blocks, then interpolate cell centers, including half-cell edges."""
    def __init__(self, data, points):
        g = data['mb_geometry']
        self.nx = data['nx1_out_mb']
        assert self.nx == data['nx2_out_mb'] == data['nx3_out_mb']
        lower = np.array([data[f'x{i}min'] for i in (1, 2, 3)])
        widths = g[:, 1]-g[:, 0]
        root_width = widths.max()
        levels = np.rint(np.log2(root_width/widths)).astype(int)
        root_n = int(round((data['x1max']-data['x1min'])/root_width))
        m = np.full(len(points), -1, dtype=int)
        for level in np.unique(levels)[::-1]:
            block_ids = np.flatnonzero(levels == level)
            width = root_width/2**level
            n = root_n*2**level
            ijk = np.rint((g[block_ids][:, (0, 2, 4)]-lower)/width).astype(np.int64)
            keys = ijk[:, 0]+n*(ijk[:, 1]+n*ijk[:, 2])
            order = np.argsort(keys)
            keys, block_ids = keys[order], block_ids[order]
            q = np.flatnonzero(m < 0)
            indices = np.floor((points[q]-lower)/width).astype(np.int64)
            qkeys = indices[:, 0]+n*(indices[:, 1]+n*indices[:, 2])
            pos = np.searchsorted(keys, qkeys)
            safe = np.minimum(pos, len(keys)-1)
            found = (pos < len(keys)) & (keys[safe] == qkeys)
            m[q[found]] = block_ids[safe[found]]
        assert (m >= 0).all(), 'A spherical sample lies outside the leaf mesh'
        mins = g[m][:, (0, 2, 4)]
        dx = widths[m]/self.nx
        index = (points-mins)/dx[:, None]-0.5
        self.base = np.clip(np.floor(index).astype(int), 0, self.nx-2)
        self.frac = index-self.base
        self.m = m

    def sample(self, field):
        v = np.zeros(len(self.m))
        i, j, k = self.base.T
        f = self.frac
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    w = (f[:, 0] if a else 1-f[:, 0])
                    w = w*(f[:, 1] if b else 1-f[:, 1])
                    w = w*(f[:, 2] if c else 1-f[:, 2])
                    v += w*field[self.m, k+c, j+b, i+a]
        return v


def sphere_grid(r, nmu, nphi):
    mu, w = np.polynomial.legendre.leggauss(nmu)
    phi = (np.arange(nphi)+0.5)*2*np.pi/nphi
    s = np.sqrt(1-mu*mu)
    unit = np.stack(np.broadcast_arrays(s[:, None]*np.cos(phi),
                                       s[:, None]*np.sin(phi), mu[:, None]), axis=-1)
    unit = unit.reshape(-1, 3)
    weights = np.broadcast_to(w[:, None]/(2*nphi), (nmu, nphi)).ravel()
    points = (r[:, None, None]*unit[None, :, :]).reshape(-1, 3)
    return points, unit, weights


def profile(root, name, last, r, nmu, nphi):
    path = fluid_paths(root, name)[-1 if last else 0]
    data, f = load(path)
    points, unit, weights = sphere_grid(r, nmu, nphi)
    sampler = AMRSampler(data, points)
    rho = sampler.sample(f['dens'])
    pressure = sampler.sample(f['eint'])*(GAMMA-1) if name == 'upstream' else sampler.sample(f['press'])
    shape = (len(r), len(unit))
    def average(v):
        return np.sum(v.reshape(shape)*weights, axis=1)
    # Scalar averages include the gas outside the numerical excision.
    good = rho > 1e-7
    result = {'rho': average(np.where(good, rho, 0))/average(good),
              'temperature': average(np.where(good, pressure/rho, 0))/average(good)}
    # COM spheres intersect the individual binary holes near r=16; use only
    # enclosing surfaces with a comfortable buffer for that flux diagnostic.
    valid_r = r >= (24 if name == 'binary_rotating' else 3)
    valid = np.repeat(valid_r, len(unit))
    q = points[valid]
    rr = np.linalg.norm(q, axis=1)
    normal = q/rr[:, None]
    u = np.column_stack([sampler.sample(f['vel'+a])[valid] for a in 'xyz'])
    if name == 'upstream':
        alpha = 1/np.sqrt(1+2/rr)
        beta = 2/(rr+2)[:, None]*normal
        un = np.sum(u*normal, axis=1)
        lorentz = np.sqrt(1+np.sum(u*u, axis=1)+2/rr*un*un)
        sqrt_minus_g = np.ones(len(q))
    else:
        ap = sorted((root/name/'bin').glob('*.adm.*.bin'))[-1 if last else 0]
        adata, af = load(ap)
        assert data['time'] == adata['time']
        assert np.array_equal(data['mb_logical'], adata['mb_logical'])
        metric = {key: sampler.sample(af['adm_'+key])[valid]
                  for key in ('gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz', 'alpha')}
        alpha = metric['alpha']
        beta = np.column_stack([sampler.sample(af['adm_beta'+a])[valid] for a in 'xyz'])
        a,b,c,d,e,h = [metric[k] for k in ('gxx','gxy','gxz','gyy','gyz','gzz')]
        det = a*(d*h-e*e)-b*(b*h-c*e)+c*(b*e-c*d)
        norm = a*u[:,0]**2+d*u[:,1]**2+h*u[:,2]**2
        norm += 2*(b*u[:,0]*u[:,1]+c*u[:,0]*u[:,2]+e*u[:,1]*u[:,2])
        assert (det > 0).all() and (alpha > 0).all() and (norm >= 0).all()
        lorentz = np.sqrt(1+norm)
        sqrt_minus_g = alpha*np.sqrt(det)
    ut = lorentz/alpha
    spatial_rot = u-ut[:, None]*beta
    ur = np.sum(spatial_rot*normal, axis=1)
    omega = OMEGA if name.endswith('_rotating') else 0.0
    grid_velocity = omega*np.column_stack([-q[:,1], q[:,0], np.zeros(len(q))])
    spatial_inertial = spatial_rot+ut[:, None]*grid_velocity
    assert np.allclose(ur, np.sum(spatial_inertial*normal, axis=1), rtol=1e-10, atol=1e-12)
    flux = sqrt_minus_g*rho[valid]*ur
    rates = -4*np.pi*r[valid_r]**2*np.sum(flux.reshape(-1,len(unit))*weights, axis=1)
    result['mdot'] = np.full(len(r), np.nan)
    result['mdot'][valid_r] = rates
    result['time'] = data['time']
    return result


def midplane(data, density):
    """Linear z=0 slice from the two cell centers bracketing the symmetry plane."""
    g = data['mb_geometry']
    mirrors = {tuple(v): i for i, v in enumerate(g)}
    patches = []
    for m in np.flatnonzero((g[:,4] == 0) & (g[:,5] > 0)):
        key = tuple(g[m, :4])+(-g[m,5], -g[m,4])
        assert key in mirrors, 'Mid-plane block lacks its reflected counterpart'
        val = 0.5*(density[m,0]+density[mirrors[key],-1])
        patches.append((g[m,:4], val))
    assert patches
    return patches


def draw_slice(ax, data, fields, name, half, rhoinf, grid=False, mask=None, wide=False):
    norm = LogNorm(1, 1e3)
    segments = []
    for bounds, field in midplane(data, fields['dens']):
        x0,x1,y0,y1 = bounds
        if x1 < -half or x0 > half or y1 < -half or y0 > half:
            continue
        ax.pcolormesh(np.linspace(x0,x1,field.shape[1]+1),
                      np.linspace(y0,y1,field.shape[0]+1), field/rhoinf,
                      cmap='magma', norm=norm, shading='flat', zorder=-2)
        if grid:
            segments.append([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)])
    ax.set_rasterization_zorder(0)
    if grid:
        ax.add_collection(LineCollection(segments, colors='white', linewidths=.28, alpha=.35))
    holes = [(-16,1), (16,1)] if name == 'binary_rotating' else [(0,2)]
    for x,rh in holes:
        ax.add_patch(Circle((x,0),rh,facecolor='black',edgecolor='white',linewidth=.55,zorder=3))
    if mask:
        ax.add_patch(Circle((0,0),mask,facecolor='none',edgecolor='#00e5ff',linewidth=1.1,
                            linestyle='--',zorder=4))
    if wide:
        ax.add_patch(Circle((0,0),161.227,fill=False,color='white',linewidth=1,linestyle='--'))
        ax.add_patch(Circle((0,0),1/OMEGA,fill=False,color='#a1d99b',linewidth=.9,linestyle=':'))
    ax.set(xlim=(-half,half),ylim=(-half,half),aspect='equal')
    ax.set_xlabel(r'$x/M$')
    ax.set_ylabel(r'$y/M$')
    return plt.cm.ScalarMappable(norm=norm,cmap='magma')


def save(fig, out, name):
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.pdf'),dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nmu',type=int,default=64)
    p.add_argument('--nphi',type=int,default=128)
    args=p.parse_args()
    root,out=args.root.resolve(),args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
                         'axes.labelsize':12,'axes.titlesize':12,'xtick.direction':'in',
                         'ytick.direction':'in','xtick.top':True,'ytick.right':True,
                         'axes.linewidth':.8,'savefig.pad_inches':.12})
    r=np.geomspace(3,240,100)
    analytic_rho,analytic_t,rhoinf,tinf,mdotb,rb=michel(r)
    profiles={}
    for name in CASES:
        paths=fluid_paths(root,name)
        if not paths:
            raise ValueError(f'No fluid outputs for {name}')
        first,_=load(paths[0]); final,_=load(paths[-1])
        if first['time'] != 0 or abs(final['time']-4) > 1e-6:
            raise ValueError('These figures require the matched t=0 and t=4 M controls')
        profiles[name]={}
        for label,last in [('initial',False),('final',True)]:
            print('Sampling',name,label,flush=True)
            profiles[name][label]=profile(root,name,last,r,args.nmu,args.nphi)
    # Measure sampling bias independently of evolution; do not present it as
    # a solver error or infer time variability from this short frozen-zone run.
    bias={n:float(np.nanmax(abs(profiles[n]['initial']['mdot']/mdotb-1)))
          for n in CASES[:3]}
    print('Checking angular quadrature convergence',flush=True)
    higher=profile(root,'single_inertial',False,r,args.nmu+32,args.nphi+64)
    angular_error=float(np.max(abs(higher['mdot']-
                                   profiles['single_inertial']['initial']['mdot']))/mdotb)
    arrays={'radius':r,'analytic_rho':analytic_rho,'analytic_temperature':analytic_t}
    for name in CASES:
        for when in ('initial','final'):
            for field in ('rho','temperature','mdot'):
                arrays[name+'_'+when+'_'+field]=profiles[name][when][field]
    np.savez_compressed(out/'radial_data.npz',**arrays)
    names=list(arrays)
    np.savetxt(out/'radial_data.csv',np.column_stack([arrays[n] for n in names]),
               delimiter=',',header=','.join(names),comments='')
    metadata={'source':'Aurora debug job 8924219','global_time':4,'gamma':GAMMA,
              'orbital_period':ORBITAL_PERIOD,'global_orbits':4/ORBITAL_PERIOD,
              'relaxation_complete':False,
              'sonic_radius':8,'bondi_radius':rb,'rho_infinity':rhoinf,
              'temperature_infinity':tinf,'analytic_mdot':mdotb,
              'angular_quadrature':[args.nmu,args.nphi],
              'higher_angular_quadrature':[args.nmu+32,args.nphi+64],
              'angular_convergence_max_abs_flux_fraction':angular_error,
              'initial_single_hole_flux_sampling_bias_max_fraction':bias,
              'flux_definition':'-r^2 integral alpha sqrt(det gamma) rho u^r dOmega',
              'binary_flux_min_radius':24,
              'notes':['Instantaneous snapshots, not steady or time-averaged profiles.',
                       'Both configurations start with the central M=1 spherical Bondi seed; the binary central peak is unrelaxed initial gas.',
                       'Global time is only 0.00352 binary orbit; stored interior patches evolve for less time.',
                       'Binary radial flux is through COM spheres, not individual horizon accretion.',
                       'Thin dotted flux curves use t=0 on the same mesh to expose interpolation bias.',
                       'Black disks mark nominal isolated-hole radii 2 m, not measured binary horizons.',
                       'Mid-plane values average the z cell centers bracketing zero.',
                       'Cyan circles are nominal mask radii; only blocks with stored zoom data are held.']}
    (out/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

    # Close counterpart of the hydrodynamic Bondi figure in Guo et al. (2025).
    fig=plt.figure(figsize=(15.2,4.4),layout='constrained')
    gs=fig.add_gridspec(1,4,width_ratios=[1,1,.045,1.35],wspace=.08)
    for col,name in enumerate(('single_rotating','binary_rotating')):
        ax=fig.add_subplot(gs[0,col]); d,f=load(fluid_paths(root,name)[-1])
        im=draw_slice(ax,d,f,name,48,rhoinf)
        ax.set_title('Single BH, rotating' if col==0 else 'Binary, rotating; a=32 M')
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[0,2]))
    cb.set_label(r'$\rho/\rho_\infty$')
    ax=fig.add_subplot(gs[0,3])
    for name,label,color,style in zip(CASES,LABELS,COLORS,STYLES):
        ax.semilogx(r,profiles[name]['final']['mdot']/mdotb,color=color,ls=style,lw=1.8,label=label)
        ax.semilogx(r,profiles[name]['initial']['mdot']/mdotb,color=color,ls=':',lw=.7,alpha=.55)
    ax.axhspan(.999,1.001,color='.85',zorder=-3)
    ax.axhline(1,color='black',ls='--',lw=.8)
    for radius in (32,64,128): ax.axvline(radius,color='.8',lw=.8,zorder=-2)
    ax.axvline(rb,color='black',lw=1)
    ax.text(rb*.97,1.0013,r'$r_{\rm B}$',ha='right',fontsize=11)
    ax.set(xlim=(3,240),ylim=(.9975,1.0015),xlabel=r'$r/M$',ylabel=r'$\dot M_{\rm net}/\dot M_{\rm B}$')
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=10,ncol=4,loc='lower center',
               bbox_to_anchor=(.5,-.065),frameon=False)
    fig.text(.5,-.105,'Unrelaxed central Bondi seed; global time = 0.00352 orbit. Dotted: initial flux; gray: analytic rate ±0.1%.',
             ha='center',fontsize=10)
    ax.set_title('Instantaneous COM-sphere mass flux')
    fig.suptitle(r'Bondi-fed controls at $t=4M$ — $M_{\rm tot}=1$, $r_{\rm B}=161M$, $B=0$',fontsize=14)
    save(fig,out,'bondi_paper_comparison')

    # Both physical scales, with one common density color range.
    fig,axs=plt.subplots(2,2,figsize=(10.5,9.1),layout='constrained')
    for col,name in enumerate(('single_rotating','binary_rotating')):
        d,f=load(fluid_paths(root,name)[-1])
        for row,half in enumerate((256,40)):
            im=draw_slice(axs[row,col],d,f,name,half,rhoinf,wide=row==0)
            axs[row,col].set_title(('Single BH' if col==0 else 'Binary BH')+(' — feeding scale' if row==0 else ' — inner scale'))
    cb=fig.colorbar(im,ax=axs,shrink=.88,pad=.02)
    cb.set_label(r'$\rho/\rho_\infty$')
    fig.suptitle('Rotating-frame equatorial density at t=4 M\nWhite dashed: Bondi radius; green dotted: light cylinder',fontsize=13)
    save(fig,out,'bondi_density_scales')

    # Instantaneous radial profiles and a same-mesh evolution residual.
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for name,label,color,style in zip(CASES,LABELS,COLORS,STYLES):
        v=profiles[name]['final']; init=profiles[name]['initial']
        axs[0,0].loglog(r,v['rho']/rhoinf,color=color,ls=style,label=label)
        axs[0,1].loglog(r,v['temperature']/tinf,color=color,ls=style)
        axs[1,0].semilogx(r,v['mdot']/mdotb,color=color,ls=style)
        axs[1,0].semilogx(r,init['mdot']/mdotb,color=color,ls=':',alpha=.5,lw=.7)
        axs[1,1].semilogx(r,100*(v['mdot']/init['mdot']-1),color=color,ls=style)
    axs[0,0].loglog(r,analytic_rho/rhoinf,'k--',lw=1,label='Analytic Michel')
    axs[0,1].loglog(r,analytic_t/tinf,'k--',lw=1)
    axs[1,0].axhline(1,color='k',ls='--',lw=.7)
    axs[1,1].axhspan(-.1,.1,color='.9'); axs[1,1].axhline(0,color='k',lw=.6)
    for ax in axs.flat:
        ax.set_xlim(3,240); ax.set_xlabel(r'$r/M$')
        ax.axvline(rb,color='k',lw=.8,alpha=.7)
        for radius in (32,64,128): ax.axvline(radius,color='.85',lw=.6,zorder=-2)
        ax.grid(alpha=.15,which='major')
    axs[0,0].set_ylabel(r'$\langle\rho\rangle_\Omega/\rho_\infty$')
    axs[0,1].set_ylabel(r'$\langle p/\rho\rangle_\Omega/T_\infty$')
    axs[1,0].set_ylabel(r'$\dot M_{\rm net}/\dot M_{\rm B}$')
    axs[1,1].set_ylabel(r'$100[\dot M(t)/\dot M(0)-1]$ [%]')
    axs[0,0].legend(frameon=False,fontsize=9)
    axs[1,0].set_title('Dotted curves: initial state on the same mesh',fontsize=10)
    axs[1,1].set_title('Evolution relative to identical sampling at t=0',fontsize=10)
    fig.suptitle('Instantaneous radial diagnostics at t=4 M — binary flux shown only for r ≥ 24 M',fontsize=13)
    save(fig,out,'bondi_radial_profiles')

    # Snapshots are stored before the scheduled AMR event. Infer the actual
    # zoom zone from the saved maximum refinement level, not nominal output time.
    fig,axs=plt.subplots(2,5,figsize=(16.4,6.9),layout='constrained')
    selected={}
    zones=(0,2,3,2,0); halfs=(40,160,256,160,40)
    for row,name in enumerate(('single_rotating','binary_rotating')):
        paths=fluid_paths(root,name)
        selected[name]=[]
        for col,(index,zone,half) in enumerate(zip((0,4,5,6,8),zones,halfs)):
            d,f=load(paths[index])
            actual=7-int(d['mb_logical'][:,3].max())
            assert actual==zone
            im=draw_slice(axs[row,col],d,f,name,half,rhoinf,grid=True,
                          mask=32*2**zone if zone else None)
            axs[row,col].set_title(f'Zone {zone}; t={d["time"]:.3f} M',fontsize=11)
            axs[row,col].tick_params(labelsize=9)
            if col==0: axs[row,col].set_ylabel(('Single BH' if row==0 else 'Binary BH')+'\n'+r'$y/M$')
            selected[name].append({'file':paths[index].name,'time':d['time'],'zone':zone,'blocks':d['n_mbs']})
    cb=fig.colorbar(im,ax=axs,shrink=.8,pad=.01); cb.set_label(r'$\rho/\rho_\infty$')
    fig.suptitle('One infrastructure-test return; total time = 0.00352 orbit\nCyan: nominal mask boundary; white grid: actual MeshBlocks; axes change scale',fontsize=13)
    save(fig,out,'bondi_zoom_sequence')
    metadata['zoom_snapshots']=selected
    (out/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Figures saved to',out,flush=True)


if __name__=='__main__':
    main()
