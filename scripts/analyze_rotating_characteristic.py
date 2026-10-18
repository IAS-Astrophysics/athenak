#!/usr/bin/env python3
"""Matched-grid 3D rotating-boundary comparisons, with exact entropy advection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent/'tests'))
from test_binary_zoom import load,final_output,inspect
from analyze_grmhd_characteristic import audit,params

def cube(case,initial=False):
    files=sorted((case/'bin').glob('*.mhd_w_bcc.*.bin'))
    d,f=load(files[0] if initial else files[-1]);cfg=params(d['header'])
    n=int(case.name.split('-')[-1]);dx=16/n;out={k:np.empty((n,n,n)) for k in f};seen=np.zeros((n,n,n),int)
    for mb,box in enumerate(d['mb_geometry']):
        xyz=[box[2*i]+(np.arange(d[f'nx{i+1}_mb'])+.5)*(box[2*i+1]-box[2*i])/d[f'nx{i+1}_mb'] for i in range(3)]
        kk,jj,ii=np.meshgrid(*[(q+8)/dx-.5 for q in xyz[::-1]],indexing='ij');ii=ii.round().astype(int);jj=jj.round().astype(int);kk=kk.round().astype(int)
        valid=(ii>=0)&(ii<n)&(jj>=0)&(jj<n)&(kk>=0)&(kk<n)
        seen[kk[valid],jj[valid],ii[valid]]+=1
        for key in out:out[key][kk[valid],jj[valid],ii[valid]]=f[key][mb][valid]
    assert np.all(seen==1),(case,'topology or common-grid mismatch')
    u=np.array([out['vel'+c] for c in 'xyz']);v=u/np.sqrt(1+np.sum(u*u,axis=0));theta=float(cfg['problem/frame_omega'])*d['time']
    inertial=v.copy();inertial[0]=np.cos(theta)*v[0]-np.sin(theta)*v[1];inertial[1]=np.sin(theta)*v[0]+np.cos(theta)*v[1]
    rho=float(cfg['problem/rho_inf']);gamma=float(cfg['mhd/gamma']);cs2=2/float(cfg['problem/r_bondi']);p=rho*cs2/(gamma*(1-cs2/(gamma-1)))
    proxy=.5*((out['press']/p-1)/gamma-v[2]/np.sqrt(cs2))
    return d,cfg,out,inertial,proxy

def analyze(runs,out,local=False):
    out.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    cases=sorted(p for p in runs.iterdir() if p.is_dir() and p.name.startswith(('acoustic3d-','entropy-')))
    for case in cases:
        family,frame,choice,ns=case.name.split('-');n=int(ns);target=.4 if family=='entropy' or frame=='fastrotation' else 180
        if local:audits.append(dict(name=case.name,checks=inspect(case)))
        else:audits.append(audit(case,target=target,plane=False))
        if choice=='reference':continue
        ref=runs/f'{family}-{frame}-reference-{n}'
        if not ref.exists():continue
        d,cfg,f,v,proxy=cube(case);dr,cr,fr,vr,pr=cube(ref)
        assert d['time']==dr['time']==target
        assert d['cycle']==dr['cycle'],(case,'Reference timestep mismatch',d['cycle'],dr['cycle'])
        _,_,fi,_,_=cube(case,True);_,_,fri,_,_=cube(ref,True)
        initial_errors={k:float(abs(fi[k]-fri[k]).max()) for k in fi}
        assert all(initial_errors[k]<=4e-7*max(float(abs(fi[k]).max()),1e-30) for k in fi),'Initial common-grid states differ beyond output precision'
        seed=float(cfg['problem/reservoir_test_pulse']);rho=float(cfg['problem/rho_inf']);gamma=float(cfg['mhd/gamma']);cs2=2/float(cfg['problem/r_bondi']);p=rho*cs2/(gamma*(1-cs2/(gamma-1)))
        diff=(proxy-pr)/seed;dd=(f['dens']-fr['dens'])/(rho*seed)
        row=dict(name=case.name,family=family,frame=frame,choice=choice,n=n,time=target,trace_order=int(cfg['problem/reservoir_trace_order']),initial_common_state_equal=all(v==0 for v in initial_errors.values()),initial_states_match_output_precision=True,initial_max_differences=initial_errors,
          incoming_proxy_max=float(abs(diff).max()),incoming_proxy_rms=float(np.sqrt(np.mean(diff*diff))),
          density_difference_max=float(abs(dd).max()),density_difference_rms=float(np.sqrt(np.mean(dd*dd))),
          pressure_difference_max=float(abs(f['press']-fr['press']).max()/(gamma*p*seed)),
          inertial_velocity_difference_max=float(np.sqrt(np.sum((v-vr)**2,axis=0)).max()),
          z_profile=diff.mean(axis=(1,2)).tolist())
        if family=='entropy':
            omega=float(cfg['problem/frame_omega']);theta=omega*target;cx=6*(np.cos(theta)+np.sin(theta));cy=6*(np.cos(theta)-np.sin(theta))
            centers=-8+(np.arange(n)+.5)*16/n;z,y,x=np.meshgrid(centers,centers,centers,indexing='ij');r2=(x-cx)**2+(y-cy)**2;z2=z*z;shape=np.zeros_like(r2);mask=(r2<1)&(z2<1);shape[mask]=np.exp(2-1/(1-r2[mask])-1/(1-z2[mask]));exact=rho*(1+seed*shape)
            row.update(exact_density_max=float(abs(f['dens']-exact).max()/(rho*seed)),exact_density_rms=float(np.sqrt(np.mean(((f['dens']-exact)/(rho*seed))**2))),
                       inertial_speed_max=float(np.sqrt(np.sum(v*v,axis=0)).max()))
        rows.append(row)
    (out/'audits.json').write_text(json.dumps(audits,indent=2)+'\n');(out/'comparisons.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for row in rows:
        color={'fixed':'tab:blue','characteristic':'tab:orange','firstorder':'tab:green'}[row['choice']]
        if row['family']=='acoustic3d':
            z=-8+(np.arange(row['n'])+.5)*16/row['n'];axs[1 if row['frame']=='fastrotation' else 0].plot(z,row['z_profile'],color=color,ls='--' if row['n']==16 else '-',label=f"{row['frame']} {row['choice']}, n={row['n']}")
        else:axs[2].scatter(row['n'],row['density_difference_rms'],color=color,label=row['choice'])
    axs[0].set(xlabel='z / M',ylabel='Plane-mean inward acoustic proxy difference / seed',title='Compact packet minus larger-domain control')
    axs[1].set(xlabel='z / M',ylabel='Plane-mean inward acoustic proxy difference / seed',title='High shift acoustic, t=0.4 M')
    axs[2].set(xlabel='Cells across short box',ylabel='Density RMS difference / seed',title='High shift entropy, t=0.4 M')
    for ax in axs:
        ax.grid(alpha=.2)
        if ax.lines or ax.collections:ax.legend(fontsize=7)
    fig.savefig(out/'rotating_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps([{k:v for k,v in row.items() if k!='z_profile'} for row in rows],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--local',action='store_true');a=p.parse_args();analyze(a.runs,a.out,a.local)
