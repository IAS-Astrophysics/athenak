#!/usr/bin/env python3
"""Bounded full-solver ambient and outgoing-packet controls (not production runs)."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from prepare_characteristic_control import prepare
from test_binary_zoom import ROOT,load,final_output,inspect

def run(exe,out,resolutions=(32,),suite='all',include_reference=True,boundaries=('fixed','characteristic'),trace_order=3,modes=('acoustic','alfven')):
    out.mkdir(parents=True,exist_ok=True)
    base=prepare((ROOT/'inputs/dyngr/bondi_hierarchy/binary_magnetized_lowres.athinput').read_text())
    base=base.replace('<problem>','''<problem>
reservoir_boundary = fixed
reservoir_trace_order = 2
reservoir_test_flat_metric = true
reservoir_test_pulse = 0
reservoir_test_pulse_mode = acoustic
reservoir_test_pulse_z = 4
reservoir_test_pulse_width = 1
''')
    # Test configuration is explicit; no missing-key command line additions.
    base=base.replace('refinement = adaptive','refinement = none').replace('excise = true','excise = false')
    (out/'input.athinput').write_text(base)
    cases=[]
    if suite in ('all','ambient'):
        for omega in (0.,2**.5):
          for boundary in boundaries:
            cases.append((f'ambient-{boundary}-{omega:.4f}',16,16,16,1.,omega,boundary,0.,'acoustic'))
    if suite in ('all','pulse'):
        for nz in resolutions:
          for mode in modes:
            if include_reference:cases.append((f'{mode}-reference-{nz}',8,8,3*nz,180.,0.,'fixed',.001,mode))
            for boundary in boundaries:
              cases.append((f'{mode}-{boundary}-{nz}',8,8,nz,180.,0.,boundary,.001,mode))
    rows=[]
    for name,nx,ny,nz,tlim,omega,boundary,amplitude,mode in cases:
        case=out/name;case.mkdir(exist_ok=True)
        flags=['problem/single_black_hole=true',f'mesh/nx1={nx}',f'mesh/nx2={ny}',f'mesh/nx3={nz}',
          f'problem/reservoir_boundary={boundary}',f'problem/reservoir_trace_order={trace_order}',f'problem/frame_omega={omega}',
          f'problem/corotating={str(omega!=0).lower()}',f'problem/reservoir_test_pulse={amplitude}',
          f'problem/reservoir_test_pulse_mode={mode}','problem/reservoir_pressure_perturbation=0',
          'coord/require_resolved_horizon=false','time/nlim=50000',f'time/tlim={tlim}',
          f'output1/dt={tlim}',f'output2/dt={tlim}',f'output3/dt={tlim}','output4/dt=0']
        for d in range(1,4):
            half=24 if '-reference-' in name and d==3 else 8
            flags.extend([f'mesh/x{d}min={-half}',f'mesh/x{d}max={half}'])
        if amplitude:
          for d in (1,2):flags.extend([f'mesh/ix{d}_bc=periodic',f'mesh/ox{d}_bc=periodic'])
        flags.append('mhd/dyn_flux_team_size=0') # Local serial build only.
        start=time.monotonic()
        with (case/'run.log').open('w') as log:
            proc=subprocess.run([str(exe.resolve()),'-i',str((out/'input.athinput').resolve()),'-d',str(case.resolve()),'-t','00:05:00']+flags,stdout=log,stderr=subprocess.STDOUT,timeout=330)
        if proc.returncode:raise RuntimeError(f'{name}: '+(case/'run.log').read_text()[-1800:])
        checks=inspect(case);data,f=load(final_output(case,'mhd_w_bcc'))
        assert data['time']==tlim,(name,data['time'])
        # Active-cell indices in binary output including physical ghosts.
        geom=data['mb_geometry'];indices=data['mb_index'];valid=[];means=[];transverse=[];profile=[]
        pressure=1*(2/500)/((5/3)*(1-(2/500)/(2/3)))
        B=np.sqrt(2*pressure);h=1+(5/3)/(2/3)*pressure;inertia=np.sqrt(h+B*B)
        for mb in range(data['n_mbs']):
            mask=np.ones(f['dens'][mb].shape,dtype=bool)
            for dim in range(3):
                n=data[f'nx{dim+1}_out_mb'];dx=(geom[mb,2*dim+1]-geom[mb,2*dim])/data[f'nx{dim+1}_mb']
                xyz=geom[mb,2*dim]+(indices[mb,2*dim]+np.arange(n)+.5)*dx
                shape=[1,1,1];shape[2-dim]=n
                mask &= ((xyz>=geom[mb,2*dim])&(xyz<=geom[mb,2*dim+1])).reshape(shape)
            # Plane packet must stay independent of transverse x,y. This
            # detects stale-primitive writes into exchanged periodic ghosts.
            active=f['dens'][mb][mask].reshape(data['nx3_mb'],data['nx2_mb'],data['nx1_mb'])
            transverse.append(float(np.ptp(active,axis=(1,2)).max()))
            rho=f['dens'][mb][mask];p=f['press'][mb][mask]
            velocity=np.array([f['vel'+k][mb][mask] for k in 'xyz']);velocity/=np.sqrt(1+np.sum(velocity**2,axis=0))
            inward=(rho-1-velocity[2]/np.sqrt(2/500))/2 if mode=='acoustic' else (velocity[0]+f['bcc1'][mb][mask]/inertia)/2
            means.append([float(abs(rho-1).max()),float(abs(p/pressure-1).max()),float(abs(inward).max())])
            inward_rows=inward.reshape(data['nx3_mb'],data['nx2_mb'],data['nx1_mb']).mean(axis=(1,2))
            zc=geom[mb,4]+(np.arange(data['nx3_mb'])+.5)*(geom[mb,5]-geom[mb,4])/data['nx3_mb']
            profile.extend(zip(zc.tolist(),inward_rows.tolist()))

        m=np.max(means,axis=0)
        if amplitude:assert max(transverse)<2e-6,(name,'broken transverse periodicity',max(transverse))
        row=dict(profile=sorted(profile),name=name,transverse_density_spread=max(transverse),seconds=time.monotonic()-start,time=tlim,cycle=data['cycle'],density_max_deviation=m[0],pressure_max_relative_deviation=m[1],max_inward_mode_amplitude=m[2],checks=checks)
        rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='profile'}),flush=True)
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resolutions',type=int,nargs='+',default=[32]);p.add_argument('--skip-references',action='store_true');p.add_argument('--boundaries',nargs='+',choices=['fixed','characteristic'],default=['fixed','characteristic']);p.add_argument('--modes',nargs='+',choices=['acoustic','alfven'],default=['acoustic','alfven']);p.add_argument('--trace-order',type=int,choices=(1,2,3),default=3);p.add_argument('--suite',choices=['all','ambient','pulse'],default='all');a=p.parse_args();run(a.exe,a.out,a.resolutions,a.suite,not a.skip_references,a.boundaries,a.trace_order,a.modes)
