#!/usr/bin/env python3
"""Validate inertial-rest magnetic reservoir and external ghost cells in GR frames."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from test_binary_zoom import ROOT,load,coords,final_output,inspect

p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);results={};comparisons={}
base=(ROOT/'inputs/dyngr/bondi_matched_zoom.athinput').read_text()
base=base.replace('<problem>','<problem>\nrho_inf = 1\nr_bondi = 500\nbeta_inf = 1\nreservoir_pressure_perturbation = 0')
for i in [1,2]:base=base.replace(f'<output{i}>',f'<output{i}>\nghost_zones = true')
input_path=a.out/'test.athinput';input_path.write_text(base)
for nx in [16,32]:
    states={}
    for name,omega,single in [('inertial',0.,True),('rotating',.2,True),('binary',.125,False)]:
        case=a.out/f'{name}{nx}';case.mkdir(exist_ok=True)
        flags=['mesh_refinement/refinement=none','problem/initial_condition=bondi_reservoir',
            'problem/rho_inf=1','problem/r_bondi=500','problem/beta_inf=1',
            f'problem/reservoir_pressure_perturbation={0 if single else .02}','problem/sep=4',
            f'problem/single_black_hole={str(single).lower()}',
            f'problem/corotating={str(omega!=0).lower()}',f'problem/frame_omega={omega}',
            'coord/require_resolved_horizon=false','time/tlim=1','time/nlim=2000',
            'output1/dt=1','output2/dt=1','output3/dt=1','output4/dt=0',
            'output1/ghost_zones=true','output2/ghost_zones=true']
        for i in [1,2,3]:flags += [f'mesh/nx{i}={nx}',f'mesh/x{i}min=-8',f'mesh/x{i}max=8']
        with (case/'run.log').open('w') as log:
            proc=subprocess.run([str(a.exe.resolve()),'-i',str(input_path.resolve()),'-d',str(case.resolve())]+flags,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert proc.returncode==0,(case,(case/'run.log').read_text()[-2500:])
        d,f=load(final_output(case,'mhd_w_bcc'));_,adm=load(final_output(case,'adm'))
        assert d['time']==1.0, 'Frame comparison did not finish its target evolution'
        _,initial=load(sorted((case/'bin').glob('*.mhd_w_bcc.*.bin'))[0])
        coordinates=[]
        for dim in range(3):
            width=d['mb_geometry'][:,2*dim+1]-d['mb_geometry'][:,2*dim]
            loc=d['mb_geometry'][:,2*dim,None]+(d['mb_index'][:,2*dim,None]+np.arange(d[f'nx{dim+1}_out_mb'])[None,:]+.5)*width[:,None]/d[f'nx{dim+1}_mb']
            coordinates.append(loc[:,None,None,:] if dim==0 else loc[:,None,:,None] if dim==1 else loc[:,:,None,None])
        x,y,z=np.broadcast_arrays(*coordinates);radius=np.sqrt(x*x+y*y+z*z)
        external=(abs(x)>8)|(abs(y)>8)|(abs(z)>8)
        assert external.any(), 'Ghost geometry was not included'
        g=np.array([[adm['adm_gxx'],adm['adm_gxy'],adm['adm_gxz']],
                    [adm['adm_gxy'],adm['adm_gyy'],adm['adm_gyz']],
                    [adm['adm_gxz'],adm['adm_gyz'],adm['adm_gzz']]])
        beta=np.stack([adm['adm_beta'+k] for k in 'xyz'])
        def inertial_velocity(fields):
            u=np.stack([fields['vel'+k] for k in 'xyz']);W=np.sqrt(1+np.einsum('ab...,a...,b...->...',g,u,u))
            v=adm['adm_alpha']*u/W-beta
            v[0]-=omega*y;v[1]+=omega*x
            return v
        safe=(radius>3) if single else ((np.sqrt((x-2)**2+y*y+z*z)>2)&(np.sqrt((x+2)**2+y*y+z*z)>2))
        initial_error=float(abs(inertial_velocity(initial)[:,safe]).max())
        ghost_error=float(abs(inertial_velocity(f)[:,external]).max())
        assert initial_error<2e-6 and ghost_error<2e-6,(initial_error,ghost_error)
        cs2=2/500;pressure=cs2/((5/3)*(1-cs2/(2/3)))
        assert np.allclose(f['dens'][external],1,rtol=2e-6)
        assert np.allclose(f['press'][external],pressure,rtol=2e-6)
        assert np.allclose(initial['press'][external],pressure,rtol=2e-6)
        assert np.allclose(initial['bcc3'],np.sqrt(2*pressure),rtol=2e-6)
        assert np.max(abs(initial['bcc1']))==0 and np.max(abs(initial['bcc2']))==0
        # Skip the first exterior cell, whose cell-centered normal field
        # includes the CT-evolved physical boundary face. Deeper ghosts
        # must retain all three components of the prescribed reservoir.
        deep_external=(abs(x)>8+16/nx)|(abs(y)>8+16/nx)|(abs(z)>8+16/nx)
        assert np.allclose(f['bcc3'][deep_external],np.sqrt(2*pressure),rtol=2e-6)
        assert np.max(abs(f['bcc1'][deep_external]))<1e-12
        assert np.max(abs(f['bcc2'][deep_external]))<1e-12
        record=inspect(case);record.update(initial_inertial_velocity_error=initial_error,external_ghost_inertial_velocity_error=ghost_error)
        results[f'{name}{nx}']=record
        if single:states[name]=f
        print(name,nx,'initial/ghost velocity errors',initial_error,ghost_error,flush=True)
    mask=(radius>3)&(radius<7)
    comparisons[str(nx)]={key:float(np.mean(abs(states['rotating'][key][mask]-states['inertial'][key][mask]))/np.mean(abs(states['inertial'][key][mask]))) for key in ['dens','press','bcc3']}
    # The intentionally underresolved nx=16 grid has only two cells per
    # single-hole horizon; require improvement and tighter errors at nx=32.
    assert max(comparisons[str(nx)].values())<(.10 if nx==16 else .05),comparisons
for key in ['dens','press','bcc3']:assert comparisons['32'][key]<comparisons['16'][key],comparisons
(a.out/'validation.json').write_text(json.dumps(dict(cases=results,frame_comparisons=comparisons),indent=2)+'\n')
print('Magnetic inertial-rest and boundary checks passed',comparisons,flush=True)
