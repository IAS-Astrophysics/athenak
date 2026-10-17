#!/usr/bin/env python3
"""Validate inertial-rest magnetic reservoir and external ghost cells in GR frames."""
import argparse,json,subprocess,resource
from pathlib import Path
import numpy as np
from test_binary_zoom import ROOT,load,coords,final_output,inspect

p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--resolutions',type=int,nargs=2,default=[16,32])
p.add_argument('--tlim',type=float,default=1.)
p.add_argument('--case-seconds',type=float,default=180.)
p.add_argument('--reuse-completed',action='store_true',help='Audit matching saved datasets; does not validate a different executable')
a=p.parse_args()
assert 0<a.resolutions[0]<a.resolutions[1] and all(n%8==0 for n in a.resolutions)
assert np.isfinite(a.tlim) and a.tlim>0 and a.case_seconds>0
a.out.mkdir(parents=True,exist_ok=True);results={};comparisons={}
base=(ROOT/'inputs/dyngr/bondi_matched_zoom.athinput').read_text()
base=base.replace('<problem>','<problem>\nrho_inf = 1\nr_bondi = 500\nbeta_inf = 1\nreservoir_pressure_perturbation = 0')
for i in [1,2]:base=base.replace(f'<output{i}>',f'<output{i}>\nghost_zones = true')
input_path=a.out/'test.athinput';input_path.write_text(base)
for nx in a.resolutions:
    states={}
    for name,omega,single in [('inertial',0.,True),('rotating',.2,True),('binary',.125,False)]:
        case=a.out/f'{name}{nx}';case.mkdir(exist_ok=True)
        flags=['mesh_refinement/refinement=none','problem/initial_condition=bondi_reservoir',
            'problem/rho_inf=1','problem/r_bondi=500','problem/beta_inf=1',
            f'problem/reservoir_pressure_perturbation={0 if single else .02}','problem/sep=4',
            f'problem/single_black_hole={str(single).lower()}',
            f'problem/corotating={str(omega!=0).lower()}',f'problem/frame_omega={omega}',
            'coord/require_resolved_horizon=false',f'time/tlim={a.tlim}','time/nlim=2000',
            f'output1/dt={a.tlim}',f'output2/dt={a.tlim}',f'output3/dt={a.tlim}','output4/dt=0',
            'output1/ghost_zones=true','output2/ghost_zones=true']
        for i in [1,2,3]:flags += [f'mesh/nx{i}={nx}',f'mesh/x{i}min=-8',f'mesh/x{i}max=8']
        reused=False
        if a.reuse_completed and (case/'run.log').exists() and 'Terminating on time limit' in (case/'run.log').read_text():
            prior,_=load(final_output(case,'mhd_w_bcc'))
            params={};block=''
            for line in prior['header']:
                line=line.split('#',1)[0].strip()
                if line.startswith('<') and line.endswith('>'):block=line[1:-1]
                elif '=' in line:
                    key,value=line.split('=',1);params[block+'/'+key.strip()]=value.strip()
            reused=prior['time']==a.tlim
            for flag in flags:
                key,value=flag.split('=',1);actual=params.get(key,'')
                try:matches=abs(float(actual)-float(value))<=1e-12*max(1,abs(float(value)))
                except ValueError:matches=actual.lower()==value.lower()
                reused=reused and matches
        if not reused:
            with (case/'run.log').open('w') as log:
                proc=subprocess.run([str(a.exe.resolve()),'-i',str(input_path.resolve()),'-d',str(case.resolve())]+flags,stdout=log,stderr=subprocess.STDOUT,timeout=a.case_seconds)
            assert proc.returncode==0,(case,(case/'run.log').read_text()[-2500:])
        d,f=load(final_output(case,'mhd_w_bcc'));_,adm=load(final_output(case,'adm'))
        assert d['time']==a.tlim, 'Frame comparison did not finish its target evolution'
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
        desired=beta.copy();desired[0]+=omega*y;desired[1]-=omega*x
        desired/=adm['adm_alpha']
        desired_norm=np.einsum('ab...,a...,b...->...',g,desired,desired)
        not_excised=(radius>.8) if single else (((x-2)**2+y*y+z*z>.8**2)&((x+2)**2+y*y+z*z>.8**2))
        # The combined tight-binary metric's static limit is not the union
        # of isolated-hole spheres. Check inertial rest where it is timelike.
        block='';config={}
        for line in d['header']:
            line=line.split('#',1)[0].strip()
            if line.startswith('<') and line.endswith('>'):block=line[1:-1]
            elif '=' in line:
                key,value=line.split('=',1);config[block+'/'+key.strip()]=value.strip()
        gamma_max=float(config['mhd/gamma_max'])
        representable_v2=1-1/gamma_max**2
        safe=not_excised&(desired_norm<representable_v2-2e-6)
        ceiling=not_excised&(desired_norm>representable_v2+2e-6)&(desired_norm<1-2e-6)
        clipped=not_excised&(desired_norm>1+2e-6)
        initial_u=np.stack([initial['vel'+k] for k in 'xyz'])
        initial_u2=np.einsum('ab...,a...,b...->...',g,initial_u,initial_u)
        if clipped.any():assert np.allclose((initial_u2/(1+initial_u2))[clipped],.9,atol=2e-6,rtol=0)

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
        record=inspect(case);record.update(reused_completed_case=reused,checked_representable_rest_cells=int(safe.sum()),limited_spacelike_cells=int(clipped.sum()),lorentz_ceiling_cells=int(ceiling.sum()),gamma_max=gamma_max,initial_inertial_velocity_error=initial_error,external_ghost_inertial_velocity_error=ghost_error)
        results[f'{name}{nx}']=record
        if single:states[name]=f
        print(name,nx,'initial/ghost velocity errors',initial_error,ghost_error,flush=True)
    mask=(radius>3)&(radius<7)
    comparisons[str(nx)]={key:float(np.mean(abs(states['rotating'][key][mask]-states['inertial'][key][mask]))/np.mean(abs(states['inertial'][key][mask]))) for key in ['dens','press','bcc3']}
    # The default coarse grid is deliberately underresolved; require smaller
    # frame differences and a tighter bound on the finer requested grid.
    assert max(comparisons[str(nx)].values())<(.10 if nx==a.resolutions[0] else .05),comparisons
for key in ['dens','press','bcc3']:assert comparisons[str(a.resolutions[1])][key]<comparisons[str(a.resolutions[0])][key],comparisons
# Negative control: a boundary placed inside the single-hole static limit
# must fail explicitly rather than silently limit the imposed reservoir.
invalid=a.out/'invalid_boundary';invalid.mkdir(exist_ok=True)
invalid_flags=['mesh_refinement/refinement=none','problem/initial_condition=bondi_reservoir',
 'problem/single_black_hole=true','problem/corotating=false','problem/frame_omega=0',
 'coord/require_resolved_horizon=false','time/nlim=1']
for i in [1,2,3]:invalid_flags += [f'mesh/nx{i}=8',f'mesh/x{i}min=-0.5',f'mesh/x{i}max=0.5']
def no_core(): resource.setrlimit(resource.RLIMIT_CORE,(0,0))
with (invalid/'run.log').open('w') as log:
 proc=subprocess.run([str(a.exe.resolve()),'-i',str(input_path.resolve()),'-d',str(invalid.resolve())]+invalid_flags,
  stdout=log,stderr=subprocess.STDOUT,timeout=30,preexec_fn=no_core)
assert proc.returncode!=0 and 'Bondi boundary is not timelike' in (invalid/'run.log').read_text()
results['invalid_boundary']={'returncode':proc.returncode,'expected_rejection':'Bondi boundary is not timelike'}
(a.out/'validation.json').write_text(json.dumps(dict(tlim=a.tlim,resolutions=a.resolutions,cases=results,frame_comparisons=comparisons),indent=2)+'\n')
print('Magnetic inertial-rest and boundary checks passed',comparisons,flush=True)
