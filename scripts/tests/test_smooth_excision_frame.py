#!/usr/bin/env python3
"""Bounded inertial/rotating single and binary MHD controls with smooth drain."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,subprocess,sys,re
from pathlib import Path
import numpy as np
from test_binary_zoom import ROOT,load,final_output
sys.path.insert(0,str(ROOT/'scripts'))
from plot_bondi_validation import AMRSampler
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reuse',action='store_true');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
base=(ROOT/'inputs/dyngr/bondi_matched_zoom.athinput').read_text()
results={};all_cases={}
for nx in (32,64):
 for single in (True,False):
  kind='single' if single else 'binary';omega=.2 if single else 4**-1.5;states={}
  for rotating in (False,True):
   name=f'{kind}-{nx}-'+('rotating' if rotating else 'inertial');case=a.out/name;case.mkdir(exist_ok=True)
   if not(a.reuse and list((case/'bin').glob('*.mhd_w_bcc.*.bin'))):
    flags=['mesh_refinement/refinement=none',f'mesh_refinement/max_nmb_per_rank={(nx//8)**3}','problem/initial_condition=bondi_reservoir','problem/rho_inf=1','problem/r_bondi=500','problem/beta_inf=1','problem/b_inf=0','problem/reservoir_pressure_perturbation=0','problem/sep=4',f'problem/single_black_hole={str(single).lower()}',f'problem/corotating={str(rotating).lower()}',f'problem/frame_omega={omega if rotating else 0}',f'adm/dynamic={str(not rotating).lower()}', 'coord/require_resolved_horizon=false','coord/smooth_excision=true','coord/smooth_excision_puncture_width_fraction=1','coord/smooth_excision_puncture_weight_exponent=1','coord/smooth_excision_inflow=false','coord/smooth_excision_b_damping=false','time/tlim=.5','time/nlim=-1','output1/dt=.5','output2/dt=.5','output3/dt=.5','output4/dt=0']
    for i in (1,2,3):flags += [f'mesh/nx{i}={nx}',f'mesh/x{i}min=-8',f'mesh/x{i}max=8']
    # Athena command-line overrides require the key to exist in the input.
    text=base
    for flag in flags:
     key,value=flag.split('=',1);block,key=key.split('/',1)
     pattern=r'(<'+re.escape(block)+r'>\n)(.*?)(?=\n<|\Z)'
     def update(m):
      body=m.group(2);line=r'(?m)^'+re.escape(key)+r'\s*=.*$'
      body=re.sub(line,key+' = '+value,body) if re.search(line,body) else body+'\n'+key+' = '+value+'\n'
      return m.group(1)+body
     text,n=re.subn(pattern,update,text,flags=re.S);assert n==1,block
    case_input=case/'input.athinput';case_input.write_text(text)
    cmd=[sys.executable,str(ROOT/'scripts/tests/run_bounded.py'),'--seconds','500','--memory-gb','8','--report',str(case/'resources.json'),'--',str(a.exe.resolve()),'-i',str(case_input.resolve()),'-d',str(case.resolve())]
    with (case/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   d,f=load(final_output(case,'mhd_w_bcc'))
   assert d['time']==.5 and all(np.isfinite(v).all() for v in f.values()) and f['dens'].min()>0 and f['press'].min()>0
   _,div=load(final_output(case,'mhd_divb'));divmax=max(float(abs(v).max()) for v in div.values());assert divmax<1e-8
   states[rotating]=(d,f);all_cases[name]={'time':d['time'],'cycle':d['cycle'],'max_abs_divb':divmax,'resources':json.loads((case/'resources.json').read_text())}
   print('Completed',name,flush=True)
  q=np.linspace(-5.8,5.8,37);z,y,x=np.meshgrid(q,q,q,indexing='ij');pts=np.column_stack([x.ravel(),y.ravel(),z.ravel()]);rad=np.linalg.norm(pts,axis=1)
  centers=[0.] if single else [-2.,2.];dist=np.min([np.linalg.norm(pts-np.array([c,0,0]),axis=1) for c in centers],axis=0)
  theta=omega*.5;c=np.cos(theta);s=np.sin(theta);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
  sampled=[]
  for rot in (False,True):
   d,f=states[rot];sam=AMRSampler(d,pts if rot else np.einsum('ij,kj->ik',pts,R))
   scalar={k:sam.sample(f[k]) for k in ('dens','press')}
   U=np.column_stack([sam.sample(f['vel'+k]) for k in 'xyz']);B=np.column_stack([sam.sample(f['bcc'+k]) for k in '123'])
   if not rot:U=np.einsum('ij,jk->ik',U,R);B=np.einsum('ij,jk->ik',B,R)
   sampled.append((scalar,U,B))
  report={}
  for group,mask in [('outside_drain',(rad<6)&(dist>1.05)),('far_exterior',(rad<6)&(dist>3)),('drain',(dist>.2)&(dist<.75))]:
   first,U0,B0=sampled[0];second,U1,B1=sampled[1];assert mask.any()
   report[group]={k:float(np.mean(abs(second[k][mask]-first[k][mask]))/np.mean(abs(first[k][mask]))) for k in ('dens','press')}
   report[group]['u_relative_L1']=float(np.mean(np.linalg.norm((U1-U0)[mask],axis=1))/max(np.mean(np.linalg.norm(U0[mask],axis=1)),1e-20))
   report[group]['B_relative_L1']=float(np.mean(np.linalg.norm((B1-B0)[mask],axis=1))/np.mean(np.linalg.norm(B0[mask],axis=1)))
  results[f'{kind}-{nx}']=report;print(kind,nx,report,flush=True)
# Only claim convergence in the explicitly separated far exterior (r_A>3 M).
# The drain and immediate horizon neighbourhood retain large timestep-dependent
# pressure differences; they are measured above and are not hidden by this gate.
for kind in ('single','binary'):
 fine=results[kind+'-64']['far_exterior'];coarse=results[kind+'-32']['far_exterior']
 for key,limit in [('dens',2e-4),('press',5e-3),('u_relative_L1',2e-3),('B_relative_L1',3e-4)]:
  assert fine[key]<limit and fine[key]<0.5*coarse[key],(kind,key,coarse[key],fine[key])
(a.out/'validation.json').write_text(json.dumps({'cases':all_cases,'frame_comparisons':results,'interpretation':'Measured rotating/inertial differences; resolution trends must be assessed before declaring a pass.'},indent=2)+'\n')
