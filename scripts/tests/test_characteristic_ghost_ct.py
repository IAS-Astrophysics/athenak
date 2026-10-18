#!/usr/bin/env python3
"""Bounded full-solver gates for characteristic ghosts and native CT.

No queue operations. Exercises post-C2P ghosts including edges/corners, both
rotation senses, transverse field perturbations, and forced native FOFC.
"""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from characteristic_rotating_cases import prepared_input,flags
from test_binary_zoom import ROOT,load,inspect,final_output
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
base=prepared_input((ROOT/'inputs/dyngr/bondi_hierarchy/binary_magnetized_lowres.athinput').read_text());inp=a.out/'input.athinput';inp.write_text(base);rows=[]
for name,omega,mode,seed,forced in [
 ('ambient-inertial',0,'acoustic3d',0,False),('ambient-rotating',32**-1.5,'acoustic3d',0,False),
 ('ambient-fast',2**.5,'acoustic3d',0,False),('ambient-reverse',-2**.5,'acoustic3d',0,False),
 ('compact-fast',2**.5,'acoustic3d',.01,False),('compact-reverse',-2**.5,'acoustic3d',.01,False),
 ('alfven',0,'alfven',.01,False),('fofc',0,'acoustic3d',.01,True)]:
 case=a.out/name;case.mkdir(exist_ok=True)
 args=flags('acoustic3d','fastrotation','characteristic',16,8)+[
  'problem/reservoir_ghost_ct=true',f'problem/frame_omega={omega}',f'problem/corotating={str(omega!=0).lower()}',
  f'problem/reservoir_test_pulse_mode={mode}',f'problem/reservoir_test_pulse={seed}',
  'time/tlim=0.4','time/nlim=10000','mhd/dyn_flux_team_size=0',
  'output1/ghost_zones=true','output2/ghost_zones=true','output3/ghost_zones=true',
  'output1/dt=0.4','output2/dt=0.4','output3/dt=0.4',
  f'mhd/enforce_maximum={str(forced).lower()}',f'mhd/dmp_M={.5 if forced else 1.2}']
 with (case/'run.log').open('w') as log:
  t=time.monotonic();subprocess.run([str(a.exe.resolve()),'-i',str(inp.resolve()),'-d',str(case.resolve()),'-t','00:02:00',*args],stdout=log,stderr=subprocess.STDOUT,timeout=125,check=True)
 check=inspect(case);d,f=load(final_output(case,'mhd_w_bcc'));assert d['time']==.4
 # All exported ghost cells must also be finite/positive and discretely solenoidal.
 maxdiv=0
 for path in (case/'bin').glob('*.mhd_divb.*.bin'):
  dd,ff=load(path);assert dd['nx1_out_mb']>dd['nx1_mb'];maxdiv=max(maxdiv,float(abs(ff['divb']).max()))
 assert maxdiv<1e-10
 if seed==0:
  assert abs(f['dens']-1).max()<2e-6 and np.ptp(f['press'])<1e-9
  speed=np.sqrt(sum(v.astype(float)**2 for k,v in f.items() if k in ('velx','vely','velz')))
  assert speed.max()<1e-10,(name,speed.max())
 rows.append(dict(name=name,status='passed',seconds=time.monotonic()-t,all_ghost_max_abs_divB=maxdiv,checks=check))
 (a.out/'results.json').write_text(json.dumps(dict(status='passed',cases=rows),indent=2)+'\n');print(name,'passed',flush=True)
