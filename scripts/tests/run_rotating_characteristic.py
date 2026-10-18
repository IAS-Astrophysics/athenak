#!/usr/bin/env python3
"""Local bounded 3D rotating tests. No queue operations or continuations."""
import argparse,json,subprocess,time
from pathlib import Path
from characteristic_rotating_cases import prepared_input,flags
from test_binary_zoom import ROOT,load,final_output,inspect
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=16);p.add_argument('--block-z',type=int,default=8);p.add_argument('--family',choices=['acoustic3d','entropy'],required=True);p.add_argument('--frame',choices=['inertial','rotating','fastrotation'],required=True);p.add_argument('--choices',nargs='+',default=['reference','fixed','characteristic']);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);base=prepared_input((ROOT/'inputs/dyngr/bondi_hierarchy/binary_magnetized_lowres.athinput').read_text());(a.out/'input.athinput').write_text(base);rows=[]
for choice in a.choices:
 name=f'{a.family}-{a.frame}-{choice}-{a.n}';case=a.out/name;case.mkdir(exist_ok=True)
 args=flags(a.family,a.frame,choice,a.n,a.block_z)+['mhd/dyn_flux_team_size=0']
 (case/'overrides.txt').write_text('\n'.join(args)+'\n');start=time.monotonic()
 with (case/'run.log').open('w') as log:
  r=subprocess.run([str(a.exe.resolve()),'-i',str((a.out/'input.athinput').resolve()),'-d',str(case.resolve()),'-t','00:09:00',*args],stdout=log,stderr=subprocess.STDOUT,timeout=570)
 if r.returncode:raise RuntimeError((case/'run.log').read_text()[-2000:])
 checks=inspect(case);d,f=load(final_output(case,'mhd_w_bcc'));target=.4 if a.family=='entropy' or a.frame=='fastrotation' else 180
 assert d['time']==target
 rows.append(dict(name=name,seconds=time.monotonic()-start,cycle=d['cycle'],time=d['time'],checks=checks))
 (a.out/'local-results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)
