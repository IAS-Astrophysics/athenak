#!/usr/bin/env python3
"""Bounded initial-cavity and short-evolution checks; uses the Bondi helpers."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from test_binary_zoom import ROOT, load, coords, final_output
p=argparse.ArgumentParser()
p.add_argument('--exe',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
fields={}; report={}
for name,radius in [('original',0),('cavity',32)]:
 case=out/name; case.mkdir(exist_ok=True)
 flags=['mesh_refinement/refinement=none','time/tlim=0.2','time/nlim=100',
        'coord/require_resolved_horizon=false',f'problem/bondi_cavity_radius={radius}',
        'problem/bondi_cavity_fraction=0','output1/dt=0.1','output2/dt=0.2',
        'output3/dt=0.2','time/ndiag=100']
 for d in (1,2,3):
  flags += [f'mesh/nx{d}=32',f'mesh/x{d}min=-32',f'mesh/x{d}max=32']
 with (case/'run.log').open('w') as log:
  subprocess.run([str(a.exe.resolve()),'-i',str(ROOT/'inputs/dyngr/bondi_matched_zoom.athinput'),'-d',str(case)]+flags,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 d,f=load(sorted((case/'bin').glob('*.mhd_w_bcc.*.bin'))[0]); fields[name]=f
 x,y,z=coords(d); r=np.sqrt(x*x+y*y+z*z)
 end,last=load(final_output(case,'mhd_w_bcc'))
 assert end['time']==0.2
 assert all(np.isfinite(v).all() for v in last.values())
 assert last['dens'].min()>0 and last['press'].min()>0
 report[name]={'central_initial_density':float(f['dens'][r<2].max()),'final_time':end['time']}
outer=r>=32
for name in fields['original']:
 assert np.array_equal(fields['original'][name][outer],fields['cavity'][name][outer]),name
core=r<2
assert np.max(fields['cavity']['dens'][core]) < .1*np.max(fields['original']['dens'][core])
inside=(r<30)&(fields['cavity']['dens']>1e-7)
assert np.allclose(fields['cavity']['press'][inside],fields['cavity']['dens'][inside]**(5/3),rtol=2e-6)
q=np.minimum(r/32,1)
weight=q**3*(10-15*q+6*q*q)
expected=1e-10+weight*(fields['original']['dens']-1e-10)
assert np.allclose(fields['cavity']['dens'],expected,rtol=3e-6,atol=1e-12)
assert fields['cavity']['dens'][r<8].min()>1e-10
report['smooth_taper_matches_quintic']=True
report['floor_reached_only_at_geometric_origin']=True
report['outer_fields_identical']=True
report['adiabat_preserved']=True
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
