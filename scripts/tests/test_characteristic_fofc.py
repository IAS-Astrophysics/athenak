#!/usr/bin/env python3
"""Force every boundary-adjacent FOFC flag and compare first/second trace runs."""
import argparse,subprocess,json
from pathlib import Path
import numpy as np
from characteristic_rotating_cases import prepared_input
from test_binary_zoom import ROOT,load,final_output,inspect
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
(a.out/'input.athinput').write_text(prepared_input((ROOT/'inputs/dyngr/bondi_hierarchy/binary_magnetized_lowres.athinput').read_text()))
checks=[]
for order in (1,3):
 case=a.out/f'order{order}';case.mkdir(exist_ok=True)
 flags=['mesh_refinement/refinement=none','coord/excise=false','coord/require_resolved_horizon=false','problem/single_black_hole=true','problem/reservoir_test_flat_metric=true','problem/corotating=false','problem/frame_omega=0','problem/reservoir_pressure_perturbation=0','problem/reservoir_boundary=characteristic',f'problem/reservoir_trace_order={order}','problem/reservoir_test_pulse=0.01','problem/reservoir_test_pulse_mode=acoustic','problem/reservoir_test_pulse_z=7','mhd/enforce_maximum=true','mhd/dmp_M=0.5','mhd/dyn_flux_team_size=0','time/tlim=0.5','time/nlim=100','output1/dt=0.5','output2/dt=0.5','output3/dt=0.5','output4/dt=0']
 for d in (1,2,3):flags += [f'mesh/nx{d}=8',f'mesh/x{d}min=-8',f'mesh/x{d}max=8']
 with (case/'run.log').open('w') as log:subprocess.run([str(a.exe.resolve()),'-i',str((a.out/'input.athinput').resolve()),'-d',str(case.resolve()),'-t','00:01:00',*flags],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=70)
 checks.append(inspect(case))
da,fa=load(final_output(a.out/'order1','mhd_w_bcc'));db,fb=load(final_output(a.out/'order3','mhd_w_bcc'))
assert da['time']==db['time']==.5
errors={k:float(abs(fa[k]-fb[k]).max()) for k in fa};assert all(x==0 for x in errors.values()),errors
result=dict(status='passed',forced_every_cell=True,first_and_third_order_bitwise_equal=True,max_abs_errors=errors,checks=checks)
(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
