#!/usr/bin/env python3
"""Coarse curved binary/static-AMR smoke; GPU suite retains full hierarchy."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from characteristic_rotating_cases import prepared_input
from test_binary_zoom import ROOT,load,final_output,inspect
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_grmhd_characteristic import compare_curved
p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
inp=a.out/'input.athinput';inp.write_text(prepared_input((ROOT/'inputs/dyngr/bondi_hierarchy/binary_magnetized_lowres.athinput').read_text()));rows=[];states=[]
for boundary in ('fixed','characteristic'):
 case=a.out/f'curved-amr-{boundary}';case.mkdir(exist_ok=True)
 flags=['mesh_refinement/refinement=static','mesh_refinement/num_levels=3','mesh_refinement/max_nmb_per_rank=256',
 'refined_region1/level=2','refined_region2/level=2','coord/require_resolved_horizon=false',
 'coord/smooth_excision=true','problem/reservoir_test_flat_metric=false','problem/reservoir_test_pulse=0',
 'problem/reservoir_pressure_perturbation=0',f'problem/reservoir_boundary={boundary}',f'problem/reservoir_ghost_ct={str(boundary=="characteristic").lower()}',
 'time/tlim=0.05','time/nlim=100','output1/ghost_zones=false','output2/ghost_zones=false',
 'output1/dt=0.05','output2/dt=0.05','output3/dt=0.05','output4/dt=0','mhd/dyn_flux_team_size=0']
 t=time.monotonic()
 with (case/'run.log').open('w') as log:subprocess.run([str(a.exe.resolve()),'-i',str(inp.resolve()),'-d',str(case.resolve()),'-t','00:03:00',*flags],stdout=log,stderr=subprocess.STDOUT,timeout=190,check=True)
 checks=inspect(case);d,f=load(final_output(case,'mhd_w_bcc'));assert d['time']==.05;assert d['mb_logical'][:,3].max()>d['mb_logical'][:,3].min();states.append((d,f))
 rows.append(dict(boundary=boundary,seconds=time.monotonic()-t,checks=checks));print(boundary,'passed',flush=True)
comparison=compare_curved(a.out);assert all(v['bitwise_equal'] for v in comparison['fields'].values())
(a.out/'results.json').write_text(json.dumps(dict(status='passed',scope='Coarse curved/static-AMR smoke, horizon not resolved; not physical-drain validation',inner_comparison=comparison,cases=rows),indent=2)+'\n')
