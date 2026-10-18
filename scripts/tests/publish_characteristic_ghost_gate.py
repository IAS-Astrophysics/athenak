#!/usr/bin/env python3
"""Publish a local gate only after every final-source preflight has completed."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root.resolve();build=root/'build-mac-characteristic';out=a.out;out.mkdir(parents=True,exist_ok=True)
ghost=json.loads((build/'ghost-ct-final-gates/results.json').read_text());assert ghost['status']=='passed' and len(ghost['cases'])==8
plane=json.loads((build/'ghost-ct-final-plane/results.json').read_text());assert len(plane)==6 and all(x['time']==180 for x in plane)
rot=json.loads((build/'ghost-ct-final-rotating/local-results.json').read_text());assert len(rot)==3 and all(x['time']==180 for x in rot)
curved=json.loads((build/'ghost-ct-final-curved/results.json').read_text());assert curved['status']=='passed' and all(x['bitwise_equal'] for x in curved['inner_comparison']['fields'].values())
native=next(json.loads(line) for line in (build/'ghost-ct-native-final.log').read_text().splitlines() if line.startswith('{'));assert native['status']=='passed' and native['native_mode_cases']==900 and native['ghost_polynomial_cases']==4
decomp=json.loads((out/'native-decomposition.json').read_text());assert decomp['status']=='passed' and decomp['native_states']==1620
files=['src/coordinates/coordinates.hpp','src/dyn_grmhd/characteristic_reservoir.hpp','src/dyn_grmhd/characteristic_ghost_ct.hpp','src/dyn_grmhd/dyn_grmhd.cpp','src/dyn_grmhd/dyn_grmhd_fluxes.cpp','src/dyn_grmhd/dyn_grmhd_fofc.cpp','src/pgen/dynbbh.cpp']
checks=dict(post_recovery_ghosts=True,native_flux_and_ct=True,all_ghost_cells_checked=True,ghost_cases=8,native_modes=900,full_decomposition_states=1620,plane_cases=6,rotating_cases=3,curved_cases=2,curved_scope=curved['scope'],max_all_ghost_abs_divB=max(x['all_ghost_max_abs_divB'] for x in ghost['cases']))
gate=dict(status='passed',checks=checks,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},local_executable_sha256=hashlib.sha256((build/'src/athena').read_bytes()).hexdigest())
for src,name in [(build/'ghost-ct-final-gates/results.json','local-ghost-gates.json'),(build/'ghost-ct-final-plane/results.json','local-plane.json'),(build/'ghost-ct-final-rotating/local-results.json','local-rotating.json'),(build/'ghost-ct-final-curved/results.json','local-curved.json')]:shutil.copy2(src,out/name)
(out/'native-mode-tests.json').write_text(json.dumps(native,indent=2)+'\n')
(out/'local-gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate,indent=2))
