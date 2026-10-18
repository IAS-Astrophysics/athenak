#!/usr/bin/env python3
"""Publish a previously completed local gate only for the exact final Aurora build."""
import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();pending=root/'characteristic-improved-local-gate.pending.json';gate=root/'characteristic-improved-local-gate.json';state=root/'characteristic-improved-gate-state.json'
with (root/'characteristic-improved-gate.started').open('x') as f:f.write('one publication owner\n')
local=json.loads(pending.read_text());assert local['status']=='passed' and local['trace_order']==3
if gate.exists():
 assert json.loads(gate.read_text())==local
 print('Identical gate already published');raise SystemExit(0)
for attempt in range(181):
 ready=root/'build-characteristic/improved-ready.json'
 if ready.exists():
  build=json.loads(ready.read_text())
  if build.get('final_source') and build['trace_order']==3:break
 if attempt==180:raise SystemExit('Final build not ready; gate unpublished and no job submitted by this tool')
 time.sleep(10)
assert build['status']=='built' and build['selective_fofc']
for name,digest in local['source_sha256'].items():assert hashlib.sha256((root/'source-characteristic'/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((root/'build-characteristic/src/athena').read_bytes()).hexdigest()==build['executable_sha256']
pending.replace(gate);state.write_text(json.dumps(dict(status='published',time=time.time(),executable_sha256=build['executable_sha256']),indent=2)+'\n');print(state.read_text())
