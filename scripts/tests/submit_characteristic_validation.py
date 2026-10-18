#!/usr/bin/env python3
"""One-shot capacity submission after isolated build and local packet gates.
No retries or continuations. A preexisting receipt always stops submission.
"""
import argparse,json,subprocess,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();root=a.root.resolve();trial=root/'boundary-grmhd-characteristic-20261018-v1';trial.mkdir(exist_ok=True);receipt=trial/'submission.json'
if receipt.exists():print(receipt.read_text());raise SystemExit('Existing receipt; no duplicate')
for attempt in range(181 if a.wait else 1):
 ready=root/'build-characteristic/final-ready.json';gate=root/'characteristic-local-packet-gate.json'
 if ready.exists() and gate.exists():break
 if not a.wait:raise SystemExit('Build/local packet gate not ready; no submission')
 if attempt==180:raise SystemExit('Timed out waiting for build; no submission')
 time.sleep(10)
build=json.loads(ready.read_text());local=json.loads(gate.read_text());assert build['status']=='built' and build['periodic_ghost_fix'] and local['status']=='passed'
subprocess.run(['sha256sum','-c','build-characteristic/executable.sha256'],cwd=root,check=True)
subprocess.run(['sha256sum','-c','build-characteristic/source.sha256'],cwd=root,check=True)
assert hashlib.sha256((root/'build-characteristic/src/athena').read_bytes()).hexdigest()==build['executable_sha256']
prior=subprocess.run(['qstat','-xf','8941753'],capture_output=True,text=True,check=True).stdout
assert 'Exit_status = 0' in prior and 'job_state = F' in prior,'Predecessor must have finished successfully'
(trial/'predecessor-qstat.txt').write_text(prior)
for src,name in [(root/'aurora_grmhd_characteristic.pbs','job.pbs'),(root/'gpu-test.athinput','input.athinput'),(root/'build-characteristic/source.sha256','source.sha256'),(ready,'build-ready.json'),(gate,'local-gate.json')]:
 (trial/name).write_bytes(src.read_bytes())
s=dict(status='submitting',predecessor='8941753',time=time.time(),queue='capacity',nodes=1,executable_sha256=build['executable_sha256'],sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in trial.iterdir() if f.is_file()})
with receipt.open('x') as f:json.dump(s,f,indent=2)
r=subprocess.run(['qsub','-v',f'CZ_BOUNDARY_TRIAL={trial}',str(trial/'job.pbs')],cwd=root,text=True,capture_output=True)
s.update(status='accepted' if r.returncode==0 else 'rejected',stdout=r.stdout,stderr=r.stderr,returncode=r.returncode)
receipt.write_text(json.dumps(s,indent=2)+'\n');print(receipt.read_text())
raise SystemExit(r.returncode)
