#!/usr/bin/env python3
"""One-shot gated capacity submission for improved rotating-boundary tests."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();root=a.root.resolve();trial=root/'boundary-characteristic-rotating-20261018-v2';trial.mkdir(exist_ok=True);receipt=trial/'submission.json'
if receipt.exists():raise SystemExit('Existing receipt, no duplicate: '+receipt.read_text())
ready=root/'build-characteristic/improved-ready.json';gate=root/'characteristic-improved-local-gate.json'
for attempt in range(181 if a.wait else 1):
 if ready.exists() and gate.exists():break
 if not a.wait or attempt==180:raise SystemExit('Build/local gate not ready; no submission')
 time.sleep(10)
build=json.loads(ready.read_text());local=json.loads(gate.read_text());assert build['status']=='built' and build['selective_fofc'] and build['trace_order']==3 and build.get('final_source') and local['status']=='passed'
manifest=root/'build-characteristic/improved-source.sha256';subprocess.run(['sha256sum','-c',str(manifest)],cwd=root,check=True)
exe=root/'build-characteristic/src/athena';assert hashlib.sha256(exe.read_bytes()).hexdigest()==build['executable_sha256']
expected=local['source_sha256']
for line in manifest.read_text().splitlines():
 digest,name=line.split();assert expected[name.removeprefix('source-characteristic/')]==digest,name
prior=subprocess.run(['qstat','-xf','8941903'],capture_output=True,text=True,check=True).stdout
assert 'job_state = F' in prior and 'Exit_status = 0' in prior
(trial/'predecessor-qstat.txt').write_text(prior)
for source,name in [(root/'aurora_characteristic_rotating.pbs','job.pbs'),(root/'rotating-test.athinput','input.athinput'),(root/'characteristic_rotating_cases.py','characteristic_rotating_cases.py'),(root/'prepare_characteristic_control.py','prepare_characteristic_control.py'),(ready,'build-ready.json'),(gate,'local-gate.json'),(manifest,'source.sha256')]:
 (trial/name).write_bytes(source.read_bytes())
(trial/'executable.sha256').write_text(f'{build["executable_sha256"]}  {exe}\n')
s=dict(status='submitting',time=time.time(),queue='capacity',nodes=1,predecessor='8941903',executable_sha256=build['executable_sha256'],sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in trial.iterdir() if f.is_file()})
with receipt.open('x') as f:json.dump(s,f,indent=2)
r=subprocess.run(['qsub','-v',f'CZ_BOUNDARY_TRIAL={trial}',str(trial/'job.pbs')],cwd=root,capture_output=True,text=True)
s.update(status='accepted' if r.returncode==0 else 'rejected',stdout=r.stdout,stderr=r.stderr,returncode=r.returncode);receipt.write_text(json.dumps(s,indent=2)+'\n');print(receipt.read_text());raise SystemExit(r.returncode)
