#!/usr/bin/env python3
"""One accepted-or-unknown receipt blocks every duplicate submission."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();root=a.root.resolve();trial=root/'boundary-characteristic-rotating-20261018-v3';receipt=trial/'submission.json'
if receipt.exists():raise SystemExit('Existing receipt; no duplicate: '+receipt.read_text())
def read_ready(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
for attempt in range(241 if a.wait else 1):
 build=read_ready(trial/'build-ready.json');gate=read_ready(trial/'local-gate.json')
 if build and gate and build.get('final_source'):break
 if not a.wait or attempt==240:raise SystemExit('Build/local gate not ready; no submission')
 time.sleep(10)
if receipt.exists():raise SystemExit('Another owner wrote a receipt; stop')
assert build['status']=='built' and build['ghost_ct'] and build['final_source'] and gate['status']=='passed'
subprocess.run(['sha256sum','-c',str(trial/'source.sha256')],cwd=root,check=True)
for line in (trial/'source.sha256').read_text().splitlines():
 digest,name=line.split();assert gate['source_sha256'][name.removeprefix('source-characteristic/')]==digest,name
exe=root/'build-characteristic/src/athena';assert hashlib.sha256(exe.read_bytes()).hexdigest()==build['executable_sha256']
prior=subprocess.run(['qstat','-xf','8942061'],capture_output=True,text=True,check=True).stdout;assert 'job_state = F' in prior and 'Exit_status = 0' in prior
(trial/'executable.sha256').write_text(f'{build["executable_sha256"]}  {exe}\n')
s=dict(status='submitting',time=time.time(),queue='capacity',nodes=1,predecessor='8942061',ghost_ct=True,executable_sha256=build['executable_sha256'],sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in trial.iterdir() if f.is_file()})
with receipt.open('x') as f:json.dump(s,f,indent=2)
r=subprocess.run(['qsub','-v',f'CZ_BOUNDARY_TRIAL={trial}',str(trial/'job.pbs')],cwd=root,capture_output=True,text=True)
s.update(status='accepted' if r.returncode==0 else 'rejected',stdout=r.stdout,stderr=r.stderr,returncode=r.returncode);receipt.write_text(json.dumps(s,indent=2)+'\n');print(receipt.read_text());raise SystemExit(r.returncode)
