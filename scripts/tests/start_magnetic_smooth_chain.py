#!/usr/bin/env python3
"""Wait for the switch test, validate it, then launch the authorized two-cycle chain.

Login-host orchestration only. A stop file disables submissions; accepted jobs
are followed through receipts. Validation/submission failures stop for review.
"""
import fcntl,json,os,re,subprocess,time
from pathlib import Path
root=Path.cwd();scripts=root/'source-magnetic-smooth/scripts/tests'
lock=(root/'magnetic-smooth-bootstrap.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record=json.loads((root/'magnetic-smooth-check.receipt').read_text());match=re.fullmatch(r'(\d+)\.\S+\s*',record.get('stdout',''));assert record.get('returncode')==0 and match
job=int(match.group(1));case=root/f'evolution-magnetic-smooth-check-{job}'
(root/'magnetic-smooth-bootstrap-state.json').write_text(json.dumps({'pid':os.getpid(),'check_job':job},indent=2)+'\n')
def run(cmd,timeout=60,env=None):return subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=timeout,env=env)
def stopped():return (root/'magnetic-smooth-monitor.stop').exists()
while not stopped():
 q=run(['qstat','-xf',str(job)]).stdout
 state=re.search(r'job_state = (\S+)',q);assert state
 if state.group(1)=='F':
  assert re.search(r'Exit_status = 0\s',q),'Switch test failed; inspect evidence before retry'
  break
 time.sleep(50)
else:raise SystemExit('Stop file present; no continuation')
env=os.environ.copy()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
r=run(['python3.11',str(scripts/'run_bounded.py'),'--seconds','300','--memory-gb','4','--report',str(case/'validation-resources.json'),'--','/home/hzhu/pyenv/bin/python',str(scripts/'validate_magnetic_smooth.py'),'--run',str(case),'--out',str(case/'validation.json')],timeout=330,env=env)
print(r.stdout,flush=True)
receipt=root/'magnetic-smooth-anchor.receipt'
while not receipt.exists():
 if stopped():raise SystemExit('Stop file present; no continuation')
 if run(['qselect','-u','hzhu','-q','debug-scaling']).stdout.strip():time.sleep(50);continue
 print(run(['python3.11',str(scripts/'submit_magnetic_smooth.py'),'--predecessor','8938971','--anchor'],timeout=90).stdout,flush=True)
record=json.loads(receipt.read_text());match=re.fullmatch(r'(\d+)\.\S+\s*',record.get('stdout',''));assert record.get('returncode')==0 and match
if stopped():raise SystemExit('Stop file present; accepted receipt retained')
os.execlp('python3.11','python3.11',str(scripts/'watch_magnetic_smooth.py'),'--job',match.group(1),'--continue-jobs')
