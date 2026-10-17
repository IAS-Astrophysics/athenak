#!/usr/bin/env python3
"""Monitor one accepted magnetic pilot chain; optionally continue validated stops.

Runs on the Aurora login host, never evolves fluid there. Each continuation
uses the guarded submission helper. A stop file prevents further submissions;
it does not cancel a running job. No other user's/agent's jobs are changed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from submit_magnetic_continuation import restart_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=int,required=True)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--target',type=float,default=20000.)
    p.add_argument('--poll',type=float,default=50.)
    p.add_argument('--continue-jobs',action='store_true')
    a=p.parse_args();root=a.root.resolve();job=a.job
    assert 0<a.target<=20000 and 1<=a.poll<=60
    lock=(root/'magnetic-monitor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=root/'magnetic-monitor.stop'
    statefile=root/'magnetic-monitor-state.json'
    scripts=root/'source-magnetic/scripts/tests'
    checked=set();last=None
    def run(cmd,timeout=60,check=True,env=None):
        return subprocess.run(cmd,cwd=root,text=True,capture_output=True,timeout=timeout,check=check,env=env)
    def emit(event,**data):print(json.dumps(dict(event=event,job=job,**data)),flush=True)
    while True:
        statefile.write_text(json.dumps({'pid':os.getpid(),'job':job,'target':a.target,
                                        'continue_jobs':a.continue_jobs},indent=2)+'\n')
        if stop.exists():emit('STOP_FILE_PRESENT');return
        try:q=run(['qstat','-xf',str(job)],timeout=30,check=False)
        except subprocess.TimeoutExpired:
            emit('OBSERVATION_TIMEOUT');time.sleep(a.poll);continue
        found=re.search(r'job_state = (\S+)',q.stdout)
        if not found:
            emit('OBSERVATION_UNAVAILABLE',returncode=q.returncode,stderr=q.stderr.strip())
            time.sleep(a.poll);continue
        status=found.group(1);case=root/f'evolution-magnetic-{job}'
        log=case/'rank-0.log'
        progress=[] if not log.exists() else [s for s in log.read_text().splitlines() if s.startswith('elapsed=')][-1:]
        signature=(status,tuple(progress))
        if signature!=last:emit('STATUS',state=status,progress=progress);last=signature
        if status!='F':time.sleep(a.poll);continue
        code=re.search(r'Exit_status = ([-\d]+)',q.stdout)
        if not code or int(code.group(1))!=0:
            emit('FAILED_JOB',exit_status=None if not code else int(code.group(1)));return
        if job not in checked:
            command=['python3.11',str(scripts/'run_bounded.py'),'--seconds','300','--memory-gb','4',
                     '--report',str(case/'validation-resources.json'),'--',
                     '/home/hzhu/pyenv/bin/python',str(scripts/'validate_magnetic_pilot.py'),
                     '--run',str(case),'--out',str(case/'validation.json')]
            analysis_env=os.environ.copy()
            for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                analysis_env[key]='1'
            result=run(command,timeout=330,check=False,env=analysis_env)
            if result.returncode:
                emit('VALIDATION_FAILED',returncode=result.returncode,output=result.stdout[-5000:],stderr=result.stderr[-5000:]);return
            checked.add(job)
            checkpoints=sorted((case/'rst').glob('*.rst'))
            native=restart_state(checkpoints[-1])
            emit('VALIDATED_STOP',checkpoint=str(checkpoints[-1]),native_state=native)
            if native['time']>=a.target-1e-6:
                emit('TARGET_REACHED',time=native['time']);return
        receipt=root/f'magnetic-continuation-{job}.receipt'
        if not receipt.exists():
            if not a.continue_jobs:emit('READY_FOR_CONTINUATION');return
            occupied=run(['qselect','-u','hzhu','-q','debug-scaling']).stdout.strip()
            if occupied:
                emit('QUEUE_OCCUPIED_WAITING',job_ids=occupied.splitlines());time.sleep(a.poll);continue
            if stop.exists():emit('STOP_FILE_PRESENT');return
            result=run(['python3.11',str(scripts/'submit_magnetic_continuation.py'),
                        '--predecessor',str(job),'--target',str(a.target)],timeout=90,check=False)
            if result.returncode:
                emit('SUBMISSION_REQUIRES_REVIEW',output=result.stdout[-5000:],stderr=result.stderr[-5000:]);return
        record=json.loads(receipt.read_text())
        child=re.fullmatch(r'(\d+)\.\S+\s*',record.get('stdout',''))
        if record.get('returncode')!=0 or not child:
            emit('UNKNOWN_OR_REJECTED_RECEIPT',receipt=str(receipt));return
        emit('FOLLOW_ACCEPTED_CHILD',child=int(child.group(1)),receipt=str(receipt))
        job=int(child.group(1));last=None


if __name__=='__main__':main()
