#!/usr/bin/env python3
"""Idempotent two-cycle smooth-excision chain, anchored at finest event 14.

Run from the isolated Aurora experiment root; debug-scaling only. All completed
jobs need validation before submission. Unknown qsub outcomes require review.
"""
import argparse,datetime,hashlib,json,re,subprocess
from pathlib import Path
from magnetic_smooth_state import checkpoint_state

TARGET=29500.
ANCHOR_JOB=8938971
ANCHOR_TIME=9903.615850407483

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--predecessor',type=int,required=True);p.add_argument('--anchor',action='store_true');p.add_argument('--check-switch',action='store_true');a=p.parse_args();root=a.root.resolve()
 assert not a.check_switch or a.anchor
 target=ANCHOR_TIME+0.5 if a.check_switch else TARGET
 receipt=root/('magnetic-smooth-check.receipt' if a.check_switch else 'magnetic-smooth-anchor.receipt' if a.anchor else f'magnetic-smooth-continuation-{a.predecessor}.receipt')
 if receipt.exists():print(receipt.read_text());return
 def run(cmd):return subprocess.run(cmd,cwd=root,capture_output=True,text=True,check=True,timeout=60)
 q=run(['qstat','-xf',str(a.predecessor)]).stdout
 assert re.search(r'job_state = F\s',q) and re.search(r'Exit_status = 0\s',q)
 case=root/f'evolution-magnetic-{a.predecessor}' if a.anchor else root/f'evolution-magnetic-smooth-{a.predecessor}'
 v=json.loads((case/'validation.json').read_text());assert len(v['ranks'])==12 and len(v['hosts'])==1 and v['max_scaled_divb']<1e-9 and v['conservative_sum_bytes']<30e9
 checkpoint=sorted((case/'rst').glob('*.rst'))[-1];state=checkpoint_state(checkpoint);last=v['states'][-1]
 assert state['cycle']==last['cycle'] and state['blocks']==last['blocks'] and abs(state['time']-last['time'])<max(1e-4,5e-6*state['time'])
 par=state.pop('parameters')
 assert par['amr_criterion1/bondi_mhd_schedule']=='true' and float(par['amr_criterion1/residence_scale'])==1
 assert par['problem/initial_condition']=='bondi_reservoir' and float(par['problem/beta_inf'])==1
 if a.anchor:
  assert a.predecessor==ANCHOR_JOB and state['id']==14 and state['zone']==0 and abs(state['time']-ANCHOR_TIME)<1e-9
  assert par['coord/smooth_excision'] in ('0','false')
 else:
  assert par['coord/smooth_excision'] in ('1','true') and par['cyclic_zoom/restart_reset_finest_dwell'] in ('0','false')
  if state['time']>=TARGET-1e-6:
   assert state['id']==42 and state['zone']==0 and state['time']-(state['next_time']-250)>=200
   print('Two smooth cycles and final comparison window completed:',state);return
  assert v['termination']=='wall clock limit'
 if a.anchor and not a.check_switch:
  check=json.loads((root/'magnetic-smooth-check.receipt').read_text())
  match=re.fullmatch(r'(\d+)\.\S+\s*',check.get('stdout',''));assert check.get('returncode')==0 and match
  checkjob=int(match.group(1));cq=run(['qstat','-xf',str(checkjob)]).stdout
  assert re.search(r'job_state = F\s',cq) and re.search(r'Exit_status = 0\s',cq)
  cv=json.loads((root/f'evolution-magnetic-smooth-check-{checkjob}'/'validation.json').read_text())
  assert len(cv['ranks'])==12 and cv['conservative_sum_bytes']<30e9 and cv['max_scaled_divb']<1e-9
  assert cv['residence_reset_records'] and cv['smooth_checkpoint']['zone']==0 and cv['smooth_checkpoint']['id']==14
  assert cv['smooth_checkpoint']['time']>=ANCHOR_TIME+0.5-1e-8
 occupied=run(['qselect','-u','hzhu','-q','debug-scaling']).stdout.strip()
 if occupied:raise SystemExit('Queue occupied; no submission: '+occupied)
 for f in ('build-magnetic-smooth/source.sha256','build-magnetic-smooth/executable.sha256'):
  run(['sha256sum','-c',f])
 sha=hashlib.sha256()
 with checkpoint.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
 env=f'CZ_RESTART={checkpoint},CZ_TLIM={target:.17g}' + (',CZ_SMOOTH_ANCHOR=1' if a.anchor else '') + (',CZ_CHECK_SWITCH=1' if a.check_switch else '')
 command=['qsub','-v',env,'source-magnetic-smooth/scripts/tests/aurora_bondi_magnetic_smooth.pbs']
 record=dict(predecessor=a.predecessor,anchor=a.anchor,restart=str(checkpoint),restart_state=state,restart_sha256=sha.hexdigest(),target=target,check_switch=a.check_switch,required_final_event=42,minimum_final_finest_phase=200,attempt_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command)
 with receipt.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
 r=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=60);record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);receipt.write_text(json.dumps(record,indent=2)+'\n');print(receipt.read_text())
 if r.returncode:raise SystemExit(r.returncode)
if __name__=='__main__':main()
