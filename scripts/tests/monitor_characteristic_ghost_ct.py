#!/usr/bin/env python3
"""Read-only one-shot status; never submits, rebuilds or touches other jobs."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);a=p.parse_args();t=a.trial
if not (t/'submission.json').exists():
 print(json.dumps(dict(submission='not-yet-submitted',build_ready=(t/'build-ready.json').exists(),local_gate=(t/'local-gate.json').exists())));raise SystemExit(0)
r=json.loads((t/'submission.json').read_text());print(json.dumps(r,indent=2))
if r['status']=='accepted':
 job=r['stdout'].strip().split()[0];q=subprocess.run(['qstat','-xf',job],capture_output=True,text=True);print(q.stdout,q.stderr)
