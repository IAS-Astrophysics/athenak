#!/usr/bin/env python3
"""Submit exactly one debug job only when this user's debug queue is empty."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    receipt=root/'submission.json'
    if receipt.exists():
        print(receipt.read_text())
        raise SystemExit('Existing submission receipt: inspect before any resubmission')
    for path in ('build-sycl2025/src/athena','kokkos_memory_guard.so',
                 'campaign/matched.athinput','campaign/upstream.athinput'):
        if not (root/path).is_file():
            raise SystemExit('Missing campaign artifact: '+path)
    existing=subprocess.check_output(['qselect','-u',os.environ['USER'],'-q','debug'],
                                     universal_newlines=True).strip()
    if existing:
        print('Debug queue occupied; no job submitted: '+existing)
        raise SystemExit(75)
    # PBS also enforces max_run=1 and queued_jobs_threshold=1 for this user.
    command=['qsub','-q','debug','-o',str(root/'results/pbs.log'),
             str(root/'source/scripts/tests/bondi_debug.pbs')]
    receipt.write_text(json.dumps({'status':'submitting','command':command},indent=2)+'\n')
    proc=subprocess.run(command,cwd=str(root),stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,universal_newlines=True)
    result={'status':'submitted' if proc.returncode == 0 else 'rejected',
            'command':command,'job_id':proc.stdout.strip(),'stderr':proc.stderr,
            'returncode':proc.returncode}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
